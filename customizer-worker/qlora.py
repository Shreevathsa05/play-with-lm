import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_scheduler,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.optim import AdamW, SGD
from tqdm.auto import tqdm
import math
from dotenv import load_dotenv

# Load environment variables (like HF_TOKEN)
load_dotenv()

class TextDataset(Dataset):
    """
    Dataset to handle tokenized texts for Causal Language Modeling.
    """
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # For Causal LM, labels are shifted inside the model
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def qlora_finetune(
    model_name_or_path: str,
    train_conversations: list[list[dict]],
    output_dir: str = "./qlora_outputs",
    
    # 1. Quantization Specific Parameters (BitsAndBytes)
    load_in_4bit: bool = True,
    load_in_8bit: bool = False, 
    bnb_4bit_quant_type: str = "nf4", # 'nf4' (normalized float 4) is highly recommended for QLoRA
    bnb_4bit_use_double_quant: bool = True, # Saves even more memory
    
    # 2. LoRA Specific Parameters
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # "all-linear" targets
    lora_bias: str = "none",
    
    # 3. Training Batch & Epoch Parameters
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    max_seq_len: int = 512,
    seed: int = 42,
    
    # 4. Optimization Parameters
    optimizer_name: str = "adamw",
    learning_rate: float = 2e-4, 
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3, # Notice: commonly lowered for QLoRA (0.3 instead of 1.0)
    
    # 5. Learning Rate Scheduler Parameters
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.03, 
    
    # 6. Hardware and Floating Point Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bf16: bool = False, # Compute dtype for active model pass during 4-bit frozen eval
    fp16: bool = False,
    
    # 7. Logging and Saving 
    log_interval: int = 10,
    save_strategy: str = "epoch",
    
    # 8. Auth
    hf_token: str = None
):
    """
    Quantized Parameter-Efficient Fine-Tuning (QLoRA).
    Uses BitsAndBytes to aggressively compress the backbone model to 4-bit or 8-bit, 
    while training high-precision LoRA adapters on top.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Loading tokenizer & backbone model: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    compute_dtype = torch.float32
    if bf16:
        compute_dtype = torch.bfloat16
    elif fp16:
        compute_dtype = torch.float16
        
    # --- QUANTIZATION CONFIGURATION ---
    print(f"Setting up BitsAndBytes Quantization Config...")
    bnb_config = None
    if load_in_4bit or load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype # High precision calculation datatype while weights sit in 4bit
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device, # bitsandbytes intrinsically supports 'auto' or device allocation
        token=hf_token
    )
    
    # Validate and Prepare model for Kbit training BEFORE injecting adapters!
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # --- LoRA INJECTION ---
    print(f"Injecting LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=lora_bias
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train() 
    
    print("Formatting conversations via chat template...")
    train_texts = [
        tokenizer.apply_chat_template(conv, tokenize=False)
        for conv in train_conversations
    ]
    
    print(f"Preparing dataset with max length = {max_seq_len}...")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_seq_len)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=per_device_train_batch_size, 
        collate_fn=default_data_collator
    )
    
    # Configure Optimizer 
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "weight_decay": weight_decay}
    ]
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer")
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    actual_warmup_steps = warmup_steps
    if warmup_ratio > 0.0:
        actual_warmup_steps = int(max_train_steps * warmup_ratio)
        
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=actual_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Standard PyTorch loop, bitsandbytes natively handles weights underneath
    print(f"Starting QLoRA Training on {device}...")
    total_steps = 0
    
    for epoch in range(num_train_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_train_epochs} ---")
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            # Batch inputs to device. Note: weights might already be on GPU via `device_map`
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps 
            
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_steps += 1
                
                if total_steps % log_interval == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    real_loss = loss.item() * gradient_accumulation_steps
                    progress_bar.set_postfix({"loss": f"{real_loss:.4f}", "lr": f"{current_lr:.2e}"})
                    
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Average epoch {epoch+1} loss: {avg_loss:.4f}")
        
        if save_strategy == "epoch":
            epoch_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            print(f"Saving QLoRA checkpoint adapters to {epoch_dir}")
            model.save_pretrained(epoch_dir, safe_serialization=False)
            tokenizer.save_pretrained(epoch_dir)
            
    print(f"Training complete. Saving final QLoRA adapters to {output_dir}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

# ----------------------------------------------------------TEST---------------------------------------------------------------

def test_finetune_qlora():
    """
    Test function for QLoRA fine-tuning on the google/gemma-3-270m-it model.
    By loading in 4-bit, we severely drop VRAM usage!
    """
    dummy_chat_data = [
        [
            {"role": "user", "content": "Explain deep learning."},
            {"role": "assistant", "content": "Deep learning is a subset of ML utilizing neural networks."}
        ],
        [
            {"role": "user", "content": "Classify the email as SPAM or HAM and explain why.\n\nEmail: Congratulations! You have won a 10,000 Amazon gift card. Click here to claim now!"},
            {"role": "assistant", "content": "SPAM because it's an unsolicited message offering an unrealistic reward and urging immediate action, which are classic phishing signs."}
        ],
        [
            {"role": "user", "content": "Write a python snippet."},
            {"role": "assistant", "content": "print('hello world')"}
        ]
    ]
    
    HF_TOKEN = os.environ.get("HF_TOKEN", None) 
    
    print("Initiating QLoRA test on google/gemma-3-270m-it using 4-bit precision...")
    
    qlora_finetune(
        model_name_or_path="google/gemma-3-270m-it", 
        train_conversations=dummy_chat_data,
        output_dir="./gemma-3-270m-qlora",
        
        # Quantization Setup
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4",      # Normalized Float 4 formulation
        bnb_4bit_use_double_quant=True, # Second-pass quantization
        
        # LoRA
        lora_r=16,
        lora_alpha=32,
        
        # Batch & Epochs
        num_train_epochs=2,
        per_device_train_batch_size=1,  # Mini batch size
        gradient_accumulation_steps=2,
        max_seq_len=128,              
        
        # Optimizer
        optimizer_name="adamw",
        learning_rate=2e-4, 
        weight_decay=0.01,
        
        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False, 
        fp16=False,
        
        log_interval=1,
        save_strategy="epoch",
        hf_token=HF_TOKEN
    )

if __name__ == "__main__":
    test_finetune_qlora()
