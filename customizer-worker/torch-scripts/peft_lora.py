import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_scheduler,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW, SGD
from tqdm.auto import tqdm
import math
from dotenv import load_dotenv

# Load environment variables (like HF_TOKEN) from .env
load_dotenv()

class TextDataset(Dataset):
    """
    A simple dataset class to handle tokenized texts for Causal Language Modeling.
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
        # For Causal LM, the labels are shifted inside the model
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def peft_lora_finetune(
    model_name_or_path: str,
    train_conversations: list[list[dict]],
    output_dir: str = "./peft_outputs",
    
    # 1. LoRA Specific Parameters
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] = ["q_proj", "v_proj"], # "all-linear" targets all dense layers 
    lora_bias: str = "none", # "none", "all", or "lora_only"
    
    # 2. Training Batch & Epoch Parameters
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    max_seq_len: int = 512,
    seed: int = 42,
    
    # 3. Optimization Parameters
    optimizer_name: str = "adamw", # Options: 'adamw' or 'sgd'
    learning_rate: float = 2e-4, # Notice: LoRA usually uses ~10x higher LR than FPFT
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    
    # 4. Learning Rate Scheduler Parameters
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.05, 
    
    # 5. Hardware and Floating Point Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fp16: bool = False,
    bf16: bool = False,
    
    # 6. Logging and Saving 
    log_interval: int = 10,
    save_strategy: str = "epoch", # Options: 'epoch' or 'no'
    
    # 7. Auth
    hf_token: str = None
):
    """
    Parameter-Efficient Fine-Tuning (PEFT) using LoRA for Causal Language Models.
    Exposes both training details and specific Low-Rank Adaptation injection properties.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Loading tokenizer & backbone model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = torch.float32
    if bf16:
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        token=hf_token
    )
    
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
    # This freezes the pre-trained weights & mounts trainable adapter layers on top
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.to(device)
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
    
    # Configure Optimizer (Only adapter weights require grad)
    print(f"Setting up optimizer ({optimizer_name}) and scheduler ({lr_scheduler_type})...")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": weight_decay,
        }
    ]
    
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Configure Scheduler
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
    
    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16) 
    
    print(f"Starting PEFT/LoRA Training on {device}...")
    total_steps = 0
    
    for epoch in range(num_train_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_train_epochs} ---")
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps 
            
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                scaler.step(optimizer)
                scaler.update()
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
            print(f"Saving checkpoint to {epoch_dir}")
            model.save_pretrained(epoch_dir, safe_serialization=False)
            tokenizer.save_pretrained(epoch_dir)
            
    print(f"Training complete. Saving final LoRA adapters to {output_dir}")
    # Clear cache and transfer to CPU to free up memory
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

def test_finetune_lora():
    """
    Test function for PEFT LoRA fine-tuning on the google/gemma-3-270m-it model.
    """
    dummy_chat_data = [
        [
            {"role": "user", "content": "Explain deep learning."},
            {"role": "assistant", "content": "Deep learning is a subset of ML utilizing neural networks."}
        ],
        [
            {"role": "user", "content": "Classify the email as SPAM or HAM and explain why.\n\nEmail: Congratulations! You have won a ₹10,000 Amazon gift card. Click here to claim now!"},
            {"role": "assistant", "content": "SPAM because it is an unsolicited message offering an unrealistic reward and urging immediate action, which are classic phishing signs."}
        ],
        [
            {"role": "user", "content": "Write a python snippet."},
            {"role": "assistant", "content": "print('hello world')"}
        ]
    ]
    
    HF_TOKEN = os.environ.get("HF_TOKEN", None) 
    
    print("Initiating PEFT LoRA fine-tuning test on google/gemma-3-270m-it...")
    
    peft_lora_finetune(
        model_name_or_path="google/gemma-3-270m-it", 
        train_conversations=dummy_chat_data,
        output_dir="./gemma-3-270m-lora",
        
        # LoRA
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Target specific attention modules for Gemma
        
        # Batch & Epochs
        num_train_epochs=2,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=2,
        max_seq_len=128,              
        
        # Optimizer
        optimizer_name="adamw",
        learning_rate=2e-4, # Higher LR for LoRA (e.g. 2e-4 vs 2e-5)
        weight_decay=0.01,
        
        # Scheduler
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False, 
        fp16=False,
        
        # Logging & Saving
        log_interval=1,
        save_strategy="epoch",
        hf_token=HF_TOKEN
    )

if __name__ == "__main__":
    test_finetune_lora()
