import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_scheduler,
    default_data_collator
)
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

def full_parameter_finetune(
    model_name_or_path: str,
    train_conversations: list[list[dict]],
    output_dir: str = "./outputs",
    
    # 1. Training Batch & Epoch Parameters
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    max_seq_len: int = 512,
    seed: int = 42,
    
    # 2. Optimization Parameters
    optimizer_name: str = "adamw", # Options: 'adamw' or 'sgd'
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    
    # 3. Learning Rate Scheduler Parameters
    lr_scheduler_type: str = "cosine", # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0, # If > 0, overrides warmup_steps
    
    # 4. Hardware and Floating Point Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fp16: bool = False,
    bf16: bool = False,
    
    # 5. Logging and Saving 
    log_interval: int = 10,
    save_strategy: str = "epoch", # Options: 'epoch' or 'no'
    
    # 6. Auth
    hf_token: str = None
):

    """
    Full Parameter Fine-Tuning for Causal Language Models using native PyTorch.
    Exposes all possible parameter tweakings for direct customization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Loading tokenizer and model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    
    # Handle missing pad token (common in standard Llama/Gemma models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = torch.float32
    if bf16:
        # BF16 is highly recommended for modern architectures if hardware supports it (Ampere+)
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        token=hf_token
    )
    model.to(device)
    model.train() # Make sure model is in training mode!
    
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
    print(f"Setting up optimizer ({optimizer_name}) and scheduler ({lr_scheduler_type})...")
    # Separate parameters that shouldn't undergo weight decay
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
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
    
    # Enable gradient scaling if fp16 is active to prevent underflow
    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16) 
    
    print(f"Starting Training on {device}...")
    total_steps = 0
    
    for epoch in range(num_train_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_train_epochs} ---")
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass context
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                
                # Normalize loss based on accumulation steps
                loss = loss / gradient_accumulation_steps 
            
            # Backward pass
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Accumulate gradients or trigger optimizer step
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Unscale prior to clipping
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_steps += 1
                
                # Logging
                if total_steps % log_interval == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    # Log real loss (undo accumulation scaling)
                    real_loss = loss.item() * gradient_accumulation_steps
                    progress_bar.set_postfix({"loss": f"{real_loss:.4f}", "lr": f"{current_lr:.2e}"})
                    
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Average epoch {epoch+1} loss: {avg_loss:.4f}")
        
        if save_strategy == "epoch":
            epoch_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            print(f"Saving checkpoint to {epoch_dir}")
            model.save_pretrained(epoch_dir, safe_serialization=False)
            tokenizer.save_pretrained(epoch_dir)
            
    print(f"Training complete. Saving final model to {output_dir}")
    # Move model to CPU and clear CUDA cache to free up VRAM/System RAM before final save
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

# ----------------------------------------------------------TEST---------------------------------------------------------------

def test_finetune_gemma():
    """
    Test function for fine-tuning the google/gemma-3-270m-it model.
    It runs using a dummy conversation dataset and passes highly configurable arguments.
    """
    # Sample tiny dataset for testing functionality (Normally you'd load a robust dataset here)
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
    
    # Pass your huggingface token if you need gated access (Gemma models require HF gate approval)
    HF_TOKEN = os.environ.get("HF_TOKEN", None) 
    
    print("Initiating full parameter fine-tuning test on google/gemma-3-270m-it...")
    
    full_parameter_finetune(
        # Model
        model_name_or_path="google/gemma-3-270m-it", 
        train_conversations=dummy_chat_data,
        output_dir="./gemma-3-270m-finetuned",
        
        # Batch & Epochs
        num_train_epochs=2,
        per_device_train_batch_size=2, # Reduce if you encounter CUDA OOM
        gradient_accumulation_steps=2,
        max_seq_len=128,              # Token length
        
        # Optimizer
        optimizer_name="adamw",
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Scheduler
        lr_scheduler_type="cosine",
        warmup_steps=2,
        warmup_ratio=0.0,
        
        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False, # Gemma is optimized for BF16
        fp16=False,
        
        # Logging & Saving
        log_interval=1,
        save_strategy="epoch",
        
        # Access
        hf_token=HF_TOKEN
    )

if __name__ == "__main__":
    test_finetune_gemma()
