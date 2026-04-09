import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

# Load environment variables (like HF_TOKEN)
load_dotenv()

def load_and_run_quantized_model(
    model_name_or_path: str,
    prompt: str | list,
    
    # Quantization Constraints
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",  # "fp4" or "nf4"
    bnb_4bit_use_double_quant: bool = True,
    
    # Hardware Config
    computational_dtype: torch.dtype = torch.bfloat16, # The precision used during the forward pass computations
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    
    # Generation Constraints
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    
    hf_token: str = None,
    save_path: str = None
):
    """
    Stand-alone script focusing purely on Model Quantization.
    It takes an existing heavy model and compresses it into 4-bit or 8-bit RAM states 
    to heavily reduce memory footprint without using LoRA or any fine-tuning.
    """
    
    print(f"Setting up BitsAndBytes Quantization Config for {model_name_or_path}...")
    
    if load_in_4bit and load_in_8bit:
        raise ValueError("You can only choose to load in 4-bit OR 8-bit, not both.")
        
    bnb_config = None
    if load_in_4bit or load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=computational_dtype
        )
        
    print(f"Loading '{model_name_or_path}' with quantized config (4-bit: {load_in_4bit}, 8-bit: {load_in_8bit})...")
    
    # Note: When using quantization_config, passing device_map="auto" is recommended
    # to let transformers and bitsandbytes securely place quantized tensors.
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto" if "cuda" in device else None,
        token=hf_token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Check actual memory footprint!
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    print(f"Model successfully loaded! Current VRAM usage: ~{gpu_mem:.2f} MB")
        
    if save_path:
        print(f"\nSaving the quantized model and tokenizer to '{save_path}'...")
        # safe_serialization=True explicitly uses safetensors which is ideal for quantized states
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        print("Quantized model saved successfully! You can rapidly load it later without BitsAndBytesConfig.")
        
    print("\nExecuting inference on the quantized model...")
    if isinstance(prompt, list):
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n---------------- OUTPUT ----------------")
    print(response)
    print("----------------------------------------")


def test_quantization():
    """
    Test function to run the quantized pipeline independently.
    """
    messages = [
        {"role": "user", "content": "Explain deep learning."}
    ]
    
    HF_TOKEN = os.environ.get("HF_TOKEN", None) 
    
    print("Initiating strictly-Quantization logic on google/gemma-3-270m-it...")
    
    load_and_run_quantized_model(
        model_name_or_path="google/gemma-3-270m-it",
        prompt=messages,
        
        # Strictly Quantization configuration (No LoRA training parameters)
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True,
        computational_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        
        max_new_tokens=50,
        hf_token=HF_TOKEN,
        save_path="gemma-3-270m-it-4bit"
    )

if __name__ == "__main__":
    test_quantization()
