# Pure Model Quantization Guide

This guide covers how to compress and load a large language model via **Quantization**, completely separate from any fine-tuning paradigms like LoRA.

## Why Quantize a Model?
Large language models are traditionally stored in `fp16` (16-bit precision) or `fp32` (32-bit precision). 
- A 7 Billion parameter model running in 16-bit natively requires **14 GB of VRAM** just to sit on your graphics card.
- By quantizing that model to 8-bit or 4-bit representations, you drastically slash the memory requirement down to **~7 GB (8-bit)** or **~4 GB (4-bit)**.

This allows massive models (like Llama-3 8B or Mixtral) to run fluid inference directly on consumer gaming GPUs (like an RTX 3060 or 4070) with negligible loss in reasoning ability.

---

## Modifiable Quantization Configs 
In `model_quantization.py`, we implement `BitsAndBytesConfig`. Here controls the levers for raw quantization:

### 1. `load_in_4bit` versus `load_in_8bit`
- **`load_in_4bit = True`**: Extremely aggressive. Cuts memory size by ~70-75%. Highly recommended for models above 7B parameters.
- **`load_in_8bit = True`**: Milder compression. Cuts memory size by exactly 50%. The model retains slightly higher precision in its responses, useful if generating tightly factual or heavily coded responses.

### 2. `bnb_4bit_quant_type`
- When using `load_in_4bit`, this determines how the mathematical distributions are squashed.
- **`nf4` (Normalized Float 4):** Always use this. It is structurally engineered to wrap around the typical Gaussian curve of neural network weights, severely reducing precision-loss compared to standard floating-point truncation. 

### 3. `bnb_4bit_use_double_quant`
- By setting this to `True`, the library runs a second layer of quantization on the *metadata constants* of the first quantization. It's essentially "free" memory optimization, saving an extra ~0.4 GB of VRAM per 7B parameters without sacrificing execution speed.

### 4. `computational_dtype`
- Even though the static weights take up only 4-bits of space, the processor still needs room to think! During a forward-pass calculation, BitsAndBytes unpacks the chunk of the weights back to a high-precision state to do the raw math. 
- Setting this to `torch.bfloat16` prevents mathematical overflow bugs and retains optimal hardware speed on Nvidia RTX-3000 series (Ampere) or newer GPUs.

## Running It

Ensure `bitsandbytes` is installed via:
```bash
pip install bitsandbytes
```

Run test inference directly:
```bash
python model_quantization.py
```
