# QLoRA (Quantized Low-Rank Adaptation) Guide

This guide breaks down **QLoRA**, the absolute golden standard for running fine-tunes on massive LLMs (like Llama-3 8B or Gemma 7B) using consumer-grade hardware. 

## What is QLoRA Constraints vs LoRA?
Standard LoRA freezes the original model in 16-bit precision. However, a 7B parameter model sitting in 16-bit precision requires ~14 GB of VRAM *just to load into memory*.
**QLoRA (Quantized LoRA)** solves this by compressing the rigid, frozen backbone of the model down to 4-bit or 8-bit using `bitsandbytes`, effectively shrinking a 14 GB model down to roughly **4-5 GB**. 

You then place 16-bit (high-precision) trainable adapter modules on top of that compressed model.

---

## Modifiable QLoRA Parameters Deep Dive

In `qlora_finetune()`, we import `BitsAndBytesConfig` and `prepare_model_for_kbit_training`. Here's what the crucial quantization parameters do:

### 1. The Core Bit-Depth (`load_in_4bit` & `load_in_8bit`)
- **What it does:** Explicitly loads the model weights directly formatted natively in 4-bit or 8-bit precision representations.
- **How to tune it:** 
  - Standard QLoRA relies intrinsically on `load_in_4bit=True`.
  - `load_in_8bit=True` is faster and yields slightly closer-to-original mathematical accuracy but inherently uses double the VRAM of 4-bit.

### 2. The Quantization Format Type (`bnb_4bit_quant_type`)
- **What it does:** Dictates the distribution mapping scale of how floats are stored mathematically in the bit stream.
- **Types:** `"fp4"` (Float 4) or `"nf4"` (Normalized Float 4). 
- **How to tune it:** Always use **`nf4`**. It's fundamentally built for normally-distributed neural network weights and maintains accuracy immensely better than FP4.

### 3. Double Quantization (`bnb_4bit_use_double_quant`)
- **What it does:** During quantization, bitsandbytes creates "quantization constants" that themselves take up RAM. Double quantization quantizes those constants too!
- **How to tune it:** Keep it **`True`**. It saves nearly 0.4 GB of VRAM per 7B parameters with no computational hit.

### 4. Gradient Norm Clipping (`max_grad_norm`)
- **What it does:** We used `1.0` in FPFT because the weights are mathematically loose. In QLoRA, adapter weights are volatile, and quantized network distributions can throw massive gradient shocks. 
- **How to tune it:** Limit it tightly to **`0.3`**.

---

## Important Architectural Note!
Before calling `get_peft_model()`, our script forcefully calls **`model = prepare_model_for_kbit_training(model)`**. 
If you skip this step, the quantized representations will fail massively during backpropagation because their underlying data types will cast incompatibly. This utility wraps gradients in safety hooks so that 4-bit weights gradient outputs correctly bridge into your 16-bit LoRA matrices securely!

## Hardware Support Checklist
Because bitsandbytes leverages heavily optimized low-level CUDA binaries, QLoRA is explicitly tethered to:
- NVIDIA GPUs.
- Windows platforms might require custom pre-compiled dll files of bitsandbytes (though recent `pip install bitsandbytes` updates fully support Windows seamlessly!).

## Execution

Ensure bitsandbytes is cleanly mapped:
```bash
pip install bitsandbytes
```

Run test fine-tuning:
```bash
python qlora.py
```
