# Full Parameter Fine-Tuning (FPFT) Guide

This guide covers the concepts, parameters, and best practices for Full Parameter Fine-Tuning (FPFT) of causal language models like `google/gemma-3-270m-it` using native PyTorch.

## What is Full Parameter Fine-Tuning?
Unlike Parameter Efficient Fine-Tuning (PEFT) methods like LoRA which freeze the model backbone and only train a small percentage of injected adapter weights, **Full Parameter Fine-Tuning** trains **all** the weights in the mathematical foundation of the model. 

While it is much slower, requires significantly more VRAM, and is prone to catastrophic forgetting if not done right, FPFT can sometimes achieve higher performance when vast domain shifts are occurring or when building foundational logic capabilities.

---

## Modifiable Parameters Deep Dive

When tweaking the `full_parameter_finetune` function, here's what every argument does and how to think about setting it:

### 1. Training Setup Parameters
- **`num_train_epochs`** (Default: 3): The number of complete passes through your entire training dataset. Usually, LLM FPFT requires fewer epochs (1 to 3) to prevent overfitting.
- **`per_device_train_batch_size`** (Default: 4): How many sequence samples to process on the GPU at one time. Higher uses more VRAM. Reduce this if you hit Memory `CUDA Out Of Memory (OOM)`.
- **`gradient_accumulation_steps`** (Default: 1): If your GPU can't fit a large batch size, use this. It accumulates gradients over multiple smaller batches before running the optimization step, simulating a larger batch size without consuming extra memory. `Effective Batch Size = batch_size * grad_acc_steps * num_gpus`.
- **`max_seq_len`** (Default: 512): The maximum context window of tokens during training. Higher is better for chat bots, but takes quadratic processing power/memory based on attention mechanics.

### 2. Optimization Parameters
- **`optimizer_name`** (Default: "adamw"): AdamW is the standard for Transformers. It handles weight decay more effectively than standard Adam.
- **`learning_rate`** (Default: 5e-5): The step size used to update weights. For FPFT, keep this low (e.g. `2e-5` to `5e-5`). If it is too high, the pre-trained weights will be destroyed (catastrophic forgetting).
- **`weight_decay`** (Default: 0.01 or 0.0): Helps prevent overfitting by penalizing large weights.
- **`max_grad_norm`** (Default: 1.0): Gradient clipping threshold. Sometimes loss spikes natively in Transformer models. This clips massive gradient vectors to a maximum of 1.0.

### 3. Learning Rate Scheduler Parameters
- **`lr_scheduler_type`** (Default: "cosine"): Modulates the learning rate over time. `"cosine"` warms up and then decays beautifully over a curve. `"linear"` decays straight down.
- **`warmup_steps` / `warmup_ratio`** (Default: 0): Instead of starting instantly at your max `learning_rate`, it slowly crawls up to it over X steps (or ratio % of total steps). Typically 5-10% of total steps is a good warmup. Essential in transformer training to avoid early destabilization.

### 4. Hardware Precision
- **`fp16` / `bf16`**: 
  - Floating Point 32 (FP32) represents standard PyTorch decimal precision. Natively, neural networks use this but it eats VRAM.
  - Using **`bf16` (BFloat16)** or **`fp16`** casts the calculations to 16-bit half-precision. It saves nearly 50% Memory and runs up to 3x faster on Ampere/Hopper GPUs. 
  - **Best Practice:** If your GPU supports it (Nvidia RTX 3000 series / A100 or newer), **always use `bf16=True`** over `fp16` for modern LLMs like Gemma, Llama 2/3, and Mistral, as it avoids overflow/underflow issues inherently present in `fp16`.

## Hardware Requirements for Gemma 3 (270M)
The requested model runs at around 270 million parameters. In Half-Precision (FP16/BF16):
- **Model weights:** ~500 MB
- **AdamW Optimizer States:** ~1.5 GB
- **Gradients and Activations:** ~1 GB to 4 GB depending on Batch Size and Sequence Length.
- **Total VRAM required for FPFT:** Around 4GB to 8GB. This can easily be trained on consumer graphics cards (e.g., RTX 3060, RTX 4070). 
*(Note: Attempting FPFT on a 7B or 8B model would require ~50GB+ of VRAM, which requires multiple A100s or moving to PEFT/LoRA).*

## Running the Process

1. Ensure dependencies are installed:
```bash
pip install torch transformers accelerate datasets sentencepiece tqdm
```

2. Gemma authentication:
Gemma models usually require acceptance of their license on the Hugging Face Hub. Once accepted, login via terminal:
```bash
huggingface-cli login
```
*(Or input your raw string directly into `HF_TOKEN` in the script).*

3. Execute the fine-tuning script:
```bash
python fpft.py
```
