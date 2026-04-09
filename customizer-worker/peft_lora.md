# Parameter-Efficient Fine-Tuning (PEFT) LoRA Guide

This guide breaks down Low-Rank Adaptation (LoRA) and how to configure its specific levers for training a causal language model like `google/gemma-3-270m-it`.

## What is LoRA?
Instead of touching all of the pre-calculated weights in the massive neural network matrix (like in Full Parameter Fine-Tuning), **PEFT LoRA** freezes the original backbone network completely.

It then attaches ("injects") tiny trainable "adapter" matrices to specific layers alongside the frozen ones. During training, only these tiny adapters learn. 
- **The Result:** Instead of requiring ~8GB memory to train a 270m model, or ~60GB to train an 8 Billion parameter model, you often only need a tiny fraction of that (often just 6-12 GB for a 7B model).
- **The Output:** The output model weight size is only a few Megabytes (e.g., 10MB to 50MB) rather than Gigabytes.

---

## Modifiable LoRA Parameters Deep Dive

When calling `peft_lora_finetune()`, you pass in standard hardware parameters, but you'll notice new ones specifically controlling the Low-Rank Adaptation injection (`LoraConfig`):

### 1. The Rank (`lora_r`)
- **What it does:** The rank controls the size of the adapter matrices that are injected into the model.
- **Default:** `8` or `16`.
- **How to tune it:** 
  - `4` to `8` is excellent for general language chat/instruct tasks where you're just shifting the style or teaching it to format outputs.
  - `16` to `32` (or even `128`) is used for injecting incredibly complex logic like high-level coding abilities or completely foreign languages. 
  - *Keep in mind: A higher rank increases VRAM usage and output file size.*

### 2. The Alpha (`lora_alpha`)
- **What it does:** The scaling factor dictating how "loudly" the LoRA weights influence the established frozen network outputs.
- **Default:** `16` or `32`.
- **How to tune it:** A golden rule is that **`lora_alpha` should typically be 2x the value of `lora_r`**. So if `lora_r=8`, make `lora_alpha=16`. If `r=32`, make `alpha=64`. This mathematically balances the adapter signal.

### 3. Target Modules (`target_modules`)
- **What it does:** Determines exactly which linear layers in the Transformer network receive the LoRA adapters.
- **Default:** `["q_proj", "v_proj"]`.
- **How to tune it:**
  - Standard minimal tuning attaches adapters to the Attention mechanism: `q_proj` (query) and `v_proj` (value).
  - Robust/High-Performance tuning (becoming the industry standard) targets **all** linear layers. For Gemma/Llama models, this means: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. You can often pass the string `"all-linear"` automatically. Doing this drastically improves the knowledge retention.

### 4. LoRA Dropout (`lora_dropout`)
- **What it does:** Randomly zeros out adapter neurons during training to prevent the LoRA from blindly memorizing the training dataset (overfitting).
- **Default:** `0.05` to `0.1`.
- **How to tune it:** 0.05 is the sweet spot. If the dataset is incredibly small (e.g., < 100 entries), you might bump this to `0.1` to enforce generalization.

---

## The Learning Rate Difference
In our `fpft.py` guide, the learning rate was tiny (`2e-5`). 
Because LoRA initializes its `B` matrices to zero and only trains a tiny pool of parameters, **it requires a much higher learning rate** to shift the mathematical scale!

For LoRA fine-tuning, the standard LR starts around **`2e-4`** or **`3e-4`** (roughly 10x higher than FPFT).

## Running the Process

1. Ensure `peft` is installed alongside PyTorch dependencies:
```bash
pip install peft
```

2. Execute the script:
```bash
python peft_lora.py
```
