"""Helpers to build Unsloth/HF generate + teacher-forced reference NLL callables."""

from __future__ import annotations

from typing import Any, Callable, Optional


def make_generate_fn(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 64,
    enable_inference: Optional[Callable[[Any], None]] = None,
) -> Callable[[str], str]:
    """
    Return generate(prompt) -> text using the given model/tokenizer.

    enable_inference: optional Unsloth FastLanguageModel.for_inference(model)
    """

    def _generate(prompt: str) -> str:
        if enable_inference is not None:
            enable_inference(model)
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif hasattr(next(model.parameters()), "device"):
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Decode only newly generated tokens when possible
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = output_ids[0][prompt_len:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return _generate


def make_reference_loss_fn(
    model: Any,
    tokenizer: Any,
) -> Callable[[str, str], float]:
    """
    Teacher-forced mean CE loss of reference tokens given prompt.
    Requires torch at runtime (GPU path). Unit tests inject callables instead.
    """

    def _loss(prompt: str, reference: str) -> float:
        import torch
        import torch.nn.functional as F

        text = f"{prompt}{reference}"
        encoded = tokenizer(text, return_tensors="pt")
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[-1]

        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Next-token prediction: logits[:-1] vs labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Only score reference portion (tokens after prompt)
        # labels index i corresponds to predicting token at i+1
        start = max(prompt_len - 1, 0)
        if start >= shift_labels.shape[1]:
            return 0.0

        ref_logits = shift_logits[:, start:, :]
        ref_labels = shift_labels[:, start:]
        loss = F.cross_entropy(
            ref_logits.view(-1, ref_logits.size(-1)),
            ref_labels.view(-1),
            reduction="mean",
        )
        return float(loss.item())

    return _loss
