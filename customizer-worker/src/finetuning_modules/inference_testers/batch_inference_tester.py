"""Batch inference + reference-based scoring after fine-tune."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from src.finetuning_modules.inference_testers.scoring import (
    accuracy,
    build_eval_report,
    exact_match,
    truncate_prompts,
)

MAX_PROMPTS = 1000


class BatchInferenceTester:
    """
    Runs bounded batch inference for base vs fine-tuned models.

    Callers inject generate/loss callables so unit tests need no GPU.
    Production wiring passes Unsloth/HF generate + teacher-forced NLL helpers.
    """

    def __init__(self, max_prompts: int = MAX_PROMPTS, max_samples: int = 5):
        self.max_prompts = max_prompts
        self.max_samples = max_samples

    def run(
        self,
        *,
        job_uuid: str,
        prompts: Sequence[dict[str, Any]],
        generate_base: Callable[[str], str],
        generate_ft: Callable[[str], str],
        reference_loss_base: Optional[Callable[[str, str], float]] = None,
        reference_loss_ft: Optional[Callable[[str, str], float]] = None,
        accuracy_field: Optional[str] = None,
    ) -> dict[str, Any]:
        items, truncated = truncate_prompts(prompts, self.max_prompts)

        base_losses: list[float] = []
        ft_losses: list[float] = []
        base_matches: list[bool] = []
        ft_matches: list[bool] = []
        samples: list[dict[str, Any]] = []

        for item in items:
            prompt = item.get("prompt") if isinstance(item, dict) else str(item)
            reference = item.get("reference") if isinstance(item, dict) else None
            if prompt is None:
                continue

            t0 = time.perf_counter()
            base_out = generate_base(prompt)
            base_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            ft_out = generate_ft(prompt)
            ft_ms = (time.perf_counter() - t1) * 1000

            if reference is not None and reference_loss_base and reference_loss_ft:
                base_losses.append(float(reference_loss_base(prompt, reference)))
                ft_losses.append(float(reference_loss_ft(prompt, reference)))

            if reference is not None:
                if accuracy_field and isinstance(reference, dict):
                    # expect generations to be parsed externally; fall back to string match
                    base_matches.append(False)
                    ft_matches.append(False)
                else:
                    base_matches.append(exact_match(base_out, str(reference)))
                    ft_matches.append(exact_match(ft_out, str(reference)))

            if len(samples) < self.max_samples:
                samples.append(
                    {
                        "prompt": prompt,
                        "reference": reference,
                        "base_generation": base_out,
                        "ft_generation": ft_out,
                        "base_latency_ms": round(base_ms, 2),
                        "ft_latency_ms": round(ft_ms, 2),
                    }
                )

        return build_eval_report(
            job_uuid=job_uuid,
            n_prompts=len(items),
            truncated=truncated,
            base_losses=base_losses,
            ft_losses=ft_losses,
            base_accuracy=accuracy(base_matches) if base_matches else None,
            ft_accuracy=accuracy(ft_matches) if ft_matches else None,
            samples=samples,
            max_samples=self.max_samples,
        )
