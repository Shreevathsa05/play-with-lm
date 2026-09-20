"""Pure scoring helpers for batch eval (no GPU required)."""

from __future__ import annotations

from typing import Any, Optional, Sequence


def nll_improvement(loss_base: float, loss_ft: float, eps: float = 1e-8) -> float:
    """
    Relative improvement: (loss_base - loss_ft) / max(loss_base, eps).
    Positive => fine-tuned model assigns higher probability to the reference.
    """
    return (loss_base - loss_ft) / max(loss_base, eps)


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def exact_match(prediction: str, reference: str, strip: bool = True) -> bool:
    if prediction is None or reference is None:
        return False
    a = prediction.strip() if strip else prediction
    b = reference.strip() if strip else reference
    return a == b


def field_match(prediction: Any, reference: Any, field: str) -> bool:
    """Compare a nested field when both sides are dict-like; else exact string match on field extraction."""
    if isinstance(prediction, dict) and isinstance(reference, dict):
        return prediction.get(field) == reference.get(field)
    return False


def accuracy(matches: Sequence[bool]) -> Optional[float]:
    if not matches:
        return None
    return sum(1 for m in matches if m) / len(matches)


def truncate_prompts(prompts: Sequence[Any], max_prompts: int = 1000) -> tuple[list[Any], bool]:
    """Return (truncated_list, was_truncated)."""
    items = list(prompts)
    if len(items) <= max_prompts:
        return items, False
    return items[:max_prompts], True


def build_eval_report(
    *,
    job_uuid: str,
    n_prompts: int,
    truncated: bool,
    base_losses: Sequence[float],
    ft_losses: Sequence[float],
    base_accuracy: Optional[float],
    ft_accuracy: Optional[float],
    samples: Sequence[dict[str, Any]],
    max_samples: int = 5,
) -> dict[str, Any]:
    paired = [(b, f) for b, f in zip(base_losses, ft_losses)]
    improvements = [nll_improvement(b, f) for b, f in paired]

    report: dict[str, Any] = {
        "job_uuid": job_uuid,
        "n_prompts": n_prompts,
        "truncated": truncated,
        "max_prompts": 1000,
        "reference_nll": {
            "mean_loss_base": mean(list(base_losses)),
            "mean_loss_ft": mean(list(ft_losses)),
            "mean_improvement": mean(improvements),
            "n_scored": len(paired),
        },
        "task_accuracy": {
            "base": base_accuracy,
            "ft": ft_accuracy,
            "delta": None
            if base_accuracy is None or ft_accuracy is None
            else (ft_accuracy - base_accuracy),
        },
        "samples": list(samples)[:max_samples],
    }
    return report
