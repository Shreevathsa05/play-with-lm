"""Compatibility + capacity gate for fine-tune jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

from src.core.config import get_settings
from src.finetuning_modules.compatibility.estimates import CapacityEstimate, estimate_capacity, parse_param_billions
from src.finetuning_modules.compatibility.recipes import RECIPE_IDS, get_recipe_spec, suggest_fallback

# Architectures we treat as non-LM multimodal / unsupported in v1.
_REJECT_ARCH_KEYWORDS = (
    "vision",
    "vlm",
    "clip",
    "whisper",
    "audio",
    "diffusion",
    "stable-diffusion",
    "llava",
    "idefics",
)


class GateDecision(str, Enum):
    ACCEPT = "accept"
    WAIT = "wait"
    REJECT = "reject"
    SUGGEST = "suggest"  # accept only if user confirms suggested recipe — surface, do not auto-swap


@dataclass
class GateResult:
    decision: GateDecision
    reason: str
    recipe: str
    estimate: Optional[CapacityEstimate] = None
    suggested_recipe: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok_to_spawn(self) -> bool:
        return self.decision == GateDecision.ACCEPT


class CompatibilityGate:
    """Validate model class, recipe, Unsloth support hint, and capacity."""

    def __init__(
        self,
        machine_max_vram_mb: Optional[int] = None,
        machine_max_ram_mb: Optional[int] = None,
        max_model_params_b: Optional[float] = None,
        unsupported_arch_keywords: Sequence[str] = _REJECT_ARCH_KEYWORDS,
    ):
        settings = get_settings()
        self.machine_max_vram_mb = machine_max_vram_mb or settings.machine_max_vram_mb
        self.machine_max_ram_mb = machine_max_ram_mb or settings.machine_max_ram_mb
        self.max_model_params_b = max_model_params_b or settings.max_model_params_b
        self.unsupported_arch_keywords = tuple(k.lower() for k in unsupported_arch_keywords)

    def evaluate(
        self,
        *,
        base_model_id: str,
        recipe: str,
        model_params_b: float | int | str | None = None,
        architecture: Optional[str] = None,
        unsloth_supported: bool = True,
        max_seq_length: int = 2048,
        batch_size: int = 2,
        free_vram_mb: Optional[int] = None,
        free_ram_mb: Optional[int] = None,
        allow_suggest: bool = True,
    ) -> GateResult:
        model_id = (base_model_id or "").strip()
        if not model_id:
            return GateResult(GateDecision.REJECT, "base_model_id is required", recipe or "")

        try:
            spec = get_recipe_spec(recipe)
        except ValueError as exc:
            return GateResult(GateDecision.REJECT, str(exc), recipe or "")

        arch_blob = f"{architecture or ''} {model_id}".lower()
        for kw in self.unsupported_arch_keywords:
            if kw in arch_blob:
                return GateResult(
                    GateDecision.REJECT,
                    f"Model/architecture looks non-LM multimodal ('{kw}'); v1 supports language models only.",
                    spec.recipe_id,
                    details={"architecture": architecture, "base_model_id": model_id},
                )

        if not unsloth_supported:
            return GateResult(
                GateDecision.REJECT,
                "Model architecture is not supported by Unsloth for v1.",
                spec.recipe_id,
                details={"base_model_id": model_id},
            )

        try:
            params_b = parse_param_billions(model_params_b)
        except (TypeError, ValueError):
            return GateResult(
                GateDecision.REJECT,
                "model_params_b must be a numeric size such as '270m' or '1b'.",
                spec.recipe_id,
                details={"model_params_b": model_params_b},
            )
        if params_b > self.max_model_params_b:
            return GateResult(
                GateDecision.REJECT,
                (
                    f"Model size {params_b:g}B exceeds this worker's {self.max_model_params_b:g}B limit. "
                    "Choose a model up to 1B parameters (for example Gemma 270M or Llama 3.2 1B)."
                ),
                spec.recipe_id,
                details={"params_b": params_b, "max_model_params_b": self.max_model_params_b},
            )

        estimate = estimate_capacity(
            spec.recipe_id,
            model_params_b,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )

        if estimate.peak_vram_mb > self.machine_max_vram_mb or estimate.peak_ram_mb > self.machine_max_ram_mb:
            suggested = suggest_fallback(spec.recipe_id) if allow_suggest else None
            if suggested and suggested != spec.recipe_id:
                alt = estimate_capacity(
                    suggested,
                    model_params_b,
                    max_seq_length=max_seq_length,
                    batch_size=batch_size,
                )
                if alt.peak_vram_mb <= self.machine_max_vram_mb and alt.peak_ram_mb <= self.machine_max_ram_mb:
                    return GateResult(
                        GateDecision.SUGGEST,
                        (
                            f"Requested recipe '{spec.recipe_id}' exceeds machine max "
                            f"(need ~{estimate.peak_vram_mb}MB VRAM / {estimate.peak_ram_mb}MB RAM; "
                            f"max {self.machine_max_vram_mb}/{self.machine_max_ram_mb}). "
                            f"Suggested recipe: '{suggested}'."
                        ),
                        spec.recipe_id,
                        estimate=estimate,
                        suggested_recipe=suggested,
                        details={"alt_estimate": alt.__dict__},
                    )
            return GateResult(
                GateDecision.REJECT,
                (
                    f"Job exceeds machine maximum capacity "
                    f"(need ~{estimate.peak_vram_mb}MB VRAM / {estimate.peak_ram_mb}MB RAM; "
                    f"max {self.machine_max_vram_mb}/{self.machine_max_ram_mb})."
                ),
                spec.recipe_id,
                estimate=estimate,
            )

        # Fits machine — check current free resources if provided
        if free_vram_mb is not None and free_vram_mb < estimate.peak_vram_mb:
            return GateResult(
                GateDecision.WAIT,
                (
                    f"Insufficient free VRAM right now "
                    f"(need ~{estimate.peak_vram_mb}MB, free {free_vram_mb}MB). Waiting FCFS."
                ),
                spec.recipe_id,
                estimate=estimate,
                details={"free_vram_mb": free_vram_mb},
            )
        if free_ram_mb is not None and free_ram_mb < estimate.peak_ram_mb:
            return GateResult(
                GateDecision.WAIT,
                (
                    f"Insufficient free RAM right now "
                    f"(need ~{estimate.peak_ram_mb}MB, free {free_ram_mb}MB). Waiting FCFS."
                ),
                spec.recipe_id,
                estimate=estimate,
                details={"free_ram_mb": free_ram_mb},
            )

        return GateResult(
            GateDecision.ACCEPT,
            "Compatible and capacity available.",
            spec.recipe_id,
            estimate=estimate,
            details={"recipe_spec": spec.__dict__},
        )


__all__ = [
    "CompatibilityGate",
    "GateDecision",
    "GateResult",
    "RECIPE_IDS",
]
