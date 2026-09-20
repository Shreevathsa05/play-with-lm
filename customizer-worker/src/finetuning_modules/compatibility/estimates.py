"""VRAM / RAM capacity estimates for the compatibility gate."""

from __future__ import annotations

from dataclasses import dataclass

from src.finetuning_modules.compatibility.recipes import get_recipe_spec


@dataclass(frozen=True)
class CapacityEstimate:
    peak_vram_mb: int
    peak_ram_mb: int
    params_b: float
    recipe: str


def parse_param_billions(model_size_hint: float | int | str | None) -> float:
    """
    Normalize a model size hint to billions of parameters.
    Accepts 0.27, 270e6, '270m', '1b', '1.5B', etc.
    """
    if model_size_hint is None:
        return 1.0
    if isinstance(model_size_hint, (int, float)):
        val = float(model_size_hint)
        # Treat large raw counts as absolute params
        if val > 100:
            return val / 1e9
        return val

    text = str(model_size_hint).strip().lower().replace(",", "")
    if text.endswith("b"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)


def estimate_capacity(
    recipe_id: str,
    model_params_b: float | int | str | None,
    max_seq_length: int = 2048,
    batch_size: int = 2,
) -> CapacityEstimate:
    """
    Rough peak VRAM/RAM estimate.

    Formula (MB): params_b * recipe_factor * 1024 * seq_scale * batch_scale
    where seq_scale grows mildly with context and batch_scale with micro-batch.
    """
    spec = get_recipe_spec(recipe_id)
    params_b = max(parse_param_billions(model_params_b), 0.01)
    seq_scale = max(max_seq_length / 2048.0, 0.5)
    batch_scale = max(batch_size / 2.0, 0.5)

    peak_vram_mb = int(params_b * spec.vram_factor * 1024 * seq_scale * batch_scale)
    # System RAM for datasets/checkpoints — coarse floor
    peak_ram_mb = max(int(peak_vram_mb * 1.5), 2048)

    return CapacityEstimate(
        peak_vram_mb=peak_vram_mb,
        peak_ram_mb=peak_ram_mb,
        params_b=params_b,
        recipe=spec.recipe_id,
    )
