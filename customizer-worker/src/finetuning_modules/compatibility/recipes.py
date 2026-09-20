"""v1 recipe IDs and load flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

RECIPE_IDS = (
    "fullparams",
    "lora",
    "qlora_4bit",
    "qlora_8bit",
    "embedding",
)

# Rough params→GB multipliers by recipe (peak training estimate, not exact).
# Tuned for planning gates on small/dev GPUs; override estimates via job payload later if needed.
_RECIPE_VRAM_FACTOR = {
    "fullparams": 16.0,   # weights + grads + optimizer ~ high
    "lora": 6.0,
    "qlora_4bit": 3.0,
    "qlora_8bit": 4.0,
    "embedding": 5.0,
}


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_peft: bool = False
    vram_factor: float = 6.0


def get_recipe_spec(recipe_id: str) -> RecipeSpec:
    rid = (recipe_id or "").strip().lower()
    if rid not in RECIPE_IDS:
        raise ValueError(f"Unknown recipe '{recipe_id}'. Supported: {list(RECIPE_IDS)}")

    if rid == "fullparams":
        return RecipeSpec(rid, use_peft=False, vram_factor=_RECIPE_VRAM_FACTOR[rid])
    if rid == "lora":
        return RecipeSpec(rid, use_peft=True, vram_factor=_RECIPE_VRAM_FACTOR[rid])
    if rid == "qlora_4bit":
        return RecipeSpec(rid, load_in_4bit=True, use_peft=True, vram_factor=_RECIPE_VRAM_FACTOR[rid])
    if rid == "qlora_8bit":
        return RecipeSpec(rid, load_in_8bit=True, use_peft=True, vram_factor=_RECIPE_VRAM_FACTOR[rid])
    # embedding
    return RecipeSpec(rid, use_peft=True, vram_factor=_RECIPE_VRAM_FACTOR[rid])


def suggest_fallback(recipe_id: str) -> Optional[str]:
    """Suggest a lower-VRAM recipe when the requested one may not fit. Never auto-apply."""
    rid = (recipe_id or "").strip().lower()
    order = ["fullparams", "embedding", "lora", "qlora_8bit", "qlora_4bit"]
    if rid not in order:
        return "qlora_4bit"
    idx = order.index(rid)
    if idx >= len(order) - 1:
        return None
    return order[idx + 1]
