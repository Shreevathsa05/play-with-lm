"""Measured worker capabilities; never infer a GPU from configuration alone."""

from __future__ import annotations

import importlib.metadata
import subprocess
from typing import Any


OBJECTIVE_IMPORTS = {
    "pretrain": ("torch", "transformers", "datasets"),
    "sft": ("torch", "trl", "unsloth"),
    "cpt": ("torch", "trl", "unsloth"),
    "dpo": ("torch", "trl"),
    "kto": ("torch", "trl"),
    "reward_model": ("torch", "trl"),
    "ppo": ("torch", "trl"),
    "grpo": ("torch", "trl"),
    "embedding": ("torch", "sentence_transformers", "unsloth"),
}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe() -> dict[str, Any]:
    """Return runtime facts used for admission and certification evidence."""
    packages = {name: _version(name) for name in {p for values in OBJECTIVE_IMPORTS.values() for p in values}}
    cuda: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch

        cuda["torch_version"] = torch.__version__
        cuda["available"] = bool(torch.cuda.is_available())
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            cuda["devices"].append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_vram_mb": free // (1024 * 1024),
                "total_vram_mb": total // (1024 * 1024),
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
            })
    except Exception as exc:
        cuda["error"] = str(exc)

    objectives = {
        name: {
            "available": cuda["available"] and all(packages.get(p) for p in required),
            "certified": False,
            "missing": [p for p in required if not packages.get(p)],
        }
        for name, required in OBJECTIVE_IMPORTS.items()
    }
    return {"hardware": cuda, "packages": packages, "objectives": objectives}


def nvidia_smi() -> str | None:
    """Best-effort driver evidence for support bundles, never an admission source."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
