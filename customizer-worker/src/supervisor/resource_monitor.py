"""Resource monitoring for FCFS capacity waits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union


def _probe_cuda_vram() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes / 1024**2), int(total_bytes / 1024**2)


def _probe_system_ram() -> tuple[int, int]:
    import psutil

    memory = psutil.virtual_memory()
    return int(memory.available / 1024**2), int(memory.total / 1024**2)


@dataclass(frozen=True)
class ResourceSnapshot:
    free_vram_mb: int
    free_ram_mb: int
    total_vram_mb: int
    total_ram_mb: int


ProbeResult = Union[int, tuple[int, int]]


class ResourceMonitor:
    """
    Reads free VRAM/RAM.

    Uses optional callables so unit tests inject values; production can wire
    torch.cuda.mem_get_info / psutil without importing them at module load.
    """

    def __init__(
        self,
        *,
        vram_probe: Optional[Callable[[], ProbeResult]] = None,
        ram_probe: Optional[Callable[[], ProbeResult]] = None,
        fallback_free_vram_mb: int = 4096,
        fallback_free_ram_mb: int = 16384,
        total_vram_mb: int = 4096,
        total_ram_mb: int = 16384,
    ):
        self._vram_probe = vram_probe or _probe_cuda_vram
        self._ram_probe = ram_probe or _probe_system_ram
        self._fallback_free_vram_mb = fallback_free_vram_mb
        self._fallback_free_ram_mb = fallback_free_ram_mb
        self._total_vram_mb = total_vram_mb
        self._total_ram_mb = total_ram_mb

    def snapshot(self) -> ResourceSnapshot:
        free_vram = self._fallback_free_vram_mb
        free_ram = self._fallback_free_ram_mb
        total_vram = self._total_vram_mb
        total_ram = self._total_ram_mb

        if self._vram_probe is not None:
            try:
                probed = self._vram_probe()
                if isinstance(probed, tuple) and len(probed) == 2:
                    free_vram, total_vram = int(probed[0]), int(probed[1])
                else:
                    free_vram = int(probed)
            except Exception:
                pass

        if self._ram_probe is not None:
            try:
                probed = self._ram_probe()
                if isinstance(probed, tuple) and len(probed) == 2:
                    free_ram, total_ram = int(probed[0]), int(probed[1])
                else:
                    free_ram = int(probed)
            except Exception:
                pass

        return ResourceSnapshot(
            free_vram_mb=free_vram,
            free_ram_mb=free_ram,
            total_vram_mb=total_vram,
            total_ram_mb=total_ram,
        )
