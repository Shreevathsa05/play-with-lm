"""FCFS supervisor: gate → wait for free resources → spawn subprocess."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.finetuning_modules.compatibility import CompatibilityGate, GateDecision, GateResult
from src.supervisor.job_spawner import JobSpawner, SpawnResult
from src.supervisor.resource_monitor import ResourceMonitor, ResourceSnapshot
from src.utils.logcarrier import LogCarrier


@dataclass
class SupervisedRun:
    gate: GateResult
    spawn: Optional[SpawnResult] = None
    waited_sec: float = 0.0


class FCFSSupervisor:
    def __init__(
        self,
        gate: Optional[CompatibilityGate] = None,
        monitor: Optional[ResourceMonitor] = None,
        spawner: Optional[JobSpawner] = None,
        poll_interval_sec: float = 2.0,
        max_wait_sec: float = 3600.0,
    ):
        self.gate = gate or CompatibilityGate()
        self.monitor = monitor or ResourceMonitor()
        self.spawner = spawner or JobSpawner()
        self.poll_interval_sec = poll_interval_sec
        self.max_wait_sec = max_wait_sec

    def run_job(
        self,
        job_payload: dict[str, Any],
        *,
        status_callback: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> SupervisedRun:
        job_uuid = str(job_payload.get("job_uuid") or job_payload.get("jobId") or "unknown")
        log = LogCarrier(job_uuid)
        recipe = job_payload.get("recipe", "")
        objective = str((job_payload.get("plan") or {}).get("objective") or job_payload.get("objective") or "sft").lower()
        dry_run = bool(job_payload.get("dry_run") or job_payload.get("dryRun"))
        if not dry_run:
            from src.capabilities import probe
            capability = probe().get("objectives", {}).get(objective)
            if capability is not None and not capability["available"]:
                return SupervisedRun(gate=GateResult(
                    GateDecision.REJECT,
                    f"Objective '{objective}' runtime unavailable: missing {capability['missing']}", recipe,
                    details={"objective": objective, "capability": capability},
                ))

        if dry_run:
            # Simulations do not import a training runtime, load a model, or allocate
            # GPU/RAM.  They must not remain in the real-training capacity queue.
            gate_result = GateResult(
                GateDecision.ACCEPT,
                "Dry run bypasses runtime and capacity admission.",
                recipe,
                details={"dry_run": True},
            )
        else:
            snap = self.monitor.snapshot()
            gate_result = self.gate.evaluate(
                base_model_id=job_payload.get("base_model_id") or job_payload.get("baseModelId") or "",
                recipe=recipe,
                model_params_b=job_payload.get("model_params_b") or job_payload.get("modelParamsB"),
                architecture=job_payload.get("architecture"),
                unsloth_supported=bool(job_payload.get("unsloth_supported", True)),
                max_seq_length=int(job_payload.get("max_seq_length") or job_payload.get("maxSeqLength") or 2048),
                batch_size=int(job_payload.get("batch_size") or job_payload.get("batchSize") or 2),
                free_vram_mb=snap.free_vram_mb,
                free_ram_mb=snap.free_ram_mb,
            )
        log.info("gate_decision", decision=gate_result.decision.value, reason=gate_result.reason)

        if gate_result.decision in (GateDecision.REJECT, GateDecision.SUGGEST):
            return SupervisedRun(gate=gate_result)

        waited = 0.0
        if gate_result.decision == GateDecision.WAIT and status_callback is not None:
            status_callback("WAITING")
        while gate_result.decision == GateDecision.WAIT:
            if should_cancel is not None and should_cancel():
                return SupervisedRun(gate=GateResult(GateDecision.REJECT, "Cancellation requested before start.", recipe), waited_sec=waited)
            if waited >= self.max_wait_sec:
                gate_result = GateResult(
                    GateDecision.REJECT,
                    f"Timed out waiting for free capacity after {waited:.0f}s.",
                    gate_result.recipe,
                    estimate=gate_result.estimate,
                )
                log.error("wait_timeout", waited_sec=waited)
                return SupervisedRun(gate=gate_result, waited_sec=waited)

            time.sleep(self.poll_interval_sec)
            waited += self.poll_interval_sec
            snap = self.monitor.snapshot()
            gate_result = self.gate.evaluate(
                base_model_id=job_payload.get("base_model_id") or job_payload.get("baseModelId") or "",
                recipe=recipe,
                model_params_b=job_payload.get("model_params_b") or job_payload.get("modelParamsB"),
                architecture=job_payload.get("architecture"),
                unsloth_supported=bool(job_payload.get("unsloth_supported", True)),
                max_seq_length=int(job_payload.get("max_seq_length") or job_payload.get("maxSeqLength") or 2048),
                batch_size=int(job_payload.get("batch_size") or job_payload.get("batchSize") or 2),
                free_vram_mb=snap.free_vram_mb,
                free_ram_mb=snap.free_ram_mb,
            )
            log.info(
                "wait_poll",
                decision=gate_result.decision.value,
                free_vram_mb=snap.free_vram_mb,
                waited_sec=waited,
            )

        if status_callback is not None:
            status_callback("RUNNING")
        timeout = job_payload.get("timeout_sec") or job_payload.get("timeoutSec")
        spawn = self.spawner.spawn(
            job_payload,
            timeout_sec=float(timeout) if timeout else None,
            heartbeat_callback=(lambda: status_callback("RUNNING")) if status_callback else None,
            should_cancel=should_cancel,
        )
        log.info("spawn_finished", returncode=spawn.returncode, pid=spawn.pid)
        return SupervisedRun(gate=gate_result, spawn=spawn, waited_sec=waited)
