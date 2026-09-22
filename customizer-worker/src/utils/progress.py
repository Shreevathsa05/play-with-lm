"""Best-effort finetuning progress webhooks for the manager UI."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from src.core.config import get_settings
from src.utils.logcarrier import LogCarrier

logger = logging.getLogger(__name__)

_LAST_POST: dict[str, float] = {}
_EVENT_PHASES = {
    "job_received": "Worker received job from queue",
    "pipeline_start": "Starting fine-tune pipeline",
    "importing_training_stack": "Importing training stack (Torch/Unsloth)",
    "dataset_loading": "Loading dataset from storage",
    "dataset_loaded": "Dataset loaded from storage",
    "model_loading": "Loading base model",
    "model_loaded": "Base model loaded",
    "baseline_eval_complete": "Baseline evaluation complete",
    "train_config": "Training config applied",
    "train_progress": "Training in progress",
    "train_complete": "Model training finished",
    "target_loss": "Loss target reached",
    "loss_drop": "Loss drop target reached",
    "eval_complete": "Post-training evaluation complete",
    "hf_push_complete": "Model published to Hugging Face",
    "pipeline_complete": "Pipeline finished successfully",
    "pipeline_failed": "Pipeline failed",
    "dry_run_pipeline": "Running simulation (dry run)",
    "scratch_wiped": "Temporary workspace cleaned up",
    "heartbeat": "Worker heartbeat",
}


def event_phase(record: dict[str, Any]) -> str:
    name = str(record.get("event") or "log")
    base = _EVENT_PHASES.get(name, name.replace("_", " ").capitalize())
    if name == "train_progress":
        step = record.get("step")
        total = record.get("total")
        loss = record.get("loss")
        stop = record.get("stop_reason")
        parts = [base]
        if stop == "target_loss":
            parts = ["Loss target reached"]
        elif stop == "loss_drop":
            parts = ["Loss drop target reached"]
        if step is not None and total:
            parts.append(f"step {step}/{total}")
        elif step is not None:
            parts.append(f"step {step}")
        if loss is not None:
            parts.append(f"loss {loss:.4f}" if isinstance(loss, (int, float)) else f"loss {loss}")
        return " · ".join(parts)
    if name == "pipeline_start":
        recipe = record.get("recipe")
        if recipe:
            return f"{base} ({recipe})"
    return base


def progress_notify(
    job_uuid: str,
    status: str = "RUNNING",
    *,
    phase: Optional[str] = None,
    event: Optional[dict[str, Any]] = None,
    patch: Optional[dict[str, Any]] = None,
    force: bool = False,
    throttle_sec: float = 1.5,
) -> None:
    """POST incremental progress; manager merges into reportJson."""
    if not job_uuid:
        return

    terminal = status in {"COMPLETED", "FAILED"}
    now = datetime.now(timezone.utc).timestamp()
    if not force and not terminal:
        last = _LAST_POST.get(job_uuid, 0.0)
        if now - last < throttle_sec:
            return

    report: dict[str, Any] = {}
    if patch:
        report.update(patch)
    if phase:
        report["phase"] = phase
    if event:
        if "ts" not in event:
            event = {**event, "ts": datetime.now(timezone.utc).isoformat()}
        report["event"] = event
        if not phase:
            report["phase"] = event_phase(event)

    payload: dict[str, Any] = {"status": status}
    if report:
        payload["report"] = report

    settings = get_settings()
    url = settings.spring_boot_job_webhook_url.format(id=job_uuid)
    try:
        response = httpx.post(
            url, json=payload, timeout=30.0,
            headers={"X-Worker-Token": settings.worker_internal_token},
        )
        response.raise_for_status()
        _LAST_POST[job_uuid] = now
    except Exception as exc:
        logger.warning("progress webhook failed for job %s: %s", job_uuid, exc)


def attach_progress_sink(log: LogCarrier, job_uuid: str) -> None:
    """Forward LogCarrier events to manager webhooks."""

    def _on_event(record: dict[str, Any]) -> None:
        name = str(record.get("event") or "")
        force = name in {
            "pipeline_start",
            "pipeline_complete",
            "pipeline_failed",
            "importing_training_stack",
            "dataset_loading",
            "dataset_loaded",
            "model_loading",
            "model_loaded",
            "train_config",
            "train_complete",
            "eval_complete",
            "hf_push_complete",
        }
        throttle = 3.0 if name == "train_progress" else 1.5
        progress_notify(
            job_uuid,
            "RUNNING",
            phase=event_phase(record),
            event=record,
            force=force,
            throttle_sec=throttle,
        )

    log.on_event = _on_event


def make_train_progress_callback(log: LogCarrier) -> Callable[..., None]:
    def _callback(**fields: Any) -> None:
        log.info("train_progress", **fields)

    return _callback
