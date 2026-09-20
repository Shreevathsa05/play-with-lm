"""RabbitMQ consumer for finetuning.jobs.queue — mirrors data-processor pattern."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Optional

import httpx

from src.core.config import get_settings
from src.supervisor import FCFSSupervisor, SupervisedRun
from src.finetuning_modules.compatibility import GateDecision
from src.utils.logcarrier import LogCarrier
from src.utils.progress import progress_notify

logger = logging.getLogger(__name__)


def notify_manager(job_id: str, status: str, report: Optional[dict[str, Any]] = None) -> None:
    settings = get_settings()
    url = settings.spring_boot_job_webhook_url.format(id=job_id)
    payload = {"status": status}
    if report is not None:
        payload["report"] = report
    response = httpx.post(url, json=payload, timeout=30.0,
                          headers={"X-Worker-Token": settings.worker_internal_token})
    response.raise_for_status()


def cancellation_requested(job_id: str) -> bool:
    settings = get_settings()
    url = settings.spring_boot_job_webhook_url.format(id=job_id) + "/control"
    response = httpx.get(url, timeout=10.0, headers={"X-Worker-Token": settings.worker_internal_token})
    response.raise_for_status()
    return bool(response.json().get("cancel_requested"))


def handle_job_message(
    body: bytes | str | dict,
    *,
    supervisor: Optional[FCFSSupervisor] = None,
    notify: Callable[[str, str, Optional[dict]], None] = notify_manager,
) -> SupervisedRun:
    if isinstance(body, (bytes, bytearray)):
        message = json.loads(body.decode("utf-8"))
    elif isinstance(body, str):
        message = json.loads(body)
    else:
        message = dict(body)

    job_uuid = str(message.get("job_uuid") or message.get("jobId") or "")
    if not job_uuid:
        raise ValueError("job payload missing job_uuid")

    log = LogCarrier(job_uuid)
    log.info("job_received", recipe=message.get("recipe"))
    attempt_id = str(uuid.uuid4())
    event_seq = 0

    def send_status(status: str, report: Optional[dict] = None) -> None:
        nonlocal event_seq
        event_seq += 1
        enriched = dict(report or {})
        enriched.update({"attempt_id": attempt_id, "event_seq": event_seq})
        notify(job_uuid, status, enriched)

    superv = supervisor or FCFSSupervisor()
    # The first callback is also a claim check: a queued record deleted after an
    # interrupted worker produces a 404 here and is never allowed to train later.
    send_status("CHECKING", {
        "phase": "Worker picked up the job",
        "event": {"event": "job_received", "recipe": message.get("recipe")},
    })
    result = superv.run_job(
        message,
        status_callback=lambda status: send_status(
            status,
            {
                "phase": "Waiting for free capacity" if status == "WAITING" else "Training subprocess is alive",
                "event": {"event": "heartbeat", "status": status},
            },
        ),
        should_cancel=lambda: cancellation_requested(job_uuid),
    )

    if result.gate.decision == GateDecision.REJECT:
        status = "CANCELLED" if "Cancellation requested" in result.gate.reason else "FAILED"
        send_status(status, {"reason": result.gate.reason, "decision": "reject"})
        return result
    if result.gate.decision == GateDecision.SUGGEST:
        send_status(
            "FAILED",
            {
                "reason": result.gate.reason,
                "decision": "suggest",
                "suggested_recipe": result.gate.suggested_recipe,
            },
        )
        return result

    if result.spawn is None:
        send_status("FAILED", {"reason": "spawn_missing"})
        return result

    report: dict[str, Any] = {"waited_sec": result.waited_sec}
    # Surface the complete structured result from job_runner on success or failure.
    try:
        for line in reversed((result.spawn.stdout or "").strip().splitlines()):
            if line.startswith("__CUSTOMIZER_RESULT__="):
                runner = json.loads(line.split("=", 1)[1])
                report.update({k: v for k, v in runner.items() if k not in {"status", "job_uuid", "attempt_id", "event_seq"}})
                break
            if line.startswith("{"):
                report["runner"] = json.loads(line)
                break
    except Exception:
        pass

    if result.spawn.cancelled:
        if result.spawn.checkpoint_path:
            report["resume_from_checkpoint"] = result.spawn.checkpoint_path
        send_status("CANCELLED", report)
    elif result.spawn.returncode == 0:
        send_status("COMPLETED", report)
    else:
        report.update({
            "returncode": result.spawn.returncode,
            "stderr_tail": (result.spawn.stderr or "")[-2000:],
        })
        send_status("FAILED", report)
    return result


def start_consumer(
    *,
    supervisor: Optional[FCFSSupervisor] = None,
    blocking: bool = True,
    retries: int = 60,
    delay_sec: float = 2.0,
) -> None:
    """
    Start consuming finetuning.jobs.queue with prefetch=1 (FCFS).

    Requires `pika` at runtime. Unit tests call handle_job_message directly.
    """
    try:
        import pika
    except ImportError as exc:
        raise ImportError("pika is required to start the RabbitMQ consumer") from exc

    settings = get_settings()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            credentials = pika.PlainCredentials(settings.rabbitmq_user, settings.rabbitmq_pass)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=settings.rabbitmq_host,
                    port=settings.rabbitmq_port,
                    credentials=credentials,
                    heartbeat=60,
                )
            )
            channel = connection.channel()
            channel.queue_declare(queue=settings.finetuning_queue, durable=True)
            channel.basic_qos(prefetch_count=1)

            def _on_message(ch, method, properties, body):
                try:
                    handle_job_message(body, supervisor=supervisor)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as exc:
                    logger.exception("Failed processing finetune job: %s", exc)
                    try:
                        message = json.loads(body)
                        job_uuid = str(message.get("job_uuid") or message.get("jobId") or "")
                        if job_uuid:
                            notify_manager(job_uuid, "FAILED", {"error": str(exc)})
                    except Exception:
                        logger.exception("Failed to notify manager of consumer error")
                    ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=settings.finetuning_queue, on_message_callback=_on_message)
            logger.info("Finetune consumer started on queue=%s", settings.finetuning_queue)
            if blocking:
                channel.start_consuming()
            return
        except Exception as exc:
            last_error = exc
            logger.error("RabbitMQ connect failed (attempt %s/%s): %s", attempt, retries, exc)
            import time

            time.sleep(delay_sec)

    raise RuntimeError(f"Finetune consumer failed after {retries} attempts") from last_error
