"""Dataset processing consumer merged into customizer-worker."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

import httpx

from src.core.config import get_settings
from src.hf_loader import load_hf_dataset
from src.storage.minio_connection import minio_client
from src.standardizer import SFTStandardizer, CPTStandardizer, EmbeddingStandardizer
from src.auditor import auditor
from src.snapshot import parse_dataset_bytes, write_jsonl

logger = logging.getLogger(__name__)

DATASET_QUEUE = "dataset.processing.queue"


def _load_minio_json(object_name: str) -> list[dict[str, Any]]:
    settings = get_settings()
    response = minio_client.get_object(settings.minio_bucket, object_name)
    try:
        raw_data = response.read()
    finally:
        response.close()
        response.release_conn()
    return parse_dataset_bytes(raw_data, object_name)


def standardize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if SFTStandardizer.is_alpaca(rows) or SFTStandardizer.is_openai(rows) or SFTStandardizer.is_sharegpt(rows):
        logger.info("Detected SFT dataset format")
        return SFTStandardizer.standardize(rows)
    if EmbeddingStandardizer.is_pair(rows) or EmbeddingStandardizer.is_triplet(rows):
        logger.info("Detected embedding dataset format")
        return EmbeddingStandardizer.standardize(rows)
    try:
        logger.info("Trying CPT dataset format")
        return CPTStandardizer.standardize_json(rows)
    except ValueError:
        logger.warning("Unknown dataset format; auditing raw rows")
        return rows


def notify_dataset_manager(dataset_id: Any, status: str, audit_report: dict[str, Any]) -> None:
    settings = get_settings()
    url = settings.spring_boot_dataset_webhook_url.format(id=dataset_id)
    response = httpx.post(url, json={"auditReport": audit_report, "status": status}, timeout=30.0,
                          headers={"X-Worker-Token": settings.worker_internal_token})
    response.raise_for_status()


def handle_dataset_message(body: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(body, (bytes, bytearray)):
        message = json.loads(body.decode("utf-8"))
    elif isinstance(body, str):
        message = json.loads(body)
    else:
        message = dict(body)

    dataset_id = message.get("datasetId")
    if not dataset_id:
        raise ValueError("dataset payload missing datasetId")

    source = str(message.get("source") or "MINIO").upper()
    if source == "HUGGINGFACE":
        hf_id = message.get("huggingFaceId")
        if not hf_id:
            raise ValueError("HUGGINGFACE dataset payload missing huggingFaceId")
        rows = load_hf_dataset(hf_id, config=message.get("huggingFaceConfig") or message.get("config"))
    else:
        object_name = message.get("minioObjectName")
        if not object_name:
            raise ValueError("MINIO dataset payload missing minioObjectName")
        rows = _load_minio_json(object_name)

    standardized = standardize_rows(rows)
    report = auditor.audit(standardized)
    snapshot_path, content_hash = write_jsonl(standardized)
    snapshot_object = f"snapshots/{dataset_id}/{content_hash}.jsonl"
    try:
        settings = get_settings()
        minio_client.fput_object(settings.minio_bucket, snapshot_object, snapshot_path,
                                  content_type="application/jsonl")
        report["snapshot"] = {
            "object_name": snapshot_object,
            "uri": f"minio://{settings.minio_bucket}/{snapshot_object}",
            "sha256": content_hash,
            "records": len(standardized),
        }
    finally:
        try:
            import os
            os.remove(snapshot_path)
        except OSError:
            pass
    notify_dataset_manager(dataset_id, "COMPLETED", report)
    return report


def start_dataset_consumer(*, blocking: bool = True, retries: int = 60, delay_sec: float = 2.0) -> None:
    try:
        import pika
    except ImportError as exc:
        raise ImportError("pika is required to start the dataset RabbitMQ consumer") from exc

    settings = get_settings()
    last_error: Optional[Exception] = None
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
            channel.queue_declare(queue=DATASET_QUEUE, durable=True)
            channel.basic_qos(prefetch_count=1)

            def _on_message(ch, method, properties, body):
                try:
                    handle_dataset_message(body)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as exc:
                    logger.exception("Failed processing dataset job: %s", exc)
                    try:
                        message = json.loads(body)
                        dataset_id = message.get("datasetId")
                        if dataset_id:
                            notify_dataset_manager(dataset_id, "FAILED", {"error": str(exc)})
                    except Exception:
                        logger.exception("Failed to notify manager of dataset consumer error")
                    ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=DATASET_QUEUE, on_message_callback=_on_message)
            logger.info("Dataset consumer started on queue=%s", DATASET_QUEUE)
            if blocking:
                channel.start_consuming()
            return
        except Exception as exc:
            last_error = exc
            logger.error("RabbitMQ dataset connect failed (attempt %s/%s): %s", attempt, retries, exc)
            time.sleep(delay_sec)

    raise RuntimeError(f"Dataset consumer failed after {retries} attempts") from last_error


def start_dataset_consumer_thread() -> threading.Thread:
    thread = threading.Thread(target=start_dataset_consumer, daemon=True, name="dataset-consumer")
    thread.start()
    return thread
