from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-based worker settings. No hardcoded secrets."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Hugging Face (optional until export/push)
    hf_token: str = ""
    # When set and HF_TOKEN is present, jobs without export.hf_repo push to
    # {hf_username}/{job_uuid}.
    hf_username: str = ""

    # App
    port: int = 8000

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_bucket: str = "datasets"

    # RabbitMQ
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = ""
    rabbitmq_pass: str = ""
    finetuning_queue: str = "finetuning.jobs.queue"

    # Manager webhooks
    spring_boot_job_webhook_url: str = "http://localhost:18080/api/webhook/jobs/{id}"
    spring_boot_dataset_webhook_url: str = "http://localhost:18080/api/webhook/datasets/{id}"
    worker_internal_token: str = ""

    # Capacity defaults (dev GTX 1650-ish); override via env for DGX
    machine_max_vram_mb: int = 4096
    machine_max_ram_mb: int = 16384
    # This installation is intentionally limited to small language models. Keep
    # this independent of the capacity estimate: quantization can make a larger
    # model appear to fit while still making a GTX 1650 unusable in practice.
    max_model_params_b: float = 1.0

    # When true, skip GPU train/load and return synthetic COMPLETED (CI / wiring tests)
    dry_run: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Lazy-friendly alias used by existing imports
settings = get_settings()
