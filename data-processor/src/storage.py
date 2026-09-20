import io
import logging
import time
from minio import Minio
from .config import settings

logger = logging.getLogger(__name__)


class MinioClientManager:
    def __init__(self, retries: int = 30, delay_sec: float = 2.0):
        last_error = None
        self.client = None
        for attempt in range(1, retries + 1):
            try:
                client = Minio(
                    settings.MINIO_URL,
                    access_key=settings.MINIO_ACCESS_KEY,
                    secret_key=settings.MINIO_SECRET_KEY,
                    secure=False,
                )
                if not client.bucket_exists(settings.MINIO_BUCKET):
                    client.make_bucket(settings.MINIO_BUCKET)
                self.client = client
                logger.info("MinIO connected on attempt %s", attempt)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("MinIO not ready (attempt %s/%s): %s", attempt, retries, exc)
                time.sleep(delay_sec)
        raise RuntimeError(f"Failed to connect to MinIO after {retries} attempts") from last_error

    def download_file(self, object_name: str) -> bytes:
        response = self.client.get_object(settings.MINIO_BUCKET, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def upload_file(self, object_name: str, data: bytes, content_type: str = "application/json"):
        data_stream = io.BytesIO(data)
        self.client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            data_stream,
            length=len(data),
            content_type=content_type,
        )
        return object_name


minio_client = MinioClientManager()
