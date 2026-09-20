from minio import Minio

from src.core.config import get_settings


def create_minio_client() -> Minio:
    s = get_settings()
    return Minio(
        endpoint=s.minio_endpoint,
        access_key=s.minio_access_key,
        secret_key=s.minio_secret_key,
        secure=s.minio_secure,
    )


# Module-level client for existing call sites; tests can patch this symbol.
minio_client = create_minio_client()
