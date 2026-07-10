import io
from minio import Minio
from .config import settings

class MinioClientManager:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_URL,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False
        )
        
        if not self.client.bucket_exists(settings.MINIO_BUCKET):
            self.client.make_bucket(settings.MINIO_BUCKET)

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
            content_type=content_type
        )
        return object_name

minio_client = MinioClientManager()
