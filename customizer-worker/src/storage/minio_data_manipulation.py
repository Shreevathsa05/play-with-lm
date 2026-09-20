from src.storage.minio_connection import minio_client


class MinioCRUD:
    """Static helpers for MinIO CRUD operations."""

    @staticmethod
    def ensure_bucket_exists(bucket_name: str):
        """Creates the bucket if it does not already exist."""
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

    @staticmethod
    def upload_file(bucket_name: str, object_name: str, file_path: str):
        """Uploads a local file to the MinIO bucket (Create/Update)."""
        MinioCRUD.ensure_bucket_exists(bucket_name)
        return minio_client.fput_object(bucket_name, object_name, file_path)

    @staticmethod
    def get_metadata(bucket_name: str, object_name: str):
        """Gets metadata for an object (Read)."""
        return minio_client.stat_object(bucket_name, object_name)

    @staticmethod
    def download_file(bucket_name: str, object_name: str, destination_path: str):
        """Downloads an object to a local file (Read)."""
        return minio_client.fget_object(bucket_name, object_name, destination_path)

    @staticmethod
    def delete_file(bucket_name: str, object_name: str):
        """Deletes an object from the bucket (Delete)."""
        minio_client.remove_object(bucket_name, object_name)
