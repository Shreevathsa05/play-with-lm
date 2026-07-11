import os
from minio import Minio
from src.storage.minio_connection import minio_client

class MinioCRUD:
    """
    A utility class for MinIO CRUD operations.
    All methods are static, so you can call them directly like MinioCRUD.upload_file(...)
    without needing to instantiate the class or use 'self'.
    """

    @staticmethod
    def ensure_bucket_exists(bucket_name: str):
        """Creates the bucket if it does not already exist."""
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created.")

    @staticmethod
    def upload_file(bucket_name: str, object_name: str, file_path: str):
        """Uploads a local file to the MinIO bucket (Create/Update)."""
        MinioCRUD.ensure_bucket_exists(bucket_name)
        result = minio_client.fput_object(bucket_name, object_name, file_path)
        print(f"Uploaded '{file_path}' to '{bucket_name}/{object_name}'.")
        return result

    @staticmethod
    def get_metadata(bucket_name: str, object_name: str):
        """Gets metadata for an object (Read)."""
        metadata = minio_client.stat_object(bucket_name, object_name)
        return metadata

    @staticmethod
    def download_file(bucket_name: str, object_name: str, destination_path: str):
        """Downloads an object to a local file (Read)."""
        result = minio_client.fget_object(bucket_name, object_name, destination_path)
        print(f"Downloaded '{bucket_name}/{object_name}' to '{destination_path}'.")
        return result

    @staticmethod
    def delete_file(bucket_name: str, object_name: str):
        """Deletes an object from the bucket (Delete)."""
        minio_client.remove_object(bucket_name, object_name)
        print(f"Deleted '{bucket_name}/{object_name}'.")

# Example Usage
if __name__ == "__main__":
    bucket = "test-bucket"
    obj_name = "test/req.txt"
    source_file = "./requirements.txt"
    download_target = "./downloaded.txt"

    # 1. Create (Upload)
    if os.path.exists(source_file):
        # Call the static methods directly on the class!
        MinioCRUD.upload_file(bucket, obj_name, source_file)
        
        # 2. Read Metadata
        meta = MinioCRUD.get_metadata(bucket, obj_name)
        print(f"Metadata size: {meta.size} bytes")
        
        # 3. Read (Download)
        MinioCRUD.download_file(bucket, obj_name, download_target)
        
        # 4. Delete
        MinioCRUD.delete_file(bucket, obj_name)
        
        # Cleanup local downloaded file for the test
        if os.path.exists(download_target):
            os.remove(download_target)
    else:
        print(f"Source file '{source_file}' not found for testing.")
