from minio import Minio

minio_client = Minio(
    endpoint="localhost:9000",
    access_key="yourusername",
    secret_key="yourstrongpassword",
    secure=False
)