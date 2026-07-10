from minio import Minio

minio_client = Minio(
    endpoint="play.min.io",   # MinIO public test server
    access_key="Q3AM3UQ867SPQQA43P2F",
    secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    secure=True
)