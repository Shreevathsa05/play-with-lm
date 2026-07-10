import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASS: str = os.getenv("RABBITMQ_PASS", "guest")
    
    MINIO_URL: str = os.getenv("MINIO_URL", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "password123")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "datasets")
    
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    SPRING_BOOT_WEBHOOK_URL: str = os.getenv("SPRING_BOOT_WEBHOOK_URL", "http://localhost:8080/api/webhook/datasets/{id}")

settings = Settings()
