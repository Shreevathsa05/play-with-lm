import pika
import json
import logging
import time
import httpx
from .config import settings
from .storage import minio_client
from .standardizer import SFTStandardizer, CPTStandardizer, EmbeddingStandardizer
from .auditor import auditor
from .hf_loader import load_hf_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        dataset_id = message.get("datasetId")
        source = message.get("source", "MINIO")
        
        parsed_data = None
        
        if source == "HUGGINGFACE":
            hf_id = message.get("huggingFaceId")
            hf_config = message.get("huggingFaceConfig") or message.get("config")
            logger.info(
                "Received HuggingFace job for Dataset ID: %s, Repo: %s, Config: %s",
                dataset_id,
                hf_id,
                hf_config,
            )
            parsed_data = load_hf_dataset(hf_id, config=hf_config)
            logger.info("Successfully loaded %s rows from HuggingFace", len(parsed_data))
            
        else:
            object_name = message.get("minioObjectName")
            logger.info(f"Received MinIO job for Dataset ID: {dataset_id}, Object: {object_name}")
            
            # Download from MinIO
            raw_data = minio_client.download_file(object_name)
            logger.info(f"Successfully downloaded {object_name} from MinIO ({len(raw_data)} bytes)")
            
            # Parse JSON
            parsed_data = json.loads(raw_data.decode('utf-8'))
            if not isinstance(parsed_data, list):
                parsed_data = [parsed_data]
            
        # 1. Standardize
        # Attempt to standardize by guessing format
        standardized_data = None
        if SFTStandardizer.is_alpaca(parsed_data) or SFTStandardizer.is_openai(parsed_data) or SFTStandardizer.is_sharegpt(parsed_data):
            standardized_data = SFTStandardizer.standardize(parsed_data)
            logger.info("Detected SFT Format")
        elif EmbeddingStandardizer.is_pair(parsed_data) or EmbeddingStandardizer.is_triplet(parsed_data):
            standardized_data = EmbeddingStandardizer.standardize(parsed_data)
            logger.info("Detected Embedding Format")
        else:
            try:
                standardized_data = CPTStandardizer.standardize_json(parsed_data)
                logger.info("Detected CPT Format")
            except ValueError:
                standardized_data = parsed_data # Fallback
                logger.warning("Unknown Format - No standardization applied")

        # 2. Audit
        report = auditor.audit(standardized_data)
        logger.info(f"Generated Audit Report for {dataset_id}: Score {report.get('health_score')}")

        # 3. Webhook Callback to Spring Boot
        webhook_url = settings.SPRING_BOOT_WEBHOOK_URL.format(id=dataset_id)
        # Using httpx synchronously in thread
        response = httpx.post(webhook_url, json={"auditReport": report, "status": "COMPLETED"})
        response.raise_for_status()
        
        identifier = hf_id if source == "HUGGINGFACE" else object_name
        logger.info(f"Successfully processed {identifier} and notified backend.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Notify backend of failure if we have dataset_id
        try:
            message = json.loads(body)
            dataset_id = message.get("datasetId")
            if dataset_id:
                webhook_url = settings.SPRING_BOOT_WEBHOOK_URL.format(id=dataset_id)
                httpx.post(webhook_url, json={"auditReport": {"error": str(e)}, "status": "FAILED"})
        except Exception as inner_e:
            logger.error(f"Failed to send failure webhook: {inner_e}")
            
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

def start_consumer(retries: int = 60, delay_sec: float = 2.0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASS)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=settings.RABBITMQ_HOST,
                    port=settings.RABBITMQ_PORT,
                    credentials=credentials,
                    heartbeat=60,
                )
            )
            channel = connection.channel()

            channel.queue_declare(queue='dataset.processing.queue', durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='dataset.processing.queue', on_message_callback=process_message)

            logger.info("RabbitMQ Consumer started. Waiting for messages...")
            channel.start_consuming()
            return
        except Exception as e:
            last_error = e
            logger.error("Failed to start RabbitMQ consumer (attempt %s/%s): %s", attempt, retries, e)
            time.sleep(delay_sec)
    raise RuntimeError(f"RabbitMQ consumer failed after {retries} attempts") from last_error
