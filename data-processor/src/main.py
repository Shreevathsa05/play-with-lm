import threading
import uvicorn
from fastapi import FastAPI
from .consumer import start_consumer
from .config import settings

app = FastAPI(title="Data Processor API")

@app.on_event("startup")
def startup_event():
    # Run the RabbitMQ consumer in a background thread
    # so it doesn't block the FastAPI event loop
    consumer_thread = threading.Thread(target=start_consumer, daemon=True)
    consumer_thread.start()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "data-processor"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.API_PORT, reload=True)
