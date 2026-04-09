# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Common Python Packages for LLM Fine-tuning
RUN pip install --no-cache-dir \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    python-dotenv \
    tqdm \
    datasets \
    trl

# Copy the application files
# Note: we use .dockerignore to exclude the venv, output dirs, model checkpoints, etc.
COPY . /workspace/

# Default behavior: run a bash shell
CMD ["bash"]
