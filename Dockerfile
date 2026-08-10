# Use official lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files to disk and enable line-buffered logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/tmp

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Copy dependency requirements file first for layer caching
COPY requirements.txt .

# Install CPU PyTorch and requirements
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Create cache and log directories with open write permissions for Hugging Face Space user
RUN mkdir -p /app/.model_cache /app/logs && chmod -R 777 /app

# Expose port 7860 for Hugging Face Docker Space
EXPOSE 7860

# Launch FastAPI web server via python -m uvicorn on 0.0.0.0:7860
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
