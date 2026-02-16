# MedGemma CXR Triage - Demo App Dockerfile
# ==========================================

FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY service/ ./service/
COPY ui/ ./ui/
COPY tests/ ./tests/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/static/crops

# Set working directory for service
WORKDIR /app/ui

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command: start Gradio app
CMD ["python", "gradio_app.py"]

# Labels
LABEL maintainer="Ashutosh Kumar Singh"
LABEL description="MedGemma CXR Triage Demo App"
LABEL version="1.0"
