# Multi-stage build for a GPU-accelerated deep learning container
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 as builder

# Avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.8 -m pip install --no-cache-dir virtualenv
RUN python3.8 -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Runtime
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# Avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Python and virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables for GPU optimization
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_THREAD_COUNT=1
ENV CUDA_VISIBLE_DEVICES=all

# Run the application
CMD ["python", "train.py"] 