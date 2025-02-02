# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:2.4.0-gpu

# Create non-root user
RUN useradd -m -s /bin/bash modeluser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory and set permissions
RUN mkdir -p /app/models && \
    chown -R modeluser:modeluser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER modeluser

# Expose port for API
EXPOSE 8080

# Set environment variables
ENV MODEL_PATH=/app/models
ENV PYTHONPATH=/app
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV NVIDIA_VISIBLE_DEVICES=all

# Start the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "3", "--timeout", "120", "app.main:app"] 