# GPU Acceleration for AI Containers

This tutorial covers how to properly set up and utilize GPU acceleration in your AI containers, specifically for computer vision applications.

## Prerequisites

Before starting, ensure you have:
- NVIDIA GPU(s) installed
- NVIDIA drivers installed
- NVIDIA Container Toolkit (nvidia-docker2) installed

## Installing NVIDIA Container Toolkit

### 1. Set up the NVIDIA Container Toolkit repository
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package listings
sudo apt-get update
```

### 2. Install NVIDIA Container Toolkit
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Verify Installation
```bash
# Test with NVIDIA's base CUDA container
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Understanding GPU Resources in Docker

### 1. GPU Allocation Options
```bash
# All GPUs
--gpus all

# Specific GPU(s)
--gpus '"device=0,1"'

# With specific capabilities
--gpus 'all,capabilities=compute,utility'
```

### 2. GPU Memory Management
```bash
# Limit GPU memory (using nvidia-smi)
nvidia-smi -pl 150  # Set power limit to 150W
```

## Optimizing Dockerfiles for GPU

### 1. Base Image Selection

```dockerfile
# For TensorFlow
FROM tensorflow/tensorflow:2.4.0-gpu

# For PyTorch
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# For general CUDA applications
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu20.04
```

### 2. CUDA and cuDNN Compatibility

```dockerfile
# Example for TensorFlow 2.4.0
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow with GPU support
RUN pip3 install tensorflow-gpu==2.4.0
```

## GPU Memory Management in Python

### 1. TensorFlow Memory Management
```python
import tensorflow as tf

def configure_gpu_memory():
    # Allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optionally, set memory limit
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
        except RuntimeError as e:
            print(e)

# Call this function at startup
configure_gpu_memory()
```

### 2. PyTorch Memory Management
```python
import torch

def configure_gpu_memory():
    if torch.cuda.is_available():
        # Set device
        device = torch.device('cuda')
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Optional: Set memory allocation
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        
        return device
    return torch.device('cpu')

# Use in your code
device = configure_gpu_memory()
model = model.to(device)
```

## Multi-GPU Strategies

### 1. Data Parallel Training
```python
import torch.nn as nn

def setup_multi_gpu(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model.to(device)
```

### 2. Distributed Training
```python
# In your Dockerfile
RUN pip install horovod

# In your Python code
import horovod.torch as hvd

def setup_distributed():
    # Initialize Horovod
    hvd.init()
    
    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())
    
    # Scale learning rate by number of GPUs
    learning_rate = learning_rate * hvd.size()
```

## Performance Monitoring

### 1. GPU Monitoring Script
```python
import pynvml
import time

def monitor_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    while True:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU Memory: {info.used/1024**2:.2f}MB / {info.total/1024**2:.2f}MB")
        print(f"GPU Utilization: {pynvml.nvmlDeviceGetUtilizationRates(handle).gpu}%")
        time.sleep(1)

# Run in a separate thread
from threading import Thread
Thread(target=monitor_gpu, daemon=True).start()
```

### 2. Docker Stats with GPU
```bash
# Monitor container GPU usage
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.PIDs}}"

# Using nvidia-smi in container
docker exec <container_name> nvidia-smi
```

## Common Issues and Solutions

### 1. Out of Memory Errors
```python
try:
    # Your GPU operation here
    result = model(input_tensor)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Clear cache and retry
        torch.cuda.empty_cache()
        # Optionally reduce batch size
        result = model(input_tensor[:batch_size//2])
```

### 2. GPU Not Detected
```bash
# Check NVIDIA driver installation
nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu

# Verify NVIDIA Container Toolkit
sudo systemctl status nvidia-docker
```

## Best Practices

1. **Memory Management**
   - Always clear GPU cache between operations
   - Monitor memory usage
   - Use appropriate batch sizes

2. **Resource Allocation**
   - Don't allocate all GPU memory at start
   - Use memory growth options
   - Set appropriate memory limits

3. **Multi-GPU Setup**
   - Use appropriate distribution strategy
   - Balance load across GPUs
   - Monitor individual GPU usage

4. **Container Configuration**
   ```dockerfile
   # In your Dockerfile
   ENV TF_FORCE_GPU_ALLOW_GROWTH=true
   ENV NVIDIA_VISIBLE_DEVICES=all
   ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
   ```

## Practice Exercise

1. Create a GPU monitoring dashboard:
   - Track GPU memory usage
   - Monitor temperature
   - Log utilization metrics

2. Implement multi-GPU training:
   - Data parallel approach
   - Distributed training
   - Compare performance

[Next Tutorial: Best Practices & Production Tips →](05-best-practices.md) 