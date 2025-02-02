# Getting Started with Docker

This tutorial will introduce you to Docker fundamentals and essential commands you'll need for containerizing AI applications.

## Installing Docker

### Linux
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up stable repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add your user to the docker group (optional, but recommended)
sudo usermod -aG docker $USER
```

### Verify Installation
```bash
docker --version
docker run hello-world
```

## Core Docker Concepts

### 1. Images vs Containers
- **Image**: A blueprint for a container (like a class in programming)
- **Container**: A running instance of an image (like an object in programming)

### 2. Basic Docker Commands

#### Working with Images
```bash
# List all images
docker images

# Pull an image from Docker Hub
docker pull python:3.9-slim

# Build an image from a Dockerfile
docker build -t myapp:1.0 .

# Remove an image
docker rmi python:3.9-slim
```

#### Working with Containers
```bash
# Run a container
docker run python:3.9-slim

# Run container interactively
docker run -it python:3.9-slim bash

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>
```

## Understanding the Dockerfile

A Dockerfile is a text file containing instructions for building a Docker image.

### Basic Structure
```dockerfile
# Start from a base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Command to run when container starts
CMD ["python", "app.py"]
```

### Common Dockerfile Instructions
- `FROM`: Specifies the base image
- `WORKDIR`: Sets the working directory
- `COPY`: Copies files from host to container
- `RUN`: Executes commands during build
- `CMD`: Default command when container starts
- `ENV`: Sets environment variables

## Practical Example: Python AI Application

Let's create a simple Dockerfile for a Python application using TensorFlow:

```dockerfile
# Use TensorFlow base image
FROM tensorflow/tensorflow:2.4.0

# Set working directory
WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/models
ENV LOG_LEVEL=INFO

# Run the application
CMD ["python", "main.py"]
```

### Building and Running
```bash
# Build the image
docker build -t ai-app:1.0 .

# Run the container
docker run -p 8080:8080 ai-app:1.0
```

## Important Docker Flags

- `-p`: Port mapping (host:container)
- `-v`: Volume mounting
- `-e`: Environment variables
- `-d`: Run in detached mode
- `--name`: Assign container name
- `--rm`: Remove container when stopped

Example with multiple flags:
```bash
docker run -d \
    --name my-ai-app \
    -p 8080:8080 \
    -v $(pwd)/data:/app/data \
    -e MODEL_PATH=/app/models \
    --rm \
    ai-app:1.0
```

## Best Practices

1. **Use Specific Tags**
   - Don't use `latest` tag in production
   - Specify exact versions of base images

2. **Minimize Layer Size**
   - Combine RUN commands using &&
   - Remove unnecessary files in the same layer

3. **Security**
   - Don't run as root
   - Don't store secrets in images

4. **Caching**
   - Order instructions from least to most frequently changing
   - Use .dockerignore file

## Practice Exercise

1. Create a simple Python script:
```python
# app.py
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Random number:", np.random.rand())
```

2. Create a `requirements.txt`:
```
numpy==1.19.5
```

3. Create a Dockerfile and build the image
4. Run the container and verify the output

## Troubleshooting Common Issues

1. **Permission Denied**
   ```bash
   # Solution
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Port Already in Use**
   ```bash
   # Find process using port
   sudo lsof -i :8080
   # Kill process or use different port
   ```

3. **Container Exit Immediately**
   - Check logs: `docker logs <container_id>`
   - Try running interactively: `docker run -it <image>`

## Next Steps

Now that you understand Docker basics, you're ready to:
1. [Build your first AI container](03-first-container.md)
2. Learn about managing dependencies
3. Explore GPU acceleration

[Next Tutorial: Building Your First AI Container →](03-first-container.md) 