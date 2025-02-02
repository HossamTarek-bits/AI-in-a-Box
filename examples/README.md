# AI Container Examples

This directory contains practical examples demonstrating different aspects of containerizing AI applications.

## Directory Structure

```
examples/
├── basic/              # Basic containerization example
├── cv-model/           # Computer vision model deployment
└── gpu-accel/          # GPU acceleration examples
```

## Basic Example

The basic example demonstrates fundamental containerization concepts with a simple Python application:

- Basic Dockerfile structure
- Environment setup
- Dependency management
- Logging configuration
- Error handling

To run the basic example:
```bash
cd basic
docker build -t basic-example .
docker run basic-example
```

## Computer Vision Model Example

A complete example of deploying a computer vision model as a REST API:

- Pre-trained model deployment
- REST API implementation
- Image preprocessing
- Batch prediction support
- Metrics and monitoring
- Production-ready practices

To run the CV model example:
```bash
cd cv-model
docker build -t cv-model .
docker run -p 8080:8080 cv-model
```

Test the API:
```python
import requests
from PIL import Image

# Test prediction
with open('path/to/image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/predict',
        files={'image': f}
    )
print(response.json())
```

## GPU Acceleration Example

Demonstrates GPU optimization techniques for deep learning containers:

- Multi-stage builds
- GPU configuration
- Memory management
- Multi-GPU support
- Performance benchmarking
- Horovod integration

To run the GPU example:
```bash
cd gpu-accel
docker build -t gpu-benchmark .
docker run --gpus all gpu-benchmark
```

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit (for GPU examples)
- CUDA-compatible GPU (for GPU examples)

## Best Practices Demonstrated

1. **Security**
   - Non-root users
   - Proper permissions
   - Environment variable handling

2. **Performance**
   - Multi-stage builds
   - Layer optimization
   - Caching strategies

3. **Monitoring**
   - Logging setup
   - Metrics collection
   - Health checks

4. **Production Readiness**
   - Error handling
   - Resource management
   - Scalability considerations

## Running the Examples

Each example includes:
- Dockerfile
- Application code
- Requirements file
- Documentation

Follow these steps for each example:

1. Navigate to the example directory
2. Review the README and code
3. Build the Docker image
4. Run the container
5. Test the functionality

## Contributing

Feel free to contribute additional examples or improvements to existing ones:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 