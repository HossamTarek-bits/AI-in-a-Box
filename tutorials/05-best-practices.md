# Best Practices & Production Tips for AI Containers

This tutorial covers essential best practices and production tips for deploying AI containers in a production environment, with a focus on computer vision applications.

## Security Best Practices

### 1. Container Security

```dockerfile
# Use specific versions instead of 'latest'
FROM tensorflow/tensorflow:2.4.0-gpu

# Create a non-root user
RUN useradd -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set proper permissions
RUN chmod -R 755 /app
```

### 2. Secrets Management
```yaml
# docker-compose.yml
version: '3.8'
services:
  ai-service:
    image: my-ai-app:1.0
    environment:
      - MODEL_API_KEY=${MODEL_API_KEY}
    secrets:
      - source: api_key
        target: /run/secrets/api_key

secrets:
  api_key:
    file: ./secrets/api_key.txt
```

### 3. Image Scanning
```bash
# Install Trivy scanner
sudo apt-get install trivy

# Scan image for vulnerabilities
trivy image my-ai-app:1.0
```

## Resource Management

### 1. Container Resource Limits
```yaml
# docker-compose.yml
services:
  ai-service:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### 2. GPU Memory Management
```python
import tensorflow as tf

class ResourceManager:
    @staticmethod
    def configure_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Memory limit per GPU
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

    @staticmethod
    def monitor_resources():
        import psutil
        import GPUtil
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        gpus = GPUtil.getGPUs()
        gpu_usage = {gpu.id: gpu.memoryUtil * 100 for gpu in gpus}
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_usage': gpu_usage
        }
```

## Logging and Monitoring

### 1. Structured Logging
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ai-service')
    
    def log(self, event_type, message, **kwargs):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'message': message,
            **kwargs
        }
        self.logger.info(json.dumps(log_entry))

# Usage
logger = StructuredLogger()
logger.log('prediction', 'Model prediction completed', 
           model_version='1.0.0',
           inference_time=0.45,
           confidence_score=0.92)
```

### 2. Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
INFERENCE_TIME = Histogram('model_inference_seconds', 'Time spent on inference')
CONFIDENCE_SCORE = Histogram('model_confidence', 'Prediction confidence scores')

class MetricsCollector:
    @staticmethod
    def record_prediction(inference_time, confidence):
        PREDICTION_COUNTER.inc()
        INFERENCE_TIME.observe(inference_time)
        CONFIDENCE_SCORE.observe(confidence)

# Start metrics server
start_http_server(8000)
```

## Model Versioning and Updates

### 1. Model Registry
```python
import mlflow

class ModelRegistry:
    def __init__(self, registry_uri):
        mlflow.set_tracking_uri(registry_uri)
    
    def register_model(self, model, name, version):
        mlflow.tensorflow.log_model(
            model,
            "model",
            registered_model_name=name
        )
        
    def load_model(self, name, version):
        return mlflow.tensorflow.load_model(
            f"models:/{name}/{version}"
        )
```

### 2. Hot Model Updates
```python
from threading import Lock
import schedule
import time

class ModelServer:
    def __init__(self):
        self.model = None
        self.model_lock = Lock()
        
    def update_model(self, new_model):
        with self.model_lock:
            self.model = new_model
    
    def predict(self, input_data):
        with self.model_lock:
            return self.model.predict(input_data)
    
    def schedule_updates(self):
        schedule.every(12).hours.do(self.check_and_update_model)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
```

## Performance Optimization

### 1. Batch Processing
```python
import numpy as np

class BatchProcessor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.batch = []
        
    def add_to_batch(self, item):
        self.batch.append(item)
        if len(self.batch) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        if not self.batch:
            return []
        
        # Process batch
        batch_array = np.stack(self.batch)
        results = self.model.predict(batch_array)
        
        # Clear batch
        self.batch = []
        return results
```

### 2. Caching
```python
from functools import lru_cache
import hashlib

class PredictionCache:
    def __init__(self, maxsize=1000):
        self.cache = lru_cache(maxsize=maxsize)(self._predict)
        
    def _predict(self, input_hash):
        # Actual prediction logic
        pass
    
    def predict(self, input_data):
        # Create hash of input data
        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        return self.cache(input_hash)
```

## Error Handling and Recovery

### 1. Graceful Degradation
```python
class ResilientModel:
    def __init__(self, primary_model, fallback_model=None):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        
    def predict(self, input_data):
        try:
            return self.primary_model.predict(input_data)
        except Exception as e:
            logging.error(f"Primary model failed: {e}")
            if self.fallback_model:
                try:
                    return self.fallback_model.predict(input_data)
                except Exception as e:
                    logging.error(f"Fallback model failed: {e}")
            raise
```

### 2. Health Checks
```python
from flask import Flask, jsonify
import tensorflow as tf

app = Flask(__name__)

def check_gpu_health():
    try:
        # Verify GPU is accessible
        tf.test.is_gpu_available()
        return True
    except:
        return False

@app.route('/health')
def health_check():
    checks = {
        'gpu': check_gpu_health(),
        'model_loaded': hasattr(app, 'model'),
        'memory_available': psutil.virtual_memory().available > 1024 * 1024 * 1024  # 1GB
    }
    
    status = 200 if all(checks.values()) else 503
    return jsonify(checks), status
```

## Production Checklist

1. **Security**
   - [ ] Non-root user configured
   - [ ] Secrets properly managed
   - [ ] Image scanned for vulnerabilities
   - [ ] Network security configured

2. **Monitoring**
   - [ ] Logging implemented
   - [ ] Metrics collection set up
   - [ ] Resource monitoring in place
   - [ ] Alerts configured

3. **Performance**
   - [ ] Resource limits set
   - [ ] Batch processing implemented
   - [ ] Caching strategy defined
   - [ ] Load testing completed

4. **Reliability**
   - [ ] Health checks implemented
   - [ ] Error handling in place
   - [ ] Backup strategy defined
   - [ ] Recovery procedures documented

## Practice Exercise

1. Implement a complete production-ready AI service:
   - Set up monitoring and logging
   - Implement caching and batch processing
   - Add health checks and graceful degradation
   - Configure resource limits and security

2. Create a deployment pipeline:
   - Automated testing
   - Security scanning
   - Resource validation
   - Monitoring setup

This concludes our tutorial series on AI containerization. You now have a comprehensive understanding of how to build, deploy, and maintain AI containers in production. 