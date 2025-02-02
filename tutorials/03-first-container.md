# Building Your First AI Container

In this tutorial, we'll build a complete AI container for a real computer vision application. We'll create a simple image classification service using TensorFlow and expose it via a REST API.

## Project Structure

First, let's set up our project structure:

```
my-cv-app/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── models/
│   └── .gitkeep
└── tests/
    └── test_model.py
```

## Step 1: Setting Up Dependencies

Create a `requirements.txt` file with our dependencies:

```txt
tensorflow==2.4.0
numpy==1.19.5
Pillow==8.2.0
Flask==2.0.1
gunicorn==20.1.0
```

## Step 2: Creating the Application

### 1. Model Handler (`app/model.py`)
```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io

class ModelHandler:
    def __init__(self):
        # Load pre-trained model
        self.model = MobileNetV2(weights='imagenet')
        
    def preprocess_image(self, image_bytes):
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        # Resize image to 224x224
        image = image.resize((224, 224))
        # Convert to array and preprocess
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return preprocess_input(image_array)
    
    def predict(self, image_bytes):
        # Preprocess image
        processed_image = self.preprocess_image(image_bytes)
        # Make prediction
        predictions = self.model.predict(processed_image)
        # Decode and return top 5 predictions
        return decode_predictions(predictions, top=5)[0]
```

### 2. Utility Functions (`app/utils.py`)
```python
def format_predictions(predictions):
    """Format model predictions into JSON-serializable format"""
    return [
        {
            "label": label,
            "name": name,
            "confidence": float(score)
        }
        for label, name, score in predictions
    ]

def validate_image(file):
    """Validate uploaded file is an image"""
    if not file:
        return False
    try:
        Image.open(io.BytesIO(file.read()))
        file.seek(0)  # Reset file pointer
        return True
    except:
        return False
```

### 3. Main Application (`app/main.py`)
```python
from flask import Flask, request, jsonify
from .model import ModelHandler
from .utils import format_predictions, validate_image

app = Flask(__name__)
model_handler = ModelHandler()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Validate image
    if not validate_image(file):
        return jsonify({"error": "Invalid image"}), 400
    
    try:
        # Make prediction
        predictions = model_handler.predict(file.read())
        # Format results
        results = format_predictions(predictions)
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Step 3: Creating the Dockerfile

```dockerfile
# Use TensorFlow base image with GPU support
FROM tensorflow/tensorflow:2.4.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/models
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Start the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "3", "app.main:app"]
```

## Step 4: Building and Running the Container

```bash
# Build the image
docker build -t cv-classifier:1.0 .

# Run the container
docker run -d \
    --name cv-classifier \
    -p 8080:8080 \
    --gpus all \
    cv-classifier:1.0
```

## Step 5: Testing the Service

Here's a simple Python script to test our service:

```python
import requests
from PIL import Image
import io

def test_classification(image_path, server_url='http://localhost:8080'):
    # Open and prepare image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Create files payload
    files = {'image': ('image.jpg', image_data)}
    
    # Make request
    response = requests.post(f'{server_url}/predict', files=files)
    
    # Print results
    if response.status_code == 200:
        results = response.json()
        print("\nPredictions:")
        for pred in results['predictions']:
            print(f"{pred['name']}: {pred['confidence']*100:.2f}%")
    else:
        print(f"Error: {response.json()}")

# Test with an image
test_classification('path/to/your/image.jpg')
```

## Performance Optimization Tips

1. **Batch Processing**
   - Implement batch processing for multiple images
   - Use TensorFlow's batching capabilities

2. **Model Optimization**
   - Consider using TensorFlow Lite for deployment
   - Quantize model weights if possible

3. **Caching**
   - Implement response caching for frequent requests
   - Cache preprocessed images

4. **Resource Management**
   - Monitor GPU memory usage
   - Adjust number of workers based on load

## Common Issues and Solutions

1. **GPU Memory Issues**
   ```python
   # In model.py, add:
   import tensorflow as tf
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           print(e)
   ```

2. **Slow First Prediction**
   - Warm up the model during initialization:
   ```python
   def __init__(self):
       self.model = MobileNetV2(weights='imagenet')
       # Warm up
       self.model.predict(np.zeros((1, 224, 224, 3)))
   ```

3. **Image Format Issues**
   - Add more robust image validation
   - Support multiple image formats

## Next Steps

Now that you have a working AI container, you can:
1. Add authentication
2. Implement model versioning
3. Add monitoring and logging
4. Scale with Kubernetes

## Practice Exercise

1. Modify the application to:
   - Support multiple model architectures
   - Add input validation
   - Implement response caching
   - Add proper logging

2. Optimize the container:
   - Reduce image size
   - Implement health checks
   - Add proper error handling

[Next Tutorial: GPU Acceleration →](04-gpu-acceleration.md) 