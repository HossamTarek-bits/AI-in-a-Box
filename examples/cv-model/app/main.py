from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, start_http_server
import logging
import os
from datetime import datetime

from .model import ModelHandler
from .utils import setup_gpu, validate_image, format_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
INFERENCE_TIME = Histogram('model_inference_seconds', 'Time spent on inference')

# Initialize model
model_handler = None

@app.before_first_request
def initialize():
    global model_handler
    # Setup GPU
    setup_gpu()
    # Initialize model
    model_handler = ModelHandler()
    # Start metrics server
    start_http_server(8000)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    checks = {
        'status': 'healthy',
        'model_loaded': model_handler is not None,
        'timestamp': datetime.utcnow().isoformat()
    }
    return jsonify(checks)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Validate image
    if not validate_image(file):
        return jsonify({'error': 'Invalid image'}), 400
    
    try:
        # Make prediction
        start_time = datetime.now()
        predictions = model_handler.predict(file.read())
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Format results
        results = format_predictions(predictions)
        
        # Record metrics
        PREDICTION_COUNTER.inc()
        INFERENCE_TIME.observe(inference_time)
        
        return jsonify({
            'predictions': results,
            'inference_time': inference_time
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Custom metrics endpoint"""
    metrics = {
        'total_predictions': PREDICTION_COUNTER._value.get(),
        'average_inference_time': sum(INFERENCE_TIME._sum.get().values())
    }
    return jsonify(metrics)

if __name__ == '__main__':
    # Initialize on startup
    initialize()
    
    # Run the application
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 