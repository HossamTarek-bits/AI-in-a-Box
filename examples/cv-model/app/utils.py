import tensorflow as tf
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    try:
        # List physical devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        else:
            logger.warning("No GPUs found, running on CPU")
    
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")
        raise

def validate_image(file):
    """Validate that the file is a valid image"""
    if not file:
        return False
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        file.seek(0)  # Reset file pointer
        
        # Check image mode
        valid_modes = {'RGB', 'RGBA', 'L'}
        if image.mode not in valid_modes:
            logger.warning(f"Invalid image mode: {image.mode}")
            return False
        
        # Check image size
        if image.size[0] < 10 or image.size[1] < 10:
            logger.warning(f"Image too small: {image.size}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False

def format_predictions(predictions):
    """Format model predictions into JSON-serializable format"""
    try:
        return [
            {
                "label": label,
                "name": name.replace('_', ' ').title(),
                "confidence": float(score)
            }
            for label, name, score in predictions
        ]
    
    except Exception as e:
        logger.error(f"Error formatting predictions: {str(e)}")
        raise

def get_model_info():
    """Get information about the current TensorFlow setup"""
    return {
        "tensorflow_version": tf.__version__,
        "gpu_available": bool(tf.config.list_physical_devices('GPU')),
        "gpu_devices": [
            device.name for device in tf.config.list_physical_devices('GPU')
        ]
    } 