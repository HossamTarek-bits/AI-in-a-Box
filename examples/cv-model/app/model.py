import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        """Initialize the model handler"""
        logger.info("Initializing model handler...")
        try:
            # Load pre-trained model
            self.model = MobileNetV2(weights='imagenet')
            # Warm up the model
            logger.info("Warming up model...")
            self._warmup()
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _warmup(self):
        """Warm up the model with a dummy prediction"""
        dummy_input = np.zeros((1, 224, 224, 3))
        self.model.predict(dummy_input)
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for model input"""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Resize image to 224x224
            image = image.resize((224, 224))
            
            # Convert to array and preprocess
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            return preprocess_input(image_array)
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image_bytes):
        """Make prediction on input image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Decode and return top 5 predictions
            return decode_predictions(predictions, top=5)[0]
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def batch_predict(self, image_batch):
        """Make predictions on a batch of images"""
        try:
            # Process batch
            processed_batch = np.vstack([
                self.preprocess_image(img_bytes)
                for img_bytes in image_batch
            ])
            
            # Make predictions
            predictions = self.model.predict(processed_batch)
            
            # Decode predictions
            return [
                decode_predictions(pred.reshape(1, -1), top=5)[0]
                for pred in predictions
            ]
        
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise 