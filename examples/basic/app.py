import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate some sample data for demonstration"""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    return pd.DataFrame(data)

def process_data(df):
    """Perform some basic data processing"""
    results = {
        'total_rows': len(df),
        'mean_value': df['value'].mean(),
        'categories': df['category'].value_counts().to_dict()
    }
    return results

def main():
    logger.info("Starting application...")
    
    # Get environment variables
    env = os.getenv('APP_ENV', 'development')
    logger.info(f"Running in {env} environment")
    
    try:
        # Generate and process data
        logger.info("Generating sample data...")
        df = generate_sample_data()
        
        logger.info("Processing data...")
        results = process_data(df)
        
        # Print results
        logger.info("Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 