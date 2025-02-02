import tensorflow as tf
import torch
import horovod.tensorflow as hvd
import horovod.torch as hvd_torch
import numpy as np
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUBenchmark:
    def __init__(self):
        # Initialize Horovod
        hvd.init()
        
        # Configure GPU
        self.setup_gpu()
        
        # Initialize metrics
        self.tensorflow_times = []
        self.pytorch_times = []
    
    def setup_gpu(self):
        """Configure GPU settings"""
        try:
            # TensorFlow GPU setup
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            
            # PyTorch GPU setup
            torch.cuda.set_device(hvd.local_rank())
            
            logger.info(f"GPU setup complete. Using GPU {hvd.local_rank()}")
        
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            raise
    
    def tensorflow_benchmark(self, batch_size=32, num_iterations=100):
        """Run TensorFlow GPU benchmark"""
        logger.info("Starting TensorFlow benchmark...")
        
        try:
            # Create random data
            data = tf.random.normal([batch_size, 224, 224, 3])
            
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1000)
            ])
            
            # Warm-up
            _ = model(data)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = model(data)
            tf.keras.backend.clear_session()
            
            elapsed = time.time() - start_time
            self.tensorflow_times.append(elapsed / num_iterations)
            
            logger.info(f"TensorFlow average time per batch: {elapsed/num_iterations:.4f} seconds")
            
        except Exception as e:
            logger.error(f"Error in TensorFlow benchmark: {str(e)}")
            raise
    
    def pytorch_benchmark(self, batch_size=32, num_iterations=100):
        """Run PyTorch GPU benchmark"""
        logger.info("Starting PyTorch benchmark...")
        
        try:
            # Create random data
            data = torch.randn(batch_size, 3, 224, 224).cuda()
            
            # Create a simple model
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 32, 3),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 1000)
            ).cuda()
            
            # Warm-up
            with torch.no_grad():
                _ = model(data)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(data)
            torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            self.pytorch_times.append(elapsed / num_iterations)
            
            logger.info(f"PyTorch average time per batch: {elapsed/num_iterations:.4f} seconds")
            
        except Exception as e:
            logger.error(f"Error in PyTorch benchmark: {str(e)}")
            raise
    
    def run_benchmarks(self, batch_sizes=[32, 64, 128]):
        """Run all benchmarks with different batch sizes"""
        logger.info("Starting GPU benchmarks...")
        
        results = {
            'tensorflow': {},
            'pytorch': {},
            'gpu_info': self.get_gpu_info()
        }
        
        for batch_size in batch_sizes:
            logger.info(f"\nRunning benchmarks with batch size {batch_size}")
            
            # TensorFlow benchmark
            self.tensorflow_benchmark(batch_size=batch_size)
            results['tensorflow'][batch_size] = self.tensorflow_times[-1]
            
            # PyTorch benchmark
            self.pytorch_benchmark(batch_size=batch_size)
            results['pytorch'][batch_size] = self.pytorch_times[-1]
        
        self.log_results(results)
        return results
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            return {
                'tensorflow': {
                    'version': tf.__version__,
                    'gpu_available': bool(tf.config.list_physical_devices('GPU')),
                    'gpu_devices': [device.name for device in tf.config.list_physical_devices('GPU')]
                },
                'pytorch': {
                    'version': torch.__version__,
                    'gpu_available': torch.cuda.is_available(),
                    'gpu_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                }
            }
        except Exception as e:
            logger.error(f"Error getting GPU info: {str(e)}")
            return {}
    
    def log_results(self, results):
        """Log benchmark results"""
        logger.info("\nBenchmark Results:")
        logger.info("=" * 50)
        
        # Log GPU information
        logger.info("\nGPU Information:")
        for framework, info in results['gpu_info'].items():
            logger.info(f"\n{framework.title()}:")
            for key, value in info.items():
                logger.info(f"  {key}: {value}")
        
        # Log benchmark results
        logger.info("\nPerformance Results:")
        for framework in ['tensorflow', 'pytorch']:
            logger.info(f"\n{framework.title()}:")
            for batch_size, time in results[framework].items():
                logger.info(f"  Batch size {batch_size}: {time:.4f} seconds per batch")

def main():
    try:
        # Run benchmarks
        benchmark = GPUBenchmark()
        results = benchmark.run_benchmarks()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f"benchmark_results_{timestamp}.npy", results)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 