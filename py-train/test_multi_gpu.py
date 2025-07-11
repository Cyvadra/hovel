#!/usr/bin/env python3
"""
Test script for multi-GPU functionality
"""

import tensorflow as tf
import numpy as np
from train import configure_gpus, display_gpu_info, monitor_gpu_memory, optimize_batch_sizes

def test_gpu_configuration():
    """Test GPU configuration and display information"""
    print("Testing Multi-GPU Configuration")
    print("="*50)
    
    # Configure GPUs
    strategy, num_gpus = configure_gpus()
    
    # Display GPU information
    display_gpu_info()
    
    # Monitor GPU memory
    monitor_gpu_memory()
    
    # Test batch size optimization
    base_batch_sizes = [64, 128, 256]
    optimized_batch_sizes = optimize_batch_sizes(num_gpus, base_batch_sizes)
    print(f"\nBatch size optimization:")
    print(f"Base batch sizes: {base_batch_sizes}")
    print(f"Optimized for {num_gpus} GPU(s): {optimized_batch_sizes}")
    
    # Test simple model creation and training
    print(f"\nTesting model creation with {num_gpus} GPU(s)...")
    
    # Create simple test data
    x_train = np.random.random((1000, 10)).astype(np.float32)
    y_train = np.random.random((1000, 5)).astype(np.float32)
    x_val = np.random.random((200, 10)).astype(np.float32)
    y_val = np.random.random((200, 5)).astype(np.float32)
    
    # Import and test model building
    from train import build_model
    
    try:
        # Build model with distribution strategy
        model = build_model((10,), 5, strategy=strategy)
        print("✓ Model built successfully with distribution strategy")
        
        # Test training
        batch_size = optimize_batch_sizes(num_gpus, [32])[0]
        print(f"Training with batch size: {batch_size}")
        
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=batch_size,
            verbose=1
        )
        print("✓ Training completed successfully")
        
        # Test evaluation
        results = model.evaluate(x_val, y_val, verbose=0)
        print(f"✓ Evaluation completed: Loss = {results[0]:.4f}")
        
    except Exception as e:
        print(f"✗ Error during model training: {e}")
        return False
    
    print(f"\n{'='*50}")
    print("Multi-GPU Test Completed Successfully!")
    print(f"{'='*50}")
    return True

if __name__ == "__main__":
    success = test_gpu_configuration()
    if success:
        print("\nAll tests passed! Multi-GPU configuration is working correctly.")
    else:
        print("\nSome tests failed. Please check your GPU configuration.") 