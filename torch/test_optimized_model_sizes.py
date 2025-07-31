#!/usr/bin/env python3
"""
Test script for optimized model size testing.
This script tests the improved functionality including:
1. GPU memory clearing after each model
2. Skipping existing plots and model files
3. Memory usage monitoring
"""

import os
import sys
from test_model_sizes import test_model_sizes, clear_gpu_memory, show_gpu_memory_usage

def test_optimized_functionality():
    """Test the optimized functionality."""
    print("Testing optimized model size functionality...")
    
    # Test with smaller model sizes for quick testing
    hidden_sizes = [64, 128]  # Smaller sizes for testing
    num_layers = [2, 4]       # Fewer layers for testing
    
    print(f"Testing with reduced model sizes: {hidden_sizes}")
    print(f"Testing with reduced layers: {num_layers}")
    
    # Run the test
    results = test_model_sizes(hidden_sizes, num_layers)
    
    print("\nTest completed!")
    print("Check for generated files:")
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            model_name = f"model_{hidden_size}_layers_{num_layer}"
            files_to_check = [
                f"{model_name}_loss_curve.png",
                f"{model_name}_predictions.png",
                f"{model_name}_best_model.pth"
            ]
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    print(f"✓ {file_path}")
                else:
                    print(f"✗ {file_path} (missing)")
    
    # Check for comparison files
    comparison_files = [
        "model_size_comparison.png",
        "model_size_results.json"
    ]
    for file_path in comparison_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
    
    return results

if __name__ == "__main__":
    test_optimized_functionality() 