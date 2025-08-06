#!/usr/bin/env python3
"""
Test client for the Optimized PyTorch Model API

This script demonstrates how to use the API service to make predictions.
"""

import requests
import json
import numpy as np
import time
from typing import List, Dict, Any

class ModelAPIClient:
    """Client for interacting with the model API service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the API service."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def predict(self, data: List[List[float]]) -> Dict[str, Any]:
        """
        Make predictions using the model.
        
        Args:
            data: Input data as a list of lists
            
        Returns:
            API response with predictions
        """
        try:
            payload = {"data": data}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    def reload_model(self) -> Dict[str, Any]:
        """Reload the model."""
        try:
            response = requests.post(f"{self.base_url}/model/reload")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Model reload failed: {e}")
            return {"error": str(e)}

def generate_test_data(input_dim: int, batch_size: int = 5) -> List[List[float]]:
    """Generate random test data for the model."""
    # Generate random data with the same distribution as training data
    # (standardized with mean=0, std=1)
    data = np.random.randn(batch_size, input_dim).astype(np.float32)
    return data.tolist()

def main():
    """Main function to test the API service."""
    print("Testing Optimized PyTorch Model API")
    print("=" * 50)
    
    # Initialize client
    client = ModelAPIClient()
    
    # Check health
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if health.get("status") != "healthy":
        print("API service is not healthy. Please make sure it's running.")
        return
    
    # Get model info
    print("\n2. Model Information:")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    if "error" in model_info:
        print("Failed to get model info.")
        return
    
    # Get input dimension from model info
    input_dim = model_info.get("input_dim")
    if input_dim is None:
        print("Could not determine input dimension from model info.")
        return
    
    print(f"\nModel expects input dimension: {input_dim}")
    
    # Generate test data
    print(f"\n3. Generating test data with {input_dim} features...")
    test_data = generate_test_data(input_dim, batch_size=3)
    print(f"Generated {len(test_data)} samples with {len(test_data[0])} features each")
    
    # Make predictions
    print("\n4. Making predictions:")
    start_time = time.time()
    result = client.predict(test_data)
    end_time = time.time()
    
    if "error" in result:
        print(f"Prediction failed: {result['error']}")
        return
    
    print(f"API call took: {end_time - start_time:.3f} seconds")
    print(f"Model processing time: {result['processing_time']:.3f} seconds")
    print(f"Input shape: {result['input_shape']}")
    print(f"Output shape: {result['output_shape']}")
    
    # Show predictions
    print("\n5. Predictions:")
    for i, pred in enumerate(result['predictions']):
        print(f"Sample {i+1}: {pred}")
    
    # Test with different batch sizes
    print("\n6. Testing different batch sizes:")
    for batch_size in [1, 5, 10]:
        test_data = generate_test_data(input_dim, batch_size)
        start_time = time.time()
        result = client.predict(test_data)
        end_time = time.time()
        
        if "error" not in result:
            print(f"Batch size {batch_size}: {result['processing_time']:.3f}s")
        else:
            print(f"Batch size {batch_size}: Failed - {result['error']}")
    
    print("\nAPI testing complete!")

if __name__ == "__main__":
    main() 