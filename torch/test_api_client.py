#!/usr/bin/env python3
"""
Test client for the Optimized PyTorch Model API

This script demonstrates how to use the API service to make predictions.
"""

import requests
import json
import numpy as np
import time
import os
import h5py
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from pathlib import Path

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
    
    def predict(self, data: List[List[float]], confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make predictions using the model.
        
        Args:
            data: Input data as a list of lists
            confidence_threshold: Threshold for confidence scores (0.0 to 1.0)
            
        Returns:
            API response with predictions and confidence scores
        """
        try:
            payload = {
                "data": data,
                "confidence_threshold": confidence_threshold
            }
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

def load_h5_data(file_path: str, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the latest n_samples from the H5 file.
    
    Args:
        file_path: Path to the H5 file
        n_samples: Number of latest samples to load
        
    Returns:
        Tuple of (X, Y, T) arrays where:
        - X: Input features
        - Y: Target values
        - T: Timestamps or indices
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Load X, Y, and T arrays as in train.py
            X = np.array(f['X'][:], dtype=np.float32)
            Y = np.array(f['Y'][:], dtype=np.float32)
            T = np.array(f['T'][:], dtype=np.int32)
            
            # Transpose if needed (following train.py logic)
            if X.shape[0] != Y.shape[0]:
                print("Warning: X and Y have different number of samples. Attempting transpose.")
                X = X.T
                Y = Y.T
            
            if T.shape[0] != Y.shape[0]:
                print("Warning: T and Y have different number of samples. Attempting transpose.")
                T = T.T
            
            # Validate shapes
            assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
            assert T.shape[0] == Y.shape[0], "T and Y must have the same number of samples"
            
            # Get the latest n_samples
            if X.shape[0] > n_samples:
                X = X[-n_samples:]
                Y = Y[-n_samples:]
                T = T[-n_samples:]
            
            print(f"X shape: {X.shape}, dtype: {X.dtype}")
            print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
            print(f"T shape: {T.shape}, dtype: {T.dtype}")
            
            return X, Y, T
            
    except Exception as e:
        print(f"Error loading H5 file: {e}")
        return None, None, None

def plot_predictions(actual: np.ndarray, predicted: np.ndarray, confidence: np.ndarray,
                    timestamps: np.ndarray = None, confidence_threshold: float = 0.5,
                    output_dir: str = "prediction_plots"):
    """
    Plot predictions vs actual values for multiple targets, separated by confidence.
    Saves plots to files instead of displaying them.
    
    Args:
        actual: Array of actual values (n_samples, n_targets)
        predicted: Array of predicted values (n_samples, n_targets)
        confidence: Array of confidence scores (n_samples,)
        timestamps: Optional array of timestamps for x-axis
        confidence_threshold: Threshold for confidence scores (default: 0.5)
        output_dir: Directory to save plot files (default: "prediction_plots")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_targets = actual.shape[1]
    x = range(len(actual)) if timestamps is None else timestamps.ravel()
    
    # Create a figure with subplots for each target
    fig, axes = plt.subplots(n_targets, 2, figsize=(15, 5*n_targets))
    
    # Handle single target case
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_targets):
        # Get data for current target
        y_true = actual[:, i]
        y_pred = predicted[:, i]
        
        # Separate high and low confidence predictions
        high_conf_mask = confidence >= confidence_threshold
        low_conf_mask = ~high_conf_mask
        
        # Plot high confidence predictions
        ax1 = axes[i, 0]
        if np.any(high_conf_mask):
            ax1.plot(x[high_conf_mask], y_true[high_conf_mask], 'b-', label='Actual', alpha=0.7)
            ax1.plot(x[high_conf_mask], y_pred[high_conf_mask], 'r--', label='Predicted', alpha=0.7)
            ax1.fill_between(x[high_conf_mask], 
                           y_pred[high_conf_mask] * (1 - confidence[high_conf_mask]),
                           y_pred[high_conf_mask] * (1 + confidence[high_conf_mask]),
                           color='r', alpha=0.2, label='Confidence Range')
            ax1.set_title(f'Target {i+1} - High Confidence (≥{confidence_threshold})')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No high confidence predictions', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot low confidence predictions
        ax2 = axes[i, 1]
        if np.any(low_conf_mask):
            ax2.plot(x[low_conf_mask], y_true[low_conf_mask], 'b-', label='Actual', alpha=0.7)
            ax2.plot(x[low_conf_mask], y_pred[low_conf_mask], 'r--', label='Predicted', alpha=0.7)
            ax2.fill_between(x[low_conf_mask],
                           y_pred[low_conf_mask] * (1 - confidence[low_conf_mask]),
                           y_pred[low_conf_mask] * (1 + confidence[low_conf_mask]),
                           color='r', alpha=0.2, label='Confidence Range')
            ax2.set_title(f'Target {i+1} - Low Confidence (<{confidence_threshold})')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No low confidence predictions', 
                    ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"predictions_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_filename}")

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
    
    # Load test data from H5 file
    print("\n3. Loading test data from H5 file...")
    h5_path = Path("training_data.h5")
    if not h5_path.exists():
        print("Warning: training_data.h5 not found in current directory.")
        print("Please provide the correct path to the H5 file.")
        return
    
    X, Y, T = load_h5_data(str(h5_path))
    if X is None or Y is None or T is None:
        print("Failed to load H5 data.")
        return
    
    print(f"Using latest {len(X)} samples for testing")
    
    # Make predictions
    print("\n4. Making predictions:")
    start_time = time.time()
    result = client.predict(X.tolist())
    end_time = time.time()
    
    if "error" in result:
        print(f"Prediction failed: {result['error']}")
        return
    
    print(f"API call took: {end_time - start_time:.3f} seconds")
    print(f"Model processing time: {result['processing_time']:.3f} seconds")
    print(f"Input shape: {result['input_shape']}")
    print(f"Output shape: {result['output_shape']}")
    
    # Convert predictions and confidence scores
    predictions = np.array(result['predictions'])
    confidence_scores = np.array(result['confidences'])
    
    # Create output directory for this test run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("test_results", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot predictions with timestamps
    print("\n5. Plotting predictions vs actual values...")
    plot_predictions(Y, predictions, confidence_scores, timestamps=T, output_dir=output_dir)
    
    # Show summary statistics
    print("\n6. Summary Statistics:")
    mse = np.mean((Y - predictions) ** 2)
    mae = np.mean(np.abs(Y - predictions))
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Calculate high/low confidence statistics
    confidence_threshold = 0.5  # Match the default in the API
    high_conf_mask = confidence_scores >= confidence_threshold
    print(f"\nHigh confidence predictions (>={confidence_threshold}): {np.sum(high_conf_mask)}")
    print(f"Low confidence predictions (<{confidence_threshold}): {np.sum(~high_conf_mask)}")
    
    if np.any(high_conf_mask):
        high_conf_mse = np.mean((Y[high_conf_mask] - predictions[high_conf_mask]) ** 2)
        print(f"High confidence MSE: {high_conf_mse:.4f}")
        print(f"Time range: {T[high_conf_mask].min()} to {T[high_conf_mask].max()}")
    
    if np.any(~high_conf_mask):
        low_conf_mse = np.mean((Y[~high_conf_mask] - predictions[~high_conf_mask]) ** 2)
        print(f"Low confidence MSE: {low_conf_mse:.4f}")
        print(f"Time range: {T[~high_conf_mask].min()} to {T[~high_conf_mask].max()}")
    
    # Save test statistics to file
    stats_file = os.path.join(output_dir, "test_statistics.txt")
    with open(stats_file, "w") as f:
        f.write("Test Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n\n")
        f.write(f"High confidence predictions (>={confidence_threshold}): {np.sum(high_conf_mask)}\n")
        f.write(f"Low confidence predictions (<{confidence_threshold}): {np.sum(~high_conf_mask)}\n\n")
        
        if np.any(high_conf_mask):
            f.write(f"High confidence MSE: {high_conf_mse:.4f}\n")
            f.write(f"High confidence time range: {T[high_conf_mask].min()} to {T[high_conf_mask].max()}\n\n")
        
        if np.any(~high_conf_mask):
            f.write(f"Low confidence MSE: {low_conf_mse:.4f}\n")
            f.write(f"Low confidence time range: {T[~high_conf_mask].min()} to {T[~high_conf_mask].max()}\n")
    
    print(f"\nTest statistics saved to: {stats_file}")
    print("\nAPI testing complete!")

if __name__ == "__main__":
    main() 