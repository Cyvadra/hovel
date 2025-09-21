#!/usr/bin/env python3
"""
Analyze model predictions across confidence percentiles.
This script evaluates model performance by dividing predictions into 20 confidence intervals
and analyzing the sign match rate and average confidence for each interval.
"""

import numpy as np
import h5py
import requests
import json
from typing import Dict, Any, Tuple, List
from pathlib import Path

class ModelAPIClient:
    """Client for interacting with the model API service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def predict(self, data: List[List[float]]) -> Dict[str, Any]:
        """Make predictions using the model."""
        try:
            payload = {
                "data": data,
                "confidence_threshold": 0.0  # Set to 0 to get all predictions
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

def load_h5_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all data from the H5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            X = np.array(f['X'][:], dtype=np.float32)
            Y = np.array(f['Y'][:], dtype=np.float32)
            T = np.array(f['T'][:], dtype=np.int32)
            
            if X.shape[0] != Y.shape[0]:
                X = X.T
                Y = Y.T
            
            if T.shape[0] != Y.shape[0]:
                T = T.T
            
            assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
            assert T.shape[0] == Y.shape[0], "T and Y must have the same number of samples"
            
            print(f"X shape: {X.shape}, dtype: {X.dtype}")
            print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
            print(f"T shape: {T.shape}, dtype: {T.dtype}")
            
            return X, Y, T
            
    except Exception as e:
        print(f"Error loading H5 file: {e}")
        return None, None, None

def calculate_sign_match_rate(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate the sign match rate between actual and predicted values."""
    actual_signs = np.sign(actual)
    predicted_signs = np.sign(predicted)
    matches = (actual_signs == predicted_signs)
    return np.mean(matches)

def analyze_predictions_by_time(
    actual: np.ndarray,
    predicted: np.ndarray,
    confidences: np.ndarray,
    timestamps: np.ndarray,
    n_intervals: int = 50
) -> None:
    """
    Analyze predictions by dividing them into time intervals.
    
    Args:
        actual: Array of actual values (n_samples, n_targets)
        predicted: Array of predicted values (n_samples, n_targets)
        confidences: Array of confidence scores (n_samples,)
        timestamps: Array of timestamps (n_samples,)
        n_intervals: Number of intervals to divide the time range into (default: 50 for 2% time windows)
    
    Raises:
        ValueError: If input arrays have incompatible shapes or dimensions
    """
    # Validate input array shapes
    if not isinstance(actual, np.ndarray) or not isinstance(predicted, np.ndarray) or \
       not isinstance(confidences, np.ndarray) or not isinstance(timestamps, np.ndarray):
        raise ValueError("All inputs must be numpy arrays")
        
    if actual.shape[0] != predicted.shape[0] or \
       actual.shape[0] != confidences.shape[0] or \
       actual.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"All arrays must have same number of samples. Got shapes: "
            f"actual={actual.shape}, predicted={predicted.shape}, "
            f"confidences={confidences.shape}, timestamps={timestamps.shape}"
        )
    # Print initial shapes for debugging
    # print(f"\nInitial shapes:")
    # print(f"actual: {actual.shape}")
    # print(f"predicted: {predicted.shape}")
    # print(f"confidences: {confidences.shape}")
    # print(f"timestamps: {timestamps.shape}")
    
    # Ensure timestamps is 1D
    if len(timestamps.shape) > 1:
        timestamps = timestamps.ravel()
    
    # Sort all data by timestamps
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    actual = actual[sort_indices]
    predicted = predicted[sort_indices]
    confidences = confidences[sort_indices]
    
    # Reshape arrays if needed to ensure proper broadcasting
    if len(actual.shape) > 2:
        actual = actual.reshape(actual.shape[0], -1)
    if len(predicted.shape) > 2:
        predicted = predicted.reshape(predicted.shape[0], -1)
    
    # Calculate time boundaries for equal-sized intervals
    time_boundaries = np.linspace(timestamps.min(), timestamps.max(), n_intervals + 1)
    
    print("\nAnalysis by Time Intervals:")
    print("=" * 70)
    print(f"{'Interval':>10} {'Time Range':>25} {'Avg Conf':>10} {'Sign Match':>10} {'Count':>8}")
    print("-" * 70)
    
    for i in range(n_intervals):
        # Get data for current time interval
        if i == n_intervals - 1:
            mask = (timestamps >= time_boundaries[i])
        else:
            mask = (timestamps >= time_boundaries[i]) & (timestamps < time_boundaries[i+1])
        
        if not np.any(mask):
            continue
        
        # Expand mask to match array dimensions if needed
        mask_idx = np.where(mask)[0]
        
        # Calculate metrics for this interval
        interval_actual = actual[mask_idx]
        interval_predicted = predicted[mask_idx]
        interval_confidences = confidences[mask_idx]  # Confidence scores are 1D array
        
        # print(f"\nInterval {i+1} data shapes:")
        # print(f"  actual: {interval_actual.shape}")
        # print(f"  predicted: {interval_predicted.shape}")
        # print(f"  confidences: {interval_confidences.shape}")
        # print(f"  mask indices: {len(mask_idx)}")
        
        avg_confidence = np.mean(interval_confidences)
        sign_match_rate = calculate_sign_match_rate(interval_actual, interval_predicted)
        sample_count = len(interval_actual)
        
        # Print results
        interval_str = f"{i+1}/{n_intervals}"
        time_range = f"{time_boundaries[i]:.0f}-{time_boundaries[i+1]:.0f}"
        print(f"{interval_str:>10} {time_range:>25} {avg_confidence:>10.3f} {sign_match_rate:>10.3f} {sample_count:>8}")

def main():
    """Main function to analyze model predictions."""
    print("Analyzing Model Predictions Across Time Intervals")
    print("=" * 50)
    
    # Initialize client
    client = ModelAPIClient()
    
    # Load all data from H5 file
    print("\n1. Loading data from H5 file...")
    h5_path = Path("training_data.h5")
    if not h5_path.exists():
        print("Error: training_data.h5 not found in current directory.")
        return
    
    X, Y, T = load_h5_data(str(h5_path))
    if X is None or Y is None or T is None:
        print("Failed to load H5 data.")
        return
    
    print(f"\nLoaded {len(X)} samples")
    
    # Make predictions
    print("\n2. Making predictions...")
    result = client.predict(X.tolist())
    
    if "error" in result:
        print(f"Prediction failed: {result['error']}")
        return
    
    # Convert predictions and confidence scores
    predictions = np.array(result['predictions'])
    confidence_scores = np.array(result['confidences'])
    
    # Analyze predictions by time intervals
    print("\n3. Analyzing predictions by time intervals...")
    analyze_predictions_by_time(Y, predictions, confidence_scores, T)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
