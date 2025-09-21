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

def analyze_predictions_by_confidence(
    actual: np.ndarray,
    predicted: np.ndarray,
    confidences: np.ndarray,
    n_intervals: int = 20
) -> None:
    """
    Analyze predictions by dividing them into confidence intervals.
    
    Args:
        actual: Array of actual values (n_samples, n_targets)
        predicted: Array of predicted values (n_samples, n_targets)
        confidences: Array of confidence scores (n_samples,)
        n_intervals: Number of intervals to divide the confidence range into
    """
    # Calculate percentile boundaries (5%, 10%, ..., 95%, 100%)
    percentiles = np.linspace(0, 100, n_intervals + 1)
    confidence_boundaries = np.percentile(confidences, percentiles)
    
    print("\nAnalysis by Confidence Intervals:")
    print("=" * 60)
    print(f"{'Interval':>10} {'Conf Range':>20} {'Avg Conf':>10} {'Sign Match':>10}")
    print("-" * 60)
    
    for i in range(n_intervals):
        # Get data for current interval
        if i == n_intervals - 1:
            mask = (confidences >= confidence_boundaries[i])
        else:
            mask = (confidences >= confidence_boundaries[i]) & (confidences < confidence_boundaries[i+1])
        
        if not np.any(mask):
            continue
        
        # Calculate metrics for this interval
        interval_confidences = confidences[mask]
        interval_actual = actual[mask]
        interval_predicted = predicted[mask]
        
        avg_confidence = np.mean(interval_confidences)
        sign_match_rate = calculate_sign_match_rate(interval_actual, interval_predicted)
        
        # Print results
        interval_str = f"{i+1}/{n_intervals}"
        conf_range = f"{confidence_boundaries[i]:.3f}-{confidence_boundaries[i+1]:.3f}"
        print(f"{interval_str:>10} {conf_range:>20} {avg_confidence:>10.3f} {sign_match_rate:>10.3f}")

def main():
    """Main function to analyze model predictions."""
    print("Analyzing Model Predictions Across Confidence Intervals")
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
    
    # Analyze predictions by confidence intervals
    print("\n3. Analyzing predictions by confidence intervals...")
    analyze_predictions_by_confidence(Y, predictions, confidence_scores)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
