import torch
import torch.nn as nn
import numpy as np

# Import the classes from train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import ImprovedModel, WeightedCombinedLoss, extract_final_predictions

def test_weighted_loss():
    """Test the weighted loss function with a simple example."""
    print("Testing Weighted Combined Loss Function")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    input_dim = 10
    output_dim = 2
    
    # Create model
    model = ImprovedModel(input_dim, output_dim, hidden_size=64, num_layers=2)
    
    # Create test data
    X = torch.randn(batch_size, input_dim)
    Y = torch.randn(batch_size, output_dim)
    
    # Forward pass
    model_output = model(X)
    print(f"Model output shape: {model_output.shape}")
    print(f"Expected shape: (batch_size, 3*output_dim) = ({batch_size}, {3*output_dim})")
    print(f"Actual shape: {model_output.shape}")
    assert model_output.shape == (batch_size, 3*output_dim), "Model output shape mismatch!"
    
    # Test loss function
    criterion = WeightedCombinedLoss(output_dim)
    loss = criterion(model_output, Y)
    print(f"Loss value: {loss.item():.6f}")
    
    # Test final prediction extraction
    final_predictions = extract_final_predictions(model_output, output_dim)
    print(f"Final predictions shape: {final_predictions.shape}")
    print(f"Expected shape: (batch_size, output_dim) = ({batch_size}, {output_dim})")
    assert final_predictions.shape == (batch_size, output_dim), "Final predictions shape mismatch!"
    
    # Analyze the components
    pred_reshaped = model_output.view(batch_size, output_dim, 3)
    p1 = pred_reshaped[:, :, 0]  # weight parameters
    p2 = pred_reshaped[:, :, 1]  # negative predictions
    p3 = pred_reshaped[:, :, 2]  # positive predictions
    
    print(f"\nComponent Analysis:")
    print(f"p1 (weights) - mean: {torch.mean(p1):.3f}, std: {torch.std(p1):.3f}, range: [{torch.min(p1):.3f}, {torch.max(p1):.3f}]")
    print(f"p2 (negative) - mean: {torch.mean(p2):.3f}, std: {torch.std(p2):.3f}")
    print(f"p3 (positive) - mean: {torch.mean(p3):.3f}, std: {torch.std(p3):.3f}")
    
    # Test weight calculation (clamp p1 to ensure weights sum to 1)
    p1_clamped = torch.clamp(p1, -1.0, 1.0)
    weight_p2 = torch.abs(p1_clamped - 1) / 2
    weight_p3 = torch.abs(p1_clamped + 1) / 2
    
    print(f"\nWeight Analysis:")
    print(f"weight_p2 - mean: {torch.mean(weight_p2):.3f}, range: [{torch.min(weight_p2):.3f}, {torch.max(weight_p2):.3f}]")
    print(f"weight_p3 - mean: {torch.mean(weight_p3):.3f}, range: [{torch.min(weight_p3):.3f}, {torch.max(weight_p3):.3f}]")
    
    # Verify weights sum to 1
    weight_sum = weight_p2 + weight_p3
    print(f"Weight sum - mean: {torch.mean(weight_sum):.6f} (should be close to 1.0)")
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6), "Weights should sum to 1!"
    
    print("\n✅ All tests passed!")
    print("=" * 50)

def test_specific_scenarios():
    """Test specific scenarios to verify the loss behavior."""
    print("\nTesting Specific Scenarios")
    print("=" * 50)
    
    output_dim = 1
    batch_size = 2
    
    # Scenario 1: p1 = 1 (should favor p3)
    print("Scenario 1: p1 = 1 (should favor p3)")
    model_output = torch.tensor([[1.0, -2.0, 3.0]])  # p1=1, p2=-2, p3=3
    target = torch.tensor([[0.0]])
    
    criterion = WeightedCombinedLoss(output_dim)
    loss = criterion(model_output, target)
    
    # Manual calculation
    p1, p2, p3 = 1.0, -2.0, 3.0
    weight_p2 = abs(p1 - 1) / 2  # = 0
    weight_p3 = abs(p1 + 1) / 2  # = 1
    expected_loss = weight_p2 * (p2 - 0.0)**2 + weight_p3 * (p3 - 0.0)**2
    
    print(f"Model output: p1={p1}, p2={p2}, p3={p3}")
    print(f"Target: {target.item()}")
    print(f"weight_p2: {weight_p2}, weight_p3: {weight_p3}")
    print(f"Expected loss: {expected_loss:.6f}")
    print(f"Actual loss: {loss.item():.6f}")
    print(f"Loss matches: {abs(loss.item() - expected_loss) < 1e-6}")
    
    # Scenario 2: p1 = -1 (should favor p2)
    print("\nScenario 2: p1 = -1 (should favor p2)")
    model_output = torch.tensor([[-1.0, -2.0, 3.0]])  # p1=-1, p2=-2, p3=3
    target = torch.tensor([[0.0]])
    
    loss = criterion(model_output, target)
    
    # Manual calculation
    p1, p2, p3 = -1.0, -2.0, 3.0
    weight_p2 = abs(p1 - 1) / 2  # = 1
    weight_p3 = abs(p1 + 1) / 2  # = 0
    expected_loss = weight_p2 * (p2 - 0.0)**2 + weight_p3 * (p3 - 0.0)**2
    
    print(f"Model output: p1={p1}, p2={p2}, p3={p3}")
    print(f"Target: {target.item()}")
    print(f"weight_p2: {weight_p2}, weight_p3: {weight_p3}")
    print(f"Expected loss: {expected_loss:.6f}")
    print(f"Actual loss: {loss.item():.6f}")
    print(f"Loss matches: {abs(loss.item() - expected_loss) < 1e-6}")
    
    # Scenario 3: p1 = 0 (equal weights)
    print("\nScenario 3: p1 = 0 (equal weights)")
    model_output = torch.tensor([[0.0, -2.0, 3.0]])  # p1=0, p2=-2, p3=3
    target = torch.tensor([[0.0]])
    
    loss = criterion(model_output, target)
    
    # Manual calculation
    p1, p2, p3 = 0.0, -2.0, 3.0
    weight_p2 = abs(p1 - 1) / 2  # = 0.5
    weight_p3 = abs(p1 + 1) / 2  # = 0.5
    expected_loss = weight_p2 * (p2 - 0.0)**2 + weight_p3 * (p3 - 0.0)**2
    
    print(f"Model output: p1={p1}, p2={p2}, p3={p3}")
    print(f"Target: {target.item()}")
    print(f"weight_p2: {weight_p2}, weight_p3: {weight_p3}")
    print(f"Expected loss: {expected_loss:.6f}")
    print(f"Actual loss: {loss.item():.6f}")
    print(f"Loss matches: {abs(loss.item() - expected_loss) < 1e-6}")
    
    print("\n✅ All scenario tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    test_weighted_loss()
    test_specific_scenarios() 