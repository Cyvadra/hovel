import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import load_and_preprocess_data, setup_training, ImprovedModel, WeightedCombinedLoss

def test_data_types():
    """Test that data types are correct throughout the pipeline."""
    print("Testing Data Types")
    print("=" * 50)
    
    # Test data loading and preprocessing
    print("1. Testing data loading and preprocessing...")
    X, Y = load_and_preprocess_data()
    
    print(f"X dtype: {X.dtype}, Y dtype: {Y.dtype}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Check that both are float32
    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"
    assert Y.dtype == np.float32, f"Y should be float32, got {Y.dtype}"
    
    print("✅ Data preprocessing types are correct!")
    
    # Test data loader setup
    print("\n2. Testing data loader setup...")
    train_loader, val_loader, test_loader = setup_training(X, Y, batch_size=16)
    
    # Test a batch from each loader
    for loader_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        for batch_idx, (inputs, targets) in enumerate(loader):
            print(f"{loader_name} batch {batch_idx}:")
            print(f"  inputs dtype: {inputs.dtype}, shape: {inputs.shape}")
            print(f"  targets dtype: {targets.dtype}, shape: {targets.shape}")
            
            # Check that both are float32
            assert inputs.dtype == torch.float32, f"{loader_name} inputs should be float32, got {inputs.dtype}"
            assert targets.dtype == torch.float32, f"{loader_name} targets should be float32, got {targets.dtype}"
            break
    
    print("✅ Data loader types are correct!")
    
    # Test model creation and forward pass
    print("\n3. Testing model creation and forward pass...")
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    model = ImprovedModel(input_dim, output_dim, hidden_size=64, num_layers=2)
    
    # Test with a batch
    for inputs, targets in train_loader:
        print(f"Model input dtype: {inputs.dtype}, shape: {inputs.shape}")
        print(f"Model target dtype: {targets.dtype}, shape: {targets.shape}")
        
        # Forward pass
        outputs = model(inputs)
        print(f"Model output dtype: {outputs.dtype}, shape: {outputs.shape}")
        
        # Check output shape
        expected_output_shape = (inputs.shape[0], 3 * output_dim)
        assert outputs.shape == expected_output_shape, f"Expected {expected_output_shape}, got {outputs.shape}"
        
        break
    
    print("✅ Model forward pass works correctly!")
    
    # Test loss function
    print("\n4. Testing loss function...")
    criterion = WeightedCombinedLoss(output_dim)
    
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        print(f"Loss value: {loss.item():.6f}")
        print(f"Loss dtype: {loss.dtype}")
        
        assert loss.dtype == torch.float32, f"Loss should be float32, got {loss.dtype}"
        break
    
    print("✅ Loss function works correctly!")
    
    print("\n🎉 All data type tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    test_data_types() 