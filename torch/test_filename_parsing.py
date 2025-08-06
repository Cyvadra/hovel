#!/usr/bin/env python3
"""
Test script for filename parsing functionality

This script tests the _parse_model_params_from_filename method to ensure
it correctly extracts hidden_size and num_layers from various filename formats.
"""

import os
import re
import sys

# Add the current directory to Python path to import from train.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_service import ModelManager

def test_filename_parsing():
    """Test the filename parsing functionality."""
    
    # Test cases with expected results
    test_cases = [
        # (filename, expected_hidden_size, expected_num_layers)
        ("model_1024_layers_16_best_model.pth", 1024, 16),
        ("model_512_layers_4_best_model.pth", 512, 4),
        ("model_256_layers_8_best_model.pth", 256, 8),
        ("model_1024_16_best_model.pth", 1024, 16),
        ("model_512_4_best_model.pth", 512, 4),
        ("model_1024_layers_16.pth", 1024, 16),
        ("model_512_layers_4.pth", 512, 4),
        ("model_1024_16.pth", 1024, 16),
        ("model_512_4.pth", 512, 4),
        # Default cases (should return 512, 4)
        ("optimized_model_best_model.pth", 512, 4),
        ("my_model.pth", 512, 4),
        ("model.pth", 512, 4),
        ("best_model.pth", 512, 4),
        # Edge cases
        ("model_1_layers_1_best_model.pth", 1, 1),
        ("model_9999_layers_999_best_model.pth", 9999, 999),
    ]
    
    print("Testing filename parsing functionality")
    print("=" * 50)
    
    # Create a temporary ModelManager instance just for testing
    temp_manager = ModelManager("dummy_path.pth")
    
    passed = 0
    failed = 0
    
    for filename, expected_hidden_size, expected_num_layers in test_cases:
        try:
            hidden_size, num_layers = temp_manager._parse_model_params_from_filename(filename)
            
            if hidden_size == expected_hidden_size and num_layers == expected_num_layers:
                print(f"✅ PASS: {filename}")
                print(f"   Expected: hidden_size={expected_hidden_size}, num_layers={expected_num_layers}")
                print(f"   Got:      hidden_size={hidden_size}, num_layers={num_layers}")
                passed += 1
            else:
                print(f"❌ FAIL: {filename}")
                print(f"   Expected: hidden_size={expected_hidden_size}, num_layers={expected_num_layers}")
                print(f"   Got:      hidden_size={hidden_size}, num_layers={num_layers}")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {filename} - {e}")
            failed += 1
        
        print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

def test_with_real_files():
    """Test with actual model files if they exist."""
    print("\nTesting with actual model files")
    print("=" * 50)
    
    # Look for model files in the current directory
    model_files = []
    for file in os.listdir('.'):
        if file.endswith('.pth') and 'model' in file.lower():
            model_files.append(file)
    
    if not model_files:
        print("No model files found in current directory")
        return
    
    temp_manager = ModelManager("dummy_path.pth")
    
    for filename in model_files:
        try:
            hidden_size, num_layers = temp_manager._parse_model_params_from_filename(filename)
            print(f"📁 {filename}")
            print(f"   Parsed: hidden_size={hidden_size}, num_layers={num_layers}")
            print()
        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")
            print()

if __name__ == "__main__":
    # Test the parsing functionality
    success = test_filename_parsing()
    
    # Test with real files if available
    test_with_real_files()
    
    if success:
        print("\n✅ Filename parsing functionality is working correctly!")
    else:
        print("\n❌ Filename parsing functionality has issues!")
        sys.exit(1) 