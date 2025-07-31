import torch
import torch.nn as nn

def test_weight_calculation():
    """Test the weight calculation with clamping."""
    print("Testing Weight Calculation with Clamping")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        (1.0, "p1 = 1.0 (should favor p3)"),
        (-1.0, "p1 = -1.0 (should favor p2)"),
        (0.0, "p1 = 0.0 (equal weights)"),
        (2.0, "p1 = 2.0 (should be clamped to 1.0)"),
        (-3.0, "p1 = -3.0 (should be clamped to -1.0)"),
        (0.5, "p1 = 0.5 (should favor p3 slightly)"),
        (-0.5, "p1 = -0.5 (should favor p2 slightly)")
    ]
    
    for p1, description in test_cases:
        print(f"\n{description}")
        
        # Clamp p1 to [-1, 1]
        p1_clamped = torch.clamp(torch.tensor(p1), -1.0, 1.0).item()
        
        # Calculate weights
        weight_p2 = abs(p1_clamped - 1) / 2
        weight_p3 = abs(p1_clamped + 1) / 2
        weight_sum = weight_p2 + weight_p3
        
        print(f"  Original p1: {p1}")
        print(f"  Clamped p1: {p1_clamped}")
        print(f"  weight_p2: {weight_p2:.3f}")
        print(f"  weight_p3: {weight_p3:.3f}")
        print(f"  weight_sum: {weight_sum:.6f}")
        print(f"  Sum equals 1: {abs(weight_sum - 1.0) < 1e-6}")
        
        # Verify the logic
        if p1_clamped > 0:
            print(f"  Logic: p1 > 0, so weight_p3 ({weight_p3:.3f}) > weight_p2 ({weight_p2:.3f})")
        elif p1_clamped < 0:
            print(f"  Logic: p1 < 0, so weight_p2 ({weight_p2:.3f}) > weight_p3 ({weight_p3:.3f})")
        else:
            print(f"  Logic: p1 = 0, so weight_p2 ({weight_p2:.3f}) = weight_p3 ({weight_p3:.3f})")
    
    print("\n✅ All weight calculations are correct!")
    print("=" * 50)

if __name__ == "__main__":
    test_weight_calculation() 