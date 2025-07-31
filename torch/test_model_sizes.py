import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from datetime import datetime

# Import functions from the original train.py
from train import load_and_preprocess_data, setup_training, train_model, plot_losses, plot_predictions

def test_model_sizes(hidden_sizes=[128, 256, 512, 1024]):
    """
    Test different model sizes and save all results.
    
    Args:
        hidden_sizes (list): List of hidden layer sizes to test.
    """
    print("Starting model size comparison test...")
    print(f"Testing hidden sizes: {hidden_sizes}")
    
    # Load and prepare data
    X, Y = load_and_preprocess_data()
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Setup data loaders (same for all models)
    train_loader, val_loader, test_loader = setup_training(X, Y, batch_size=32)
    
    # Store results for comparison
    results = {}
    
    for hidden_size in hidden_sizes:
        print(f"\n{'='*50}")
        print(f"Testing model with hidden size: {hidden_size}")
        print(f"{'='*50}")
        
        # Create model name for file naming
        model_name = f"model_{hidden_size}"
        
        try:
            # Train the model
            model, train_losses, val_losses, test_loss = train_model(
                train_loader, val_loader, test_loader, input_dim, output_dim, 
                hidden_size=hidden_size, model_name=model_name
            )
            
            # Plot results
            plot_losses(train_losses, val_losses, model_name)
            plot_predictions(model, val_loader, output_dim, model_name=model_name)
            
            # Store results
            results[hidden_size] = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_test_loss': test_loss,
                'best_val_loss': min(val_losses),
                'epochs_trained': len(train_losses),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            print(f"Model {hidden_size} completed successfully!")
            print(f"Final train loss: {train_losses[-1]:.6f}")
            print(f"Final val loss: {val_losses[-1]:.6f}")
            print(f"Final test loss: {test_loss:.6f}")
            print(f"Best val loss: {min(val_losses):.6f}")
            print(f"Epochs trained: {len(train_losses)}")
            
        except Exception as e:
            print(f"Error training model with hidden size {hidden_size}: {e}")
            results[hidden_size] = {'error': str(e)}
    
    # Create comparison plots
    create_comparison_plots(results, hidden_sizes)
    
    # Save results to JSON
    save_results(results, hidden_sizes)
    
    # Print summary
    print_summary(results, hidden_sizes)
    
    return results

def create_comparison_plots(results, hidden_sizes):
    """Create comparison plots for all models."""
    
    # Filter out failed models
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("No successful models to compare!")
        return
    
    # 1. Loss comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for hidden_size in hidden_sizes:
        if hidden_size in successful_results:
            plt.plot(successful_results[hidden_size]['train_losses'], 
                    label=f'Train {hidden_size}', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for hidden_size in hidden_sizes:
        if hidden_size in successful_results:
            plt.plot(successful_results[hidden_size]['val_losses'], 
                    label=f'Val {hidden_size}', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # 3. Final loss comparison
    plt.subplot(2, 2, 3)
    sizes = list(successful_results.keys())
    final_train_losses = [successful_results[s]['final_train_loss'] for s in sizes]
    final_val_losses = [successful_results[s]['final_val_loss'] for s in sizes]
    final_test_losses = [successful_results[s]['final_test_loss'] for s in sizes]
    
    x = np.arange(len(sizes))
    width = 0.25
    
    plt.bar(x - width, final_train_losses, width, label='Final Train Loss', alpha=0.8)
    plt.bar(x, final_val_losses, width, label='Final Val Loss', alpha=0.8)
    plt.bar(x + width, final_test_losses, width, label='Final Test Loss', alpha=0.8)
    plt.xlabel('Hidden Size')
    plt.ylabel('Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Best validation loss comparison
    plt.subplot(2, 2, 4)
    best_val_losses = [successful_results[s]['best_val_loss'] for s in sizes]
    epochs_trained = [successful_results[s]['epochs_trained'] for s in sizes]
    
    bars = plt.bar(sizes, best_val_losses, alpha=0.7, color='skyblue', label='Best Val Loss')
    plt.xlabel('Hidden Size')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Validation Loss by Model Size')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars, best_val_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plots saved to 'model_size_comparison.png'")

def save_results(results, hidden_sizes):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for hidden_size, result in results.items():
        if 'error' not in result:
            json_results[hidden_size] = {
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']),
                'final_test_loss': float(result['final_test_loss']),
                'best_val_loss': float(result['best_val_loss']),
                'epochs_trained': result['epochs_trained']
            }
        else:
            json_results[hidden_size] = {'error': result['error']}
    
    # Add metadata
    json_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'hidden_sizes_tested': hidden_sizes,
        'input_dim': results.get('input_dim', 'unknown'),
        'output_dim': results.get('output_dim', 'unknown')
    }
    
    with open('model_size_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("Results saved to 'model_size_results.json'")

def print_summary(results, hidden_sizes):
    """Print a summary of all results."""
    print(f"\n{'='*60}")
    print("MODEL SIZE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("No successful models to summarize!")
        return
    
    # Find best model
    best_model = min(successful_results.items(), key=lambda x: x[1]['best_val_loss'])
    
    print(f"\nBest performing model: Hidden size {best_model[0]}")
    print(f"Best validation loss: {best_model[1]['best_val_loss']:.6f}")
    print(f"Final validation loss: {best_model[1]['final_val_loss']:.6f}")
    print(f"Epochs trained: {best_model[1]['epochs_trained']}")
    
    print(f"\nDetailed Results:")
    print(f"{'Hidden Size':<12} {'Best Val Loss':<15} {'Final Val Loss':<15} {'Epochs':<8}")
    print("-" * 55)
    
    for hidden_size in hidden_sizes:
        if hidden_size in successful_results:
            result = successful_results[hidden_size]
            print(f"{hidden_size:<12} {result['best_val_loss']:<15.6f} {result['final_val_loss']:<15.6f} {result['epochs_trained']:<8}")
        else:
            print(f"{hidden_size:<12} {'ERROR':<15} {'ERROR':<15} {'ERROR':<8}")
    
    print(f"\nFiles generated:")
    print("- model_size_comparison.png (comparison plots)")
    print("- model_size_results.json (detailed results)")
    for hidden_size in hidden_sizes:
        if hidden_size in successful_results:
            print(f"- model_{hidden_size}_loss_curve.png")
            print(f"- model_{hidden_size}_predictions.png")
            print(f"- model_{hidden_size}_best_model.pth")

if __name__ == "__main__":
    # Test the specified model sizes
    hidden_sizes = [128, 256, 512, 1024]
    results = test_model_sizes(hidden_sizes) 