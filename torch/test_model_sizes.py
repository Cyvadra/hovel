import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import gc
import argparse
import glob

# Import functions from the new optimized train.py
from train import (
    load_and_preprocess_data, 
    prepare_optimized_data, 
    train_model_optimized, 
    plot_losses, 
    OptimizedTrainingConfig,
    extract_final_predictions,
    OptimizedModel
)

def clear_gpu_memory():
    """Clear GPU memory and garbage collect."""
    if torch.cuda.is_available():
        # Print memory usage before clearing
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory before clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Print memory usage after clearing
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory after clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    # Force garbage collection
    gc.collect()

def check_existing_plots(model_name):
    """
    Check if plots for a specific model already exist.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        bool: True if all plots exist, False otherwise
    """
    plot_files = [
        f"{model_name}_loss_analysis.png",
        f"{model_name}_predictions.png"
    ]
    
    all_exist = all(os.path.exists(f) for f in plot_files)
    if all_exist:
        print(f"Plots for {model_name} already exist, skipping...")
    return all_exist

def check_existing_model_files(model_name):
    """
    Check if model files for a specific model already exist.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        bool: True if model file exists, False otherwise
    """
    model_file = f"{model_name}_best_model.pth"
    exists = os.path.exists(model_file)
    if exists:
        print(f"Model file for {model_name} already exists, skipping...")
    return exists

def show_gpu_memory_usage():
    """Display current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
    else:
        print("CUDA not available")

def plot_predictions(model, data_dict, output_dim, model_name="model"):
    """
    Plot predictions vs actual values for validation data.
    
    Args:
        model: Trained model
        data_dict: Data dictionary from prepare_optimized_data
        output_dim: Original output dimension
        model_name: Name for saving the plot
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get validation data
    X_val, Y_val = data_dict['val']
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_val)
        # Extract final predictions (p3 values)
        predictions = extract_final_predictions(predictions, output_dim)
    
    # Convert to numpy for plotting
    predictions = predictions.cpu().numpy()
    actual = Y_val.cpu().numpy()
    
    # Create subplots for each output dimension
    num_outputs = min(output_dim, 24)  # Limit to 24 for readability
    fig, axes = plt.subplots(6, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_outputs):
        ax = axes[i]
        ax.scatter(actual[:, i], predictions[:, i], alpha=0.6, s=20)
        ax.plot([actual[:, i].min(), actual[:, i].max()], 
                [actual[:, i].min(), actual[:, i].max()], 'r--', lw=2)
        ax.set_xlabel(f'Actual Y{i+1}')
        ax.set_ylabel(f'Predicted Y{i+1}')
        ax.set_title(f'Y{i+1}: Predictions vs Actual')
        ax.grid(True, alpha=0.3)
        
        # Calculate R² score
        correlation = np.corrcoef(actual[:, i], predictions[:, i])[0, 1]
        ax.text(0.05, 0.95, f'R² = {correlation:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_outputs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions plot saved to '{model_name}_predictions.png'")

def regenerate_all_plots(hidden_sizes=[1536, 1024, 512, 128], num_layers=[16, 8, 4, 2], test_split=0.05, val_split=0.05):
    """
    Regenerate plots for all existing models without retraining.
    
    Args:
        hidden_sizes (list): List of hidden layer sizes to check.
        num_layers (list): List of number of layers to check.
    """
    print("Regenerating plots for all existing models...")
    print(f"Checking hidden sizes: {hidden_sizes}")
    print(f"Checking number of layers: {num_layers}")
    
    # Load and prepare data (same for all models)
    X, Y, T = load_and_preprocess_data()
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Setup data preparation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = prepare_optimized_data(X, Y, T, batch_size=24, device=device, test_split=test_split, val_split=val_split)
    
    # Track which models were processed
    processed_models = []
    skipped_models = []
    
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            model_name = f"model_{hidden_size}_layers_{num_layer}"
            model_file = f"{model_name}_best_model.pth"
            
            if os.path.exists(model_file):
                print(f"\n{'='*50}")
                print(f"Regenerating plots for: {model_name}")
                print(f"{'='*50}")
                
                try:
                    # Load the trained model
                    state_dict = torch.load(model_file, map_location=device)
                    
                    # Create model instance with the same architecture
                    # Extract hidden_size and num_layers from model name
                    parts = model_name.split('_')
                    hidden_size = int(parts[1])  # model_1024_layers_4 -> 1024
                    num_layers = int(parts[3])   # model_1024_layers_4 -> 4
                    
                    print(f"Creating model with hidden_size={hidden_size}, num_layers={num_layers}")
                    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
                    
                    # Create model with same architecture
                    model = OptimizedModel(input_dim, output_dim, hidden_size, num_layers)
                    
                    # Load state dictionary
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    
                    # Load loss data if available
                    train_losses, val_losses = load_loss_data(model_name)
                    
                    # Regenerate plots
                    if train_losses is not None and val_losses is not None:
                        plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'], model_name)
                    else:
                        print(f"No loss data found for {model_name}, skipping loss plot")
                    
                    plot_predictions(model, data_dict, output_dim, model_name=model_name)
                    
                    processed_models.append(model_name)
                    print(f"Successfully regenerated plots for {model_name}")
                    
                except Exception as e:
                    print(f"Error regenerating plots for {model_name}: {e}")
                    skipped_models.append(model_name)
            else:
                print(f"Model file not found: {model_file}")
                skipped_models.append(model_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PLOT REGENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(processed_models)} models")
    print(f"Skipped/Failed: {len(skipped_models)} models")
    
    if processed_models:
        print(f"\nSuccessfully regenerated plots for:")
        for model in processed_models:
            print(f"  - {model}")
    
    if skipped_models:
        print(f"\nSkipped/Failed models:")
        for model in skipped_models:
            print(f"  - {model}")
    
    return processed_models, skipped_models

def regenerate_plots_by_pattern(pattern="model_*_layers_*_best_model.pth", test_split=0.05, val_split=0.05):
    """
    Regenerate plots for all models matching a specific pattern.
    
    Args:
        pattern (str): Glob pattern to match model files
    """
    print(f"Regenerating plots for models matching pattern: {pattern}")
    
    # Find all matching model files
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No model files found matching pattern: {pattern}")
        return [], []
    
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        print(f"  - {file}")
    
    # Load and prepare data
    X, Y, T = load_and_preprocess_data()
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Setup data preparation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = prepare_optimized_data(X, Y, T, batch_size=24, device=device, test_split=test_split, val_split=val_split)
    
    # Track which models were processed
    processed_models = []
    skipped_models = []
    
    for model_file in model_files:
        # Extract model name from file path
        model_name = os.path.splitext(os.path.basename(model_file))[0].replace("_best_model", "")
        
        print(f"\n{'='*50}")
        print(f"Regenerating plots for: {model_name}")
        print(f"Model file: {model_file}")
        print(f"{'='*50}")
        
        try:
            # Load the trained model
            state_dict = torch.load(model_file, map_location=device)
            
            # Create model instance with the same architecture
            # Extract hidden_size and num_layers from model name
            parts = model_name.split('_')
            hidden_size = int(parts[1])  # model_1024_layers_4 -> 1024
            num_layers = int(parts[3])   # model_1024_layers_4 -> 4
            
            print(f"Creating model with hidden_size={hidden_size}, num_layers={num_layers}")
            print(f"Input dim: {input_dim}, Output dim: {output_dim}")
            
            # Create model with same architecture
            model = OptimizedModel(input_dim, output_dim, hidden_size, num_layers)
            
            # Load state dictionary
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            # Load loss data if available
            train_losses, val_losses = load_loss_data(model_name)
            
            # Regenerate plots
            if train_losses is not None and val_losses is not None:
                plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'], model_name)
            else:
                print(f"No loss data found for {model_name}, skipping loss plot")
            
            plot_predictions(model, data_dict, output_dim, model_name=model_name)
            
            processed_models.append(model_name)
            print(f"Successfully regenerated plots for {model_name}")
            
        except Exception as e:
            print(f"Error regenerating plots for {model_name}: {e}")
            skipped_models.append(model_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PLOT REGENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(processed_models)} models")
    print(f"Skipped/Failed: {len(skipped_models)} models")
    
    if processed_models:
        print(f"\nSuccessfully regenerated plots for:")
        for model in processed_models:
            print(f"  - {model}")
    
    if skipped_models:
        print(f"\nSkipped/Failed models:")
        for model in skipped_models:
            print(f"  - {model}")
    
    return processed_models, skipped_models

def save_loss_data(train_losses, val_losses, model_name):
    """
    Save training and validation loss data to a JSON file.
    
    Args:
        train_losses (list): Training loss history
        val_losses (list): Validation loss history
        model_name (str): Name of the model
    """
    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'timestamp': datetime.now().isoformat()
    }
    
    loss_file = f"{model_name}_loss_data.json"
    with open(loss_file, 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"Loss data saved to '{loss_file}'")

def load_loss_data(model_name):
    """
    Load training and validation loss data from a JSON file.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (train_losses, val_losses) or (None, None) if file doesn't exist
    """
    loss_file = f"{model_name}_loss_data.json"
    
    if os.path.exists(loss_file):
        try:
            with open(loss_file, 'r') as f:
                loss_data = json.load(f)
            return loss_data['train_losses'], loss_data['val_losses']
        except Exception as e:
            print(f"Error loading loss data for {model_name}: {e}")
            return None, None
    else:
        return None, None

def test_model_sizes(hidden_sizes=[1536, 1024, 512, 128], num_layers=[16, 8, 4, 2], max_epochs=580, test_split=0.05, val_split=0.05):
    """
    Test different model sizes and save all results.
    
    Args:
        hidden_sizes (list): List of hidden layer sizes to test.
        num_layers (list): List of number of layers to test.
    """
    print("Starting model size comparison test...")
    print(f"Testing hidden sizes: {hidden_sizes}")
    print(f"Testing number of layers: {num_layers}")
    
    # Show initial GPU memory usage
    print("\nInitial GPU memory status:")
    show_gpu_memory_usage()
    
    # Load and prepare data
    X, Y, T = load_and_preprocess_data()
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Setup data preparation (same for all models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = prepare_optimized_data(X, Y, T, batch_size=24, device=device, test_split=test_split, val_split=val_split)
    
    # Store results for comparison
    results = {}
    
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            print(f"\n{'='*50}")
            print(f"Testing model with hidden size: {hidden_size}, layers: {num_layer}")
            print(f"{'='*50}")
            
            # Show current GPU memory usage
            show_gpu_memory_usage()
            
            # Create model name for file naming
            model_name = f"model_{hidden_size}_layers_{num_layer}"
            
            # Check if model files and plots already exist
            if check_existing_model_files(model_name) and check_existing_plots(model_name):
                print(f"Skipping {model_name} - model files and plots already exist")
                continue
            
            try:
                # Create configuration for this model
                config = OptimizedTrainingConfig()
                config.update(
                    hidden_size=hidden_size,
                    num_layers=num_layer,
                    batch_size=24,
                    max_epochs=max_epochs,
                    patience=max_epochs  # No early stopping
                )
                
                # Train the model
                model, train_losses, val_losses, test_loss = train_model_optimized(
                    data_dict, input_dim, output_dim,
                    hidden_size=hidden_size, num_layers=num_layer, model_name=model_name,
                    config=config
                )
                
                # Save loss data
                save_loss_data(train_losses, val_losses, model_name)
                
                # Plot results
                plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'], model_name)
                plot_predictions(model, data_dict, output_dim, model_name=model_name)
                
                # Store results
                key = f"{hidden_size}_{num_layer}"
                results[key] = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layer,
                    'final_train_loss': train_losses[-1],
                    'final_val_loss': val_losses[-1],
                    'final_test_loss': test_loss,
                    'best_val_loss': min(val_losses),
                    'epochs_trained': len(train_losses),
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                
                print(f"Model {hidden_size} layers {num_layer} completed successfully!")
                print(f"Final train loss: {train_losses[-1]:.6f}")
                print(f"Final val loss: {val_losses[-1]:.6f}")
                print(f"Final test loss: {test_loss:.6f}")
                print(f"Best val loss: {min(val_losses):.6f}")
                print(f"Epochs trained: {len(train_losses)}")
                
            except Exception as e:
                print(f"Error training model with hidden size {hidden_size}, layers {num_layer}: {e}")
                key = f"{hidden_size}_{num_layer}"
                results[key] = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layer,
                    'error': str(e)
                }
            finally:
                # Clear GPU memory after each model training
                print("Clearing GPU memory...")
                clear_gpu_memory()
    
    # Create comparison plots
    create_comparison_plots(results, hidden_sizes, num_layers)
    
    # Save results to JSON
    save_results(results, hidden_sizes, num_layers)
    
    # Print summary
    print_summary(results, hidden_sizes, num_layers)
    
    return results

def create_comparison_plots(results, hidden_sizes, num_layers):
    """Create comparison plots for all models."""
    
    # Filter out failed models
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("No successful models to compare!")
        return
    
    # Create a larger figure for better visualization
    plt.figure(figsize=(16, 12))
    
    # 1. Training loss comparison
    plt.subplot(2, 3, 1)
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                plt.plot(successful_results[key]['train_losses'], 
                        label=f'{hidden_size}h_{num_layer}l', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 2. Validation loss comparison
    plt.subplot(2, 3, 2)
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                plt.plot(successful_results[key]['val_losses'], 
                        label=f'{hidden_size}h_{num_layer}l', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 3. Final loss comparison (heatmap style)
    plt.subplot(2, 3, 3)
    final_val_losses = np.zeros((len(hidden_sizes), len(num_layers)))
    final_val_losses.fill(np.nan)
    
    for i, hidden_size in enumerate(hidden_sizes):
        for j, num_layer in enumerate(num_layers):
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                final_val_losses[i, j] = successful_results[key]['final_val_loss']
    
    im = plt.imshow(final_val_losses, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Final Validation Loss')
    plt.xlabel('Number of Layers')
    plt.ylabel('Hidden Size')
    plt.title('Final Validation Loss Heatmap')
    plt.xticks(range(len(num_layers)), num_layers)
    plt.yticks(range(len(hidden_sizes)), hidden_sizes)
    
    # Add text annotations
    for i in range(len(hidden_sizes)):
        for j in range(len(num_layers)):
            if not np.isnan(final_val_losses[i, j]):
                plt.text(j, i, f'{final_val_losses[i, j]:.4f}', 
                        ha='center', va='center', color='white', fontweight='bold')
    
    # 4. Best validation loss comparison
    plt.subplot(2, 3, 4)
    best_val_losses = np.zeros((len(hidden_sizes), len(num_layers)))
    best_val_losses.fill(np.nan)
    
    for i, hidden_size in enumerate(hidden_sizes):
        for j, num_layer in enumerate(num_layers):
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                best_val_losses[i, j] = successful_results[key]['best_val_loss']
    
    im2 = plt.imshow(best_val_losses, cmap='plasma', aspect='auto')
    plt.colorbar(im2, label='Best Validation Loss')
    plt.xlabel('Number of Layers')
    plt.ylabel('Hidden Size')
    plt.title('Best Validation Loss Heatmap')
    plt.xticks(range(len(num_layers)), num_layers)
    plt.yticks(range(len(hidden_sizes)), hidden_sizes)
    
    # Add text annotations
    for i in range(len(hidden_sizes)):
        for j in range(len(num_layers)):
            if not np.isnan(best_val_losses[i, j]):
                plt.text(j, i, f'{best_val_losses[i, j]:.4f}', 
                        ha='center', va='center', color='white', fontweight='bold')
    
    # 5. Epochs trained comparison
    plt.subplot(2, 3, 5)
    epochs_trained = np.zeros((len(hidden_sizes), len(num_layers)))
    epochs_trained.fill(np.nan)
    
    for i, hidden_size in enumerate(hidden_sizes):
        for j, num_layer in enumerate(num_layers):
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                epochs_trained[i, j] = successful_results[key]['epochs_trained']
    
    im3 = plt.imshow(epochs_trained, cmap='coolwarm', aspect='auto')
    plt.colorbar(im3, label='Epochs Trained')
    plt.xlabel('Number of Layers')
    plt.ylabel('Hidden Size')
    plt.title('Epochs Trained Heatmap')
    plt.xticks(range(len(num_layers)), num_layers)
    plt.yticks(range(len(hidden_sizes)), hidden_sizes)
    
    # Add text annotations
    for i in range(len(hidden_sizes)):
        for j in range(len(num_layers)):
            if not np.isnan(epochs_trained[i, j]):
                plt.text(j, i, f'{int(epochs_trained[i, j])}', 
                        ha='center', va='center', color='white', fontweight='bold')
    
    # 6. Model complexity comparison (hidden_size * num_layers)
    plt.subplot(2, 3, 6)
    model_complexity = np.zeros((len(hidden_sizes), len(num_layers)))
    for i, hidden_size in enumerate(hidden_sizes):
        for j, num_layer in enumerate(num_layers):
            model_complexity[i, j] = hidden_size * num_layer
    
    im4 = plt.imshow(model_complexity, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im4, label='Model Complexity (h×l)')
    plt.xlabel('Number of Layers')
    plt.ylabel('Hidden Size')
    plt.title('Model Complexity Heatmap')
    plt.xticks(range(len(num_layers)), num_layers)
    plt.yticks(range(len(hidden_sizes)), hidden_sizes)
    
    # Add text annotations
    for i in range(len(hidden_sizes)):
        for j in range(len(num_layers)):
            plt.text(j, i, f'{int(model_complexity[i, j])}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_size_comparison.png', dpi=360, bbox_inches='tight')
    plt.close()
    
    print("Comparison plots saved to 'model_size_comparison.png'")

def save_results(results, hidden_sizes, num_layers):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, result in results.items():
        if 'error' not in result:
            json_results[key] = {
                'hidden_size': result['hidden_size'],
                'num_layers': result['num_layers'],
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']),
                'final_test_loss': float(result['final_test_loss']),
                'best_val_loss': float(result['best_val_loss']),
                'epochs_trained': result['epochs_trained']
            }
        else:
            json_results[key] = {
                'hidden_size': result['hidden_size'],
                'num_layers': result['num_layers'],
                'error': result['error']
            }
    
    # Add metadata
    json_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'hidden_sizes_tested': hidden_sizes,
        'num_layers_tested': num_layers,
        'input_dim': results.get('input_dim', 'unknown'),
        'output_dim': results.get('output_dim', 'unknown')
    }
    
    with open('model_size_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("Results saved to 'model_size_results.json'")

def print_summary(results, hidden_sizes, num_layers):
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
    
    print(f"\nBest performing model: Hidden size {best_model[1]['hidden_size']}, Layers {best_model[1]['num_layers']}")
    print(f"Best validation loss: {best_model[1]['best_val_loss']:.6f}")
    print(f"Final validation loss: {best_model[1]['final_val_loss']:.6f}")
    print(f"Epochs trained: {best_model[1]['epochs_trained']}")
    
    print(f"\nDetailed Results:")
    print(f"{'Hidden Size':<12} {'Layers':<8} {'Best Val Loss':<15} {'Final Val Loss':<15} {'Epochs':<8}")
    print("-" * 70)
    
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                result = successful_results[key]
                print(f"{result['hidden_size']:<12} {result['num_layers']:<8} {result['best_val_loss']:<15.6f} {result['final_val_loss']:<15.6f} {result['epochs_trained']:<8}")
            else:
                print(f"{hidden_size:<12} {num_layer:<8} {'ERROR':<15} {'ERROR':<15} {'ERROR':<8}")
    
    print(f"\nFiles generated:")
    print("- model_size_comparison.png (comparison plots)")
    print("- model_size_results.json (detailed results)")
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            key = f"{hidden_size}_{num_layer}"
            if key in successful_results:
                print(f"- model_{hidden_size}_layers_{num_layer}_loss_analysis.png")
                print(f"- model_{hidden_size}_layers_{num_layer}_predictions.png")
                print(f"- model_{hidden_size}_layers_{num_layer}_best_model.pth")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Test different model sizes and regenerate plots')
    parser.add_argument('--regenerate-plots', action='store_true', 
                       help='Regenerate plots for all existing models without retraining')
    parser.add_argument('--regenerate-pattern', type=str, default="model_*_layers_*_best_model.pth",
                       help='Glob pattern to match model files for plot regeneration')
    # Example usage:
    # python test_model_sizes.py --hidden-sizes 2048 256 128 --num-layers 12 8 4
    # This will test models with hidden sizes [2048,256,128] and layers [12,8,4]
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[2048, 256],
                       help='List of hidden layer sizes to test')
    parser.add_argument('--num-layers', nargs='+', type=int, default=[12],
                       help='List of number of layers to test')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only regenerate plots, do not train new models')
    parser.add_argument('--max-epochs', type=int, default=580,
                       help='Maximum number of epochs to train')
    parser.add_argument('--test-split', type=float, default=0.05,
                       help='Fraction of data to use for testing')
    parser.add_argument('--val-split', type=float, default=0.05,
                       help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    
    if args.regenerate_plots or args.plot_only:
        # Regenerate plots for existing models
        if args.plot_only:
            print("Plot-only mode: Regenerating plots for all existing models...")
            processed, skipped = regenerate_plots_by_pattern(args.regenerate_pattern, args.test_split, args.val_split)
        else:
            print("Regenerating plots for specified model sizes...")
            processed, skipped = regenerate_all_plots(args.hidden_sizes, args.num_layers, args.test_split, args.val_split)
        
        if not args.plot_only:
            print("\nContinuing with normal training after plot regeneration...")
            results = test_model_sizes(args.hidden_sizes, args.num_layers, args.max_epochs, args.test_split, args.val_split)
    else:
        # Normal training mode
        print("Starting normal model training...")
        results = test_model_sizes(args.hidden_sizes, args.num_layers, args.max_epochs, args.test_split, args.val_split) 