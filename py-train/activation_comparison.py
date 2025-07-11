import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from train import build_model, split_data, train_model_advanced, evaluate_model, warmup_training

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 确保使用 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data():
    """Load data"""
    with h5py.File('training_data.h5', 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)
    
    Y = tf.clip_by_value(Y, clip_value_min=-90.0, clip_value_max=90.0)
    Y = Y.numpy()
    return X, Y

def compare_activations():
    """Compare different activation functions"""
    # Load data
    X, Y = load_data()
    print(f"Data shape: X {X.shape}, Y {Y.shape}")
    
    # Data splitting
    x_train, x_val, y_train, y_val = split_data(X, Y)
    
    # Activation functions to test
    activation_types = ['relu', 'elu', 'swish', 'gelu', 'softmax', 'mixed']
    results = {}
    
    for activation_type in activation_types:
        print(f"\n{'='*50}")
        print(f"Test Activation Function: {activation_type.upper()}")
        print(f"{'='*50}")
        
        # Build model
        model = build_model((x_train.shape[1],), y_train.shape[1], activation_type, optimizer_type='adamw')
        
        # Warmup training
        print(f"Starting {activation_type.upper()} Warmup Training...")
        warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=10)
        
        # Advanced training model (using cosine annealing learning rate scheduler)
        print(f"Starting {activation_type.upper()} Main Training...")
        history = train_model_advanced(
            model, x_train, y_train, x_val, y_val, 
            epochs=100, 
            batch_size=128, 
            lr_schedule='cosine'
        )
        
        # Evaluate model
        results[activation_type] = model.evaluate(x_val, y_val, verbose=0)
        
        print(f"{activation_type.upper()} Results:")
        print(f"  Loss: {results[activation_type][0]:.4f}")
        print(f"  MAE: {results[activation_type][1]:.4f}")
        print(f"  MSE: {results[activation_type][2]:.4f}")
    
    # Print
    print(f"\n{'='*60}")
    print("Activation Function Performance Comparison")
    print(f"{'='*60}")
    print(f"{'Activation Function':<12} {'Loss':<10} {'MAE':<10} {'MSE':<10}")
    print("-" * 50)
    
    for activation_type in activation_types:
        loss, mae, mse = results[activation_type]
        print(f"{activation_type.upper():<12} {loss:<10.4f} {mae:<10.4f} {mse:<10.4f}")
    
    return results, activation_types

def plot_comparison(results, activation_types):
    """Plot comparison charts"""
    metrics = ['Loss', 'MAE', 'MSE']
    metric_names = ['loss', 'mae', 'mse']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results[act][i] for act in activation_types]
        axes[i].bar(activation_types, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add numerical labels
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values) * 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Compare different activation functions
    results, activation_types = compare_activations()
    
    # Plot comparison charts
    plot_comparison(results, activation_types)
    
    # Find best activation function
    best_activation = min(results.keys(), key=lambda x: results[x][0])  # Sort by loss
    print(f"\nBest Activation Function: {best_activation.upper()}")
    print(f"Best Loss: {results[best_activation][0]:.4f}")
    print(f"Best MAE: {results[best_activation][1]:.4f}")
    print(f"Best MSE: {results[best_activation