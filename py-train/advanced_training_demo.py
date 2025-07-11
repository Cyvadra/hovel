import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from train import (
    build_model, split_data, train_model_advanced, evaluate_model, 
    warmup_training, progressive_training, plot_training_history
)

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 确保使用 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

def load_data():
    """Load data"""
    with h5py.File('training_data.h5', 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)
    
    Y = tf.clip_by_value(Y, clip_value_min=-90.0, clip_value_max=90.0)
    Y = Y.numpy()
    return X, Y

def demo_different_training_strategies():
    """Demonstrate different training strategies"""
    
    # Load data
    X, Y = load_data()
    print(f"Data shape: X {X.shape}, Y {Y.shape}")
    
    # Data splitting
    x_train, x_val, y_train, y_val = split_data(X, Y)
    
    strategies = {
        'cosine_annealing': 'Cosine Annealing Learning Rate Scheduler',
        'exponential_decay': 'Exponential Decay Learning Rate Scheduler',
        'plateau_reduction': 'Plateau Learning Rate Reduction',
        'progressive_training': 'Progressive Training'
    }
    
    results = {}
    
    for strategy_name, strategy_desc in strategies.items():
        print(f"\n{'='*60}")
        print(f"Test Strategy: {strategy_desc}")
        print(f"{'='*60}")
        
        # Build model
        model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
        
        if strategy_name == 'progressive_training':
            # Progressive training
            warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=10)
            training_history = progressive_training(
                model, x_train, y_train, x_val, y_val,
                batch_sizes=[64, 128, 256],
                epochs_per_stage=50
            )
            all_history = [warmup_history] + training_history
        else:
            # Other strategies use advanced training
            warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=10)
            training_history = train_model_advanced(
                model, x_train, y_train, x_val, y_val,
                epochs=150,
                batch_size=128,
                lr_schedule=strategy_name.replace('_', '')
            )
            all_history = [warmup_history, training_history]
        
        # Evaluate model
        final_results = evaluate_model(model, x_val, y_val)
        results[strategy_name] = {
            'results': final_results,
            'history': all_history,
            'description': strategy_desc
        }
        
        # Save model
        model.save(f'model_{strategy_name}.h5')
        print(f"Model saved as 'model_{strategy_name}.h5'")
    
    return results

def compare_strategies(results):
    """Compare different training strategies"""
    print(f"\n{'='*80}")
    print("Training Strategy Performance Comparison")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'Loss':<10} {'MAE':<10} {'MSE':<10}")
    print("-" * 60)
    
    for strategy_name, data in results.items():
        loss, mae, mse = data['results']
        print(f"{data['description']:<20} {loss:<10.4f} {mae:<10.4f} {mse:<10.4f}")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda x: results[x]['results'][0])
    print(f"\nBest Training Strategy: {results[best_strategy]['description']}")
    print(f"Best Loss: {results[best_strategy]['results'][0]:.4f}")
    print(f"Best MAE: {results[best_strategy]['results'][1]:.4f}")
    print(f"Best MSE: {results[best_strategy]['results'][2]:.4f}")

def plot_strategy_comparison(results):
    """Plot strategy comparison charts"""
    strategies = list(results.keys())
    descriptions = [results[s]['description'] for s in strategies]
    losses = [results[s]['results'][0] for s in strategies]
    maes = [results[s]['results'][1] for s in strategies]
    mses = [results[s]['results'][2] for s in strategies]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Loss comparison
    axes[0].bar(descriptions, losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0].set_title('Loss Comparison')
    axes[0].set_ylabel('Loss')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[1].bar(descriptions, maes, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # MSE comparison
    axes[2].bar(descriptions, mses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[2].set_title('MSE Comparison')
    axes[2].set_ylabel('MSE')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_learning_rate_schedules():
    """Demonstrate different learning rate schedulers"""
    print(f"\n{'='*60}")
    print("Learning Rate Scheduler Demonstration")
    print(f"{'='*60}")
    
    from train import WarmUpCosineDecay, ExponentialDecayWithWarmup
    
    # Create schedulers
    cosine_scheduler = WarmUpCosineDecay(initial_lr=1e-3, warmup_epochs=10, total_epochs=100)
    exp_scheduler = ExponentialDecayWithWarmup(initial_lr=1e-3, warmup_epochs=5)
    
    # Calculate learning rate changes
    epochs = range(100)
    cosine_lrs = [cosine_scheduler(epoch) for epoch in epochs]
    exp_lrs = [exp_scheduler(epoch) for epoch in epochs]
    
    # Plot learning rate changes
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, cosine_lrs, label='Cosine Annealing', linewidth=2)
    plt.plot(epochs, exp_lrs, label='Exponential Decay', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Comparison of Different Learning Rate Schedulers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('lr_schedules_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Demonstrate learning rate schedulers
    demo_learning_rate_schedules()
    
    # Demonstrate different training strategies
    results = demo_different_training_strategies()
    
    # Compare strategy performance
    compare_strategies(results)
    
    # Plot strategy comparison charts
    plot_strategy_comparison(results)
    
    # Plot detailed training history for best strategy
    best_strategy = min(results.keys(), key=lambda x: results[x]['results'][0])
    print(f"\nPlotting detailed training history for best strategy '{results[best_strategy]['description']}'...")
    plot_training_history(results[best_strategy]['history'], f"Best Strategy: {results[best_strategy]['description']}") 