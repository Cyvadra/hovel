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
    print(f"使用GPU: {physical_devices[0]}")

def load_data():
    """加载数据"""
    with h5py.File('training_data.h5', 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)
    
    Y = tf.clip_by_value(Y, clip_value_min=-90.0, clip_value_max=90.0)
    return X, Y

def demo_different_training_strategies():
    """演示不同的训练策略"""
    
    # 加载数据
    X, Y = load_data()
    print(f"数据形状: X {X.shape}, Y {Y.shape}")
    
    # 数据切分
    x_train, x_val, y_train, y_val = split_data(X, Y)
    
    strategies = {
        'cosine_annealing': '余弦退火学习率调度',
        'exponential_decay': '指数衰减学习率调度',
        'plateau_reduction': '平台学习率衰减',
        'progressive_training': '渐进式训练'
    }
    
    results = {}
    
    for strategy_name, strategy_desc in strategies.items():
        print(f"\n{'='*60}")
        print(f"测试策略: {strategy_desc}")
        print(f"{'='*60}")
        
        # 构建模型
        model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
        
        if strategy_name == 'progressive_training':
            # 渐进式训练
            warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=10)
            training_history = progressive_training(
                model, x_train, y_train, x_val, y_val,
                batch_sizes=[64, 128, 256],
                epochs_per_stage=50
            )
            all_history = [warmup_history] + training_history
        else:
            # 其他策略使用高级训练
            warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=10)
            training_history = train_model_advanced(
                model, x_train, y_train, x_val, y_val,
                epochs=150,
                batch_size=128,
                lr_schedule=strategy_name.replace('_', '')
            )
            all_history = [warmup_history, training_history]
        
        # 评估模型
        final_results = evaluate_model(model, x_val, y_val)
        results[strategy_name] = {
            'results': final_results,
            'history': all_history,
            'description': strategy_desc
        }
        
        # 保存模型
        model.save(f'model_{strategy_name}.h5')
        print(f"模型已保存为 'model_{strategy_name}.h5'")
    
    return results

def compare_strategies(results):
    """比较不同训练策略的性能"""
    print(f"\n{'='*80}")
    print("训练策略性能比较")
    print(f"{'='*80}")
    print(f"{'策略':<20} {'损失':<10} {'MAE':<10} {'MSE':<10}")
    print("-" * 60)
    
    for strategy_name, data in results.items():
        loss, mae, mse = data['results']
        print(f"{data['description']:<20} {loss:<10.4f} {mae:<10.4f} {mse:<10.4f}")
    
    # 找出最佳策略
    best_strategy = min(results.keys(), key=lambda x: results[x]['results'][0])
    print(f"\n最佳训练策略: {results[best_strategy]['description']}")
    print(f"最佳损失: {results[best_strategy]['results'][0]:.4f}")
    print(f"最佳MAE: {results[best_strategy]['results'][1]:.4f}")
    print(f"最佳MSE: {results[best_strategy]['results'][2]:.4f}")

def plot_strategy_comparison(results):
    """绘制策略比较图表"""
    strategies = list(results.keys())
    descriptions = [results[s]['description'] for s in strategies]
    losses = [results[s]['results'][0] for s in strategies]
    maes = [results[s]['results'][1] for s in strategies]
    mses = [results[s]['results'][2] for s in strategies]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 损失比较
    axes[0].bar(descriptions, losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0].set_title('损失比较')
    axes[0].set_ylabel('损失')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # MAE比较
    axes[1].bar(descriptions, maes, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1].set_title('MAE比较')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # MSE比较
    axes[2].bar(descriptions, mses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[2].set_title('MSE比较')
    axes[2].set_ylabel('MSE')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_learning_rate_schedules():
    """演示不同学习率调度器的效果"""
    print(f"\n{'='*60}")
    print("学习率调度器演示")
    print(f"{'='*60}")
    
    from train import WarmUpCosineDecay, ExponentialDecayWithWarmup
    
    # 创建调度器
    cosine_scheduler = WarmUpCosineDecay(initial_lr=1e-3, warmup_epochs=10, total_epochs=100)
    exp_scheduler = ExponentialDecayWithWarmup(initial_lr=1e-3, warmup_epochs=5)
    
    # 计算学习率变化
    epochs = range(100)
    cosine_lrs = [cosine_scheduler(epoch) for epoch in epochs]
    exp_lrs = [exp_scheduler(epoch) for epoch in epochs]
    
    # 绘制学习率变化
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, cosine_lrs, label='余弦退火', linewidth=2)
    plt.plot(epochs, exp_lrs, label='指数衰减', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.title('不同学习率调度器的比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('lr_schedules_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 演示学习率调度器
    demo_learning_rate_schedules()
    
    # 演示不同训练策略
    results = demo_different_training_strategies()
    
    # 比较策略性能
    compare_strategies(results)
    
    # 绘制策略比较图表
    plot_strategy_comparison(results)
    
    # 为最佳策略绘制详细训练历史
    best_strategy = min(results.keys(), key=lambda x: results[x]['results'][0])
    print(f"\n绘制最佳策略 '{results[best_strategy]['description']}' 的详细训练历史...")
    plot_training_history(results[best_strategy]['history'], f"最佳策略: {results[best_strategy]['description']}") 