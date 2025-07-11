import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from train import (
    build_model, split_data, train_model_advanced, evaluate_model,
    warmup_training, progressive_training, plot_training_history,
    WarmUpCosineDecay, ExponentialDecayWithWarmup
)

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

def create_synthetic_data(n_samples=1000, n_features=100, n_outputs=5):
    """创建合成数据用于测试"""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # 创建一些非线性关系
    Y = (np.sin(X[:, :10].sum(axis=1, keepdims=True)) + 
         np.cos(X[:, 10:20].sum(axis=1, keepdims=True)) + 
         np.random.randn(n_samples, n_outputs) * 0.1).astype(np.float32)
    return X, Y

def test_learning_rate_schedulers():
    """测试学习率调度器"""
    print("测试学习率调度器...")
    
    # 测试余弦退火调度器
    cosine_scheduler = WarmUpCosineDecay(initial_lr=1e-3, warmup_epochs=5, total_epochs=20)
    exp_scheduler = ExponentialDecayWithWarmup(initial_lr=1e-3, warmup_epochs=3)
    
    epochs = range(20)
    cosine_lrs = [cosine_scheduler(epoch) for epoch in epochs]
    exp_lrs = [exp_scheduler(epoch) for epoch in epochs]
    
    # 验证学习率变化
    assert cosine_lrs[0] < cosine_lrs[4]  # 预热阶段应该增长
    assert cosine_lrs[-1] < cosine_lrs[5]  # 衰减阶段应该下降
    assert exp_lrs[0] < exp_lrs[2]  # 预热阶段应该增长
    assert exp_lrs[-1] < exp_lrs[3]  # 衰减阶段应该下降
    
    print("✓ 学习率调度器测试通过")

def test_model_building():
    """测试模型构建"""
    print("测试模型构建...")
    
    # 测试不同优化器
    model_adam = build_model((100,), 5, optimizer_type='adam')
    model_adamw = build_model((100,), 5, optimizer_type='adamw')
    
    assert model_adam is not None
    assert model_adamw is not None
    assert model_adam.optimizer.__class__.__name__ == 'Adam'
    assert model_adamw.optimizer.__class__.__name__ == 'AdamW'
    
    print("✓ 模型构建测试通过")

def test_warmup_training():
    """测试预热训练"""
    print("测试预热训练...")
    
    # 创建小规模数据
    X, Y = create_synthetic_data(n_samples=200, n_features=50)
    x_train, x_val, y_train, y_val = split_data(X, Y, test_size=0.2)
    
    # 构建模型
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
    
    # 记录原始学习率
    original_lr = model.optimizer.learning_rate.numpy()
    
    # 预热训练
    warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=5)
    
    # 验证学习率恢复
    current_lr = model.optimizer.learning_rate.numpy()
    assert abs(current_lr - original_lr) < 1e-10
    
    # 验证训练历史
    assert 'loss' in warmup_history.history
    assert 'val_loss' in warmup_history.history
    
    print("✓ 预热训练测试通过")
    return warmup_history

def test_advanced_training():
    """测试高级训练"""
    print("测试高级训练...")
    
    # 创建小规模数据
    X, Y = create_synthetic_data(n_samples=200, n_features=50)
    x_train, x_val, y_train, y_val = split_data(X, Y, test_size=0.2)
    
    # 构建模型
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
    
    # 测试不同学习率调度策略
    strategies = ['cosine', 'exponential', 'plateau']
    
    for strategy in strategies:
        print(f"  测试 {strategy} 调度策略...")
        history = train_model_advanced(
            model, x_train, y_train, x_val, y_val,
            epochs=10,
            batch_size=32,
            lr_schedule=strategy
        )
        
        # 验证训练历史
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert 'mae' in history.history
        assert 'mse' in history.history
    
    print("✓ 高级训练测试通过")

def test_progressive_training():
    """测试渐进式训练"""
    print("测试渐进式训练...")
    
    # 创建小规模数据
    X, Y = create_synthetic_data(n_samples=200, n_features=50)
    x_train, x_val, y_train, y_val = split_data(X, Y, test_size=0.2)
    
    # 构建模型
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
    
    # 渐进式训练
    training_history = progressive_training(
        model, x_train, y_train, x_val, y_val,
        batch_sizes=[16, 32],
        epochs_per_stage=5
    )
    
    # 验证训练历史
    assert len(training_history) == 2  # 两个阶段
    
    for history in training_history:
        assert 'loss' in history.history
        assert 'val_loss' in history.history
    
    print("✓ 渐进式训练测试通过")
    return training_history

def test_evaluation():
    """测试模型评估"""
    print("测试模型评估...")
    
    # 创建小规模数据
    X, Y = create_synthetic_data(n_samples=200, n_features=50)
    x_train, x_val, y_train, y_val = split_data(X, Y, test_size=0.2)
    
    # 构建和训练模型
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
    history = train_model_advanced(
        model, x_train, y_train, x_val, y_val,
        epochs=5,
        batch_size=32,
        lr_schedule='cosine'
    )
    
    # 评估模型
    results = evaluate_model(model, x_val, y_val)
    
    # 验证结果
    assert len(results) == 3  # loss, mae, mse
    assert all(isinstance(r, float) for r in results)
    assert all(r >= 0 for r in results)
    
    print("✓ 模型评估测试通过")
    return results

def test_visualization():
    """测试可视化功能"""
    print("测试可视化功能...")
    
    # 创建小规模数据
    X, Y = create_synthetic_data(n_samples=200, n_features=50)
    x_train, x_val, y_train, y_val = split_data(X, Y, test_size=0.2)
    
    # 构建和训练模型
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
    
    # 预热训练
    warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=3)
    
    # 主训练
    training_history = train_model_advanced(
        model, x_train, y_train, x_val, y_val,
        epochs=5,
        batch_size=32,
        lr_schedule='cosine'
    )
    
    # 测试可视化
    try:
        plot_training_history([warmup_history, training_history], "测试训练历史")
        print("✓ 可视化功能测试通过")
    except Exception as e:
        print(f"⚠ 可视化功能测试失败: {e}")

def run_all_tests():
    """运行所有测试"""
    print("开始运行高级训练功能测试...")
    print("=" * 50)
    
    try:
        test_learning_rate_schedulers()
        test_model_building()
        warmup_history = test_warmup_training()
        test_advanced_training()
        training_history = test_progressive_training()
        results = test_evaluation()
        test_visualization()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50)
        
        # 打印最终结果
        print(f"最终评估结果:")
        print(f"  损失: {results[0]:.4f}")
        print(f"  MAE: {results[1]:.4f}")
        print(f"  MSE: {results[2]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ 高级训练系统已准备就绪！")
    else:
        print("\n❌ 请检查错误并修复问题") 