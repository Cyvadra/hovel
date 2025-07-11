# 高级模型训练系统

这个项目现在包含了多种先进的训练策略，用于提升深度学习模型的训练效果。

## 🚀 主要特性

### 1. 动态学习率调度
- **余弦退火学习率调度** (`WarmUpCosineDecay`): 包含预热阶段的余弦退火调度
- **指数衰减学习率调度** (`ExponentialDecayWithWarmup`): 包含预热阶段的指数衰减调度
- **平台学习率衰减** (`ReduceLROnPlateau`): 基于验证损失的平台衰减

### 2. 模型预热训练
- 使用较小的学习率进行初始训练
- 帮助模型在训练初期稳定收敛
- 避免训练初期的不稳定现象

### 3. 渐进式训练
- 从小批量开始，逐步增加批量大小
- 每个阶段自动调整学习率
- 提高训练稳定性和最终性能

### 4. 高级优化器
- **AdamW优化器**: 包含权重衰减的Adam优化器
- 更好的正则化效果
- 更稳定的训练过程

### 5. 可视化训练过程
- 实时绘制训练历史
- 比较不同策略的性能
- 学习率变化可视化

## 📁 文件结构

```
├── train.py                    # 主要训练模块
├── activation_comparison.py    # 激活函数比较
├── advanced_training_demo.py   # 训练策略演示
└── README_advanced_training.md # 本文档
```

## 🛠️ 使用方法

### 基本训练

```python
from train import build_model, split_data, train_model_advanced, evaluate_model

# 构建模型
model = build_model(input_shape, output_units, optimizer_type='adamw')

# 高级训练
history = train_model_advanced(
    model, x_train, y_train, x_val, y_val,
    epochs=500,
    batch_size=128,
    lr_schedule='cosine'  # 'cosine', 'exponential', 'plateau'
)
```

### 模型预热训练

```python
from train import warmup_training

# 预热训练
warmup_history = warmup_training(
    model, x_train, y_train, x_val, y_val, 
    warmup_epochs=20
)
```

### 渐进式训练

```python
from train import progressive_training

# 渐进式训练
training_history = progressive_training(
    model, x_train, y_train, x_val, y_val,
    batch_sizes=[64, 128, 256],
    epochs_per_stage=100
)
```

### 完整训练流程

```python
# 1. 模型预热
warmup_history = warmup_training(model, x_train, y_train, x_val, y_val)

# 2. 渐进式训练
training_history = progressive_training(model, x_train, y_train, x_val, y_val)

# 3. 评估模型
results = evaluate_model(model, x_val, y_val)

# 4. 可视化
plot_training_history([warmup_history] + training_history)
```

## 🎯 训练策略比较

### 1. 余弦退火学习率调度
- **优点**: 平滑的学习率衰减，避免突然的学习率变化
- **适用场景**: 大多数深度学习任务
- **参数**: `initial_lr=1e-3`, `warmup_epochs=10`, `total_epochs=500`

### 2. 指数衰减学习率调度
- **优点**: 快速的学习率衰减，适合快速收敛
- **适用场景**: 需要快速收敛的任务
- **参数**: `initial_lr=1e-3`, `warmup_epochs=5`, `decay_rate=0.95`

### 3. 平台学习率衰减
- **优点**: 基于验证性能自适应调整
- **适用场景**: 验证损失波动较大的任务
- **参数**: `factor=0.5`, `patience=10`, `min_lr=1e-7`

### 4. 渐进式训练
- **优点**: 提高训练稳定性，避免局部最优
- **适用场景**: 复杂模型或困难数据集
- **参数**: `batch_sizes=[64, 128, 256]`, `epochs_per_stage=100`

## 📊 性能监控

### 回调函数
- **ModelCheckpoint**: 自动保存最佳模型
- **EarlyStopping**: 防止过拟合
- **LearningRateScheduler**: 动态学习率调度
- **ReduceLROnPlateau**: 平台学习率衰减

### 可视化
- 训练损失曲线
- 验证损失曲线
- MAE和MSE变化
- 学习率变化曲线

## 🔧 参数调优建议

### 学习率设置
```python
# 对于不同规模的数据集
small_dataset = {'initial_lr': 1e-3, 'warmup_epochs': 5}
medium_dataset = {'initial_lr': 1e-3, 'warmup_epochs': 10}
large_dataset = {'initial_lr': 5e-4, 'warmup_epochs': 15}
```

### 批量大小设置
```python
# 渐进式训练的批量大小
batch_sizes_small = [32, 64, 128]
batch_sizes_medium = [64, 128, 256]
batch_sizes_large = [128, 256, 512]
```

### 预热轮数设置
```python
# 根据数据集大小调整预热轮数
warmup_epochs = min(20, total_epochs // 10)
```

## 🚀 运行示例

### 运行激活函数比较
```bash
python activation_comparison.py
```

### 运行训练策略演示
```bash
python advanced_training_demo.py
```

### 运行基本训练
```bash
python train.py
```

## 📈 预期改进

使用这些高级训练策略，预期可以获得以下改进：

1. **更稳定的训练过程**: 预热训练和渐进式训练减少训练初期的不稳定
2. **更好的收敛性能**: 动态学习率调度帮助模型找到更好的局部最优
3. **更高的最终精度**: AdamW优化器和高级调度策略提升模型性能
4. **更少的过拟合**: 更好的正则化和早停机制

## 🔍 故障排除

### 常见问题

1. **内存不足**: 减少批量大小或使用渐进式训练
2. **训练不稳定**: 增加预热轮数或降低初始学习率
3. **收敛缓慢**: 尝试不同的学习率调度策略
4. **过拟合**: 增加正则化强度或使用早停

### 调试技巧

1. 监控学习率变化
2. 观察训练和验证损失曲线
3. 使用不同的批量大小进行实验
4. 比较不同优化器的性能

## 📝 更新日志

- **v2.0**: 添加动态学习率调度和模型预热
- **v2.1**: 添加渐进式训练和AdamW优化器
- **v2.2**: 添加可视化功能和策略比较
- **v2.3**: 优化参数设置和文档完善 