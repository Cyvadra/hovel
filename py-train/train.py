import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.losses import Huber
from tensorflow.keras.activations import relu, elu, swish, gelu
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

# 设置随机种子保证可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 1. 数据准备和切分
def split_data(x, y, test_size=0.1, random_state=42):
    """
    随机切分数据集为训练集和验证集
    参数:
        x: 特征数据 (17108, 366)
        y: 标签数据 (17108, 5)
        test_size: 验证集比例 (默认10%)
        random_state: 随机种子
    返回:
        X_train, x_val, y_train, y_val
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"训练集形状: x_train {x_train.shape}, y_train {y_train.shape}")
    print(f"验证集形状: x_val {x_val.shape}, y_val {y_val.shape}")
    return x_train, x_val, y_train, y_val

# 2. 动态学习率调度器
class WarmUpCosineDecay:
    """余弦退火学习率调度器，包含预热阶段"""
    
    def __init__(self, initial_lr=1e-4, warmup_epochs=10, total_epochs=500, min_lr=1e-7):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
    
    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return lr

class ExponentialDecayWithWarmup:
    """指数衰减学习率调度器，包含预热阶段"""
    
    def __init__(self, initial_lr=1e-3, warmup_epochs=5, decay_rate=0.95, min_lr=1e-7):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.min_lr = min_lr
    
    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 指数衰减阶段
            lr = max(self.initial_lr * (self.decay_rate ** (epoch - self.warmup_epochs)), self.min_lr)
        
        return lr

# 3. 模型构建
def build_model(input_shape, output_units, activation_type='mixed', optimizer_type='adamw'):
    """
    构建深层神经网络模型
    参数:
        input_shape: 输入特征维度
        output_units: 输出单元数
        activation_type: 激活函数类型 ('relu', 'elu', 'swish', 'gelu', 'softmax', 'mixed')
        optimizer_type: 优化器类型 ('adam', 'adamw')
    返回:
        编译好的模型
    """
    
    if activation_type == 'relu':
        activations = ['relu'] * 5
    elif activation_type == 'elu':
        activations = ['elu'] * 5
    elif activation_type == 'swish':
        activations = ['swish'] * 5
    elif activation_type == 'gelu':
        activations = ['gelu'] * 5
    elif activation_type == 'softmax':
        activations = ['softmax'] * 5
    elif activation_type == 'mixed':
        # 混合激活函数：不同层使用不同的激活函数
        activations = ['swish', 'elu', 'gelu', 'relu', 'softmax']
    else:
        activations = ['relu'] * 5
    
    # 构建模型
    model = Sequential([
        Dense(256, activation=activations[0], input_shape=input_shape,
            kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation=activations[1], kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation=activations[2], kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation=activations[3], kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),

        Dense(output_units, activation=activations[4], kernel_regularizer=l2(0.01)),
    ])
    
    # 选择优化器
    if optimizer_type == 'adamw':
        optimizer = AdamW(learning_rate=1e-3, weight_decay=0.01)
    else:
        optimizer = Adam(learning_rate=1e-3)
    
    # 使用Huber损失
    huber_loss = Huber(delta=9.5)
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=huber_loss,
        metrics=['mae', 'mse']
    )
    return model

# 4. 高级训练过程
def train_model_advanced(model, x_train, y_train, x_val, y_val, 
                        epochs=500, batch_size=128, lr_schedule='cosine'):
    """
    高级训练模型，包含动态学习率调度和预热
    参数:
        model: 编译好的模型
        x_train, y_train: 训练数据
        x_val, y_val: 验证数据
        epochs: 训练轮次
        batch_size: 批量大小
        lr_schedule: 学习率调度类型 ('cosine', 'exponential', 'plateau')
    返回:
        history: 训练历史对象
    """
    
    # 设置学习率调度器
    if lr_schedule == 'cosine':
        lr_scheduler = WarmUpCosineDecay(initial_lr=1e-3, warmup_epochs=10, total_epochs=epochs)
        lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)
    elif lr_schedule == 'exponential':
        lr_scheduler = ExponentialDecayWithWarmup(initial_lr=1e-3, warmup_epochs=5)
        lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)
    else:
        lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    
    # 回调函数设置
    callbacks = [
        lr_callback,
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ) if lr_schedule != 'plateau' else None
    ]
    
    # 移除None值
    callbacks = [cb for cb in callbacks if cb is not None]
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    return history

# 5. 渐进式训练
def progressive_training(model, x_train, y_train, x_val, y_val, 
                        batch_sizes=[64, 128, 256], epochs_per_stage=100):
    """
    渐进式训练：从小批量开始，逐步增加批量大小
    """
    history_list = []
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n{'='*50}")
        print(f"训练阶段 {i+1}/{len(batch_sizes)}: 批量大小 = {batch_size}")
        print(f"{'='*50}")
        
        # 调整学习率
        current_lr = model.optimizer.learning_rate.numpy()
        new_lr = current_lr * (0.8 ** i)  # 每个阶段降低学习率
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        
        # 训练当前阶段
        history = train_model_advanced(
            model, x_train, y_train, x_val, y_val,
            epochs=epochs_per_stage,
            batch_size=batch_size,
            lr_schedule='plateau'
        )
        history_list.append(history)
    
    return history_list

# 6. 评估函数
def evaluate_model(model, x_val, y_val):
    """
    评估模型性能
    参数:
        model: 训练好的模型
        x_val, y_val: 验证数据
    """
    print("\n模型评估:")
    results = model.evaluate(x_val, y_val, verbose=0)
    print(f"验证集损失(Huber): {results[0]:.4f}")
    print(f"验证集MAE: {results[1]:.4f}")
    print(f"验证集MSE: {results[2]:.4f}")
    return results

# 7. 可视化训练过程
def plot_training_history(history_list, title="训练历史"):
    """绘制训练历史图表"""
    if not isinstance(history_list, list):
        history_list = [history_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, history in enumerate(history_list):
        # 损失曲线
        axes[0, 0].plot(history.history['loss'], label=f'训练损失 (阶段{i+1})', alpha=0.7)
        axes[0, 0].plot(history.history['val_loss'], label=f'验证损失 (阶段{i+1})', alpha=0.7)
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE曲线
        axes[0, 1].plot(history.history['mae'], label=f'训练MAE (阶段{i+1})', alpha=0.7)
        axes[0, 1].plot(history.history['val_mae'], label=f'验证MAE (阶段{i+1})', alpha=0.7)
        axes[0, 1].set_title('MAE曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE曲线
        axes[1, 0].plot(history.history['mse'], label=f'训练MSE (阶段{i+1})', alpha=0.7)
        axes[1, 0].plot(history.history['val_mse'], label=f'验证MSE (阶段{i+1})', alpha=0.7)
        axes[1, 0].set_title('MSE曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线（如果有的话）
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label=f'学习率 (阶段{i+1})', alpha=0.7)
            axes[1, 1].set_title('学习率变化')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('学习率')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# 8. 模型预热训练
def warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=20):
    """
    模型预热训练：使用较小的学习率进行初始训练
    """
    print(f"\n{'='*50}")
    print("开始模型预热训练")
    print(f"{'='*50}")
    
    # 保存原始学习率
    original_lr = model.optimizer.learning_rate.numpy()
    
    # 设置预热学习率（较小的学习率）
    warmup_lr = original_lr * 0.1
    tf.keras.backend.set_value(model.optimizer.learning_rate, warmup_lr)
    
    # 预热训练
    warmup_history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=warmup_epochs,
        batch_size=64,
        verbose=1,
        shuffle=True
    )
    
    # 恢复原始学习率
    tf.keras.backend.set_value(model.optimizer.learning_rate, original_lr)
    
    print(f"预热训练完成，恢复学习率: {original_lr}")
    return warmup_history

# 主程序
if __name__ == "__main__":
    # 确保使用 GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"使用GPU: {physical_devices[0]}")
    
    # 加载数据
    import h5py
    with h5py.File('training_data.h5', 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)

    Y = tf.clip_by_value(Y, clip_value_min=-90.0, clip_value_max=90.0)
    Y = Y.numpy()

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    # 1. 数据切分
    x_train, x_val, y_train, y_val = split_data(X, Y)
    
    # 2. 构建模型（使用AdamW优化器）
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw')
    print("\n模型架构摘要:")
    model.summary()
    
    # 3. 模型预热训练
    warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=15)
    
    # 4. 渐进式训练
    print("\n开始渐进式训练...")
    training_history = progressive_training(
        model, x_train, y_train, x_val, y_val,
        batch_sizes=[64, 128, 256],
        epochs_per_stage=80
    )
    
    # 5. 最终评估
    final_results = evaluate_model(model, x_val, y_val)
    
    # 6. 可视化训练过程
    all_history = [warmup_history] + training_history
    plot_training_history(all_history, "完整训练历史")
    
    # 7. 保存最终模型
    model.save('final_advanced_model.h5')
    print("\n模型已保存为 'final_advanced_model.h5'")
    
    # 8. 打印最终结果
    print(f"\n{'='*60}")
    print("最终训练结果")
    print(f"{'='*60}")
    print(f"最终损失: {final_results[0]:.4f}")
    print(f"最终MAE: {final_results[1]:.4f}")
    print(f"最终MSE: {final_results[2]:.4f}")