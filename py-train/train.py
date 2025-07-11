import numpy as np
import tensorflow as tf
import os
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

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision training for better GPU performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# GPU Configuration function
def configure_gpus():
    """
    Configure multiple GPUs for distributed training
    Returns:
        strategy: TensorFlow distribution strategy
        num_gpus: Number of available GPUs
    """
    try:
        # Try to configure GPU memory growth first
        physical_devices = tf.config.list_physical_devices('GPU')
        num_gpus = len(physical_devices)
        
        if num_gpus == 0:
            print("No GPUs found. Using CPU.")
            return tf.distribute.OneDeviceStrategy("/cpu:0"), 0
        elif num_gpus == 1:
            print(f"Using single GPU: {physical_devices[0]}")
            # Enable memory growth for single GPU
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                print("Warning: Could not set memory growth for GPU")
            return tf.distribute.OneDeviceStrategy("/gpu:0"), 1
        else:
            print(f"Found {num_gpus} GPUs. Configuring for distributed training.")
            # Enable memory growth for all GPUs
            for gpu in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    print(f"Warning: Could not set memory growth for {gpu}")
            
            # Use MirroredStrategy for synchronous distributed training
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {strategy.num_replicas_in_sync} GPUs for distributed training")
            return strategy, num_gpus
    except Exception as e:
        print(f"GPU configuration failed: {e}")
        print("Falling back to CPU training")
        return tf.distribute.OneDeviceStrategy("/cpu:0"), 0

def display_gpu_info():
    """Display detailed GPU information"""
    print("\n" + "="*60)
    print("GPU Configuration Information")
    print("="*60)
    
    # List all physical devices
    physical_devices = tf.config.list_physical_devices()
    print(f"Total physical devices: {len(physical_devices)}")
    
    # GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"GPU devices: {len(gpu_devices)}")
    for i, gpu in enumerate(gpu_devices):
        print(f"  GPU {i}: {gpu}")
    
    # CPU devices
    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"CPU devices: {len(cpu_devices)}")
    for i, cpu in enumerate(cpu_devices):
        print(f"  CPU {i}: {cpu}")
    
    # TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if GPU is available
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print("="*60)

def optimize_batch_sizes(num_gpus, base_batch_sizes):
    """
    Optimize batch sizes for multi-GPU training
    Parameters:
        num_gpus: Number of available GPUs
        base_batch_sizes: List of base batch sizes for single GPU
    Returns:
        optimized_batch_sizes: List of optimized batch sizes for multi-GPU
    """
    if num_gpus <= 1:
        return base_batch_sizes
    
    # Scale batch sizes by number of GPUs
    scaled_batch_sizes = [bs * num_gpus for bs in base_batch_sizes]
    
    # Apply additional optimizations for multi-GPU
    # For better memory utilization, we can increase batch sizes further
    # but need to be careful about memory limits
    optimized_batch_sizes = []
    for bs in scaled_batch_sizes:
        # For multi-GPU, we can often use larger batch sizes
        # but cap at reasonable limits to avoid memory issues
        if num_gpus >= 4:
            optimized_bs = min(bs * 1.5, 512)  # Cap at 512 for very large setups
        elif num_gpus >= 2:
            optimized_bs = min(bs * 1.2, 256)  # Cap at 256 for medium setups
        else:
            optimized_bs = bs
        
        optimized_batch_sizes.append(int(optimized_bs))
    
    return optimized_batch_sizes

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    try:
        # Get GPU memory info
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print("\nGPU Memory Information:")
            for i, gpu in enumerate(gpus):
                try:
                    # Get memory info for each GPU
                    memory_info = tf.config.experimental.get_memory_info(gpu.name)
                    print(f"  GPU {i}: {gpu.name}")
                    print(f"    Current memory: {memory_info['current'] / 1024**3:.2f} GB")
                    print(f"    Peak memory: {memory_info['peak'] / 1024**3:.2f} GB")
                except:
                    print(f"  GPU {i}: {gpu.name} - Memory info not available")
    except:
        print("GPU memory monitoring not available")

# 1. Data preparation and splitting
def split_data(x, y, test_size=0.1, random_state=42):
    """
    Randomly split dataset into training and validation sets
    Parameters:
        x: feature data (17108, 366)
        y: label data (17108, 5)
        test_size: validation set ratio (default 10%)
        random_state: random seed
    Returns:
        X_train, x_val, y_train, y_val
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"Training set shape: x_train {x_train.shape}, y_train {y_train.shape}")
    print(f"Validation set shape: x_val {x_val.shape}, y_val {y_val.shape}")
    return x_train, x_val, y_train, y_val

# 2. Dynamic learning rate scheduler
class WarmUpCosineDecay:
    """Cosine annealing learning rate scheduler with warmup phase"""
    
    def __init__(self, initial_lr=1e-4, warmup_epochs=10, total_epochs=500, min_lr=1e-7):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
    
    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup phase: linear growth
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return lr

class ExponentialDecayWithWarmup:
    """Exponential decay learning rate scheduler with warmup phase"""
    
    def __init__(self, initial_lr=1e-3, warmup_epochs=5, decay_rate=0.95, min_lr=1e-7):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.min_lr = min_lr
    
    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup phase: linear growth
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Exponential decay phase
            lr = max(self.initial_lr * (self.decay_rate ** (epoch - self.warmup_epochs)), self.min_lr)
        
        return lr

# 3. Model building
def build_model(input_shape, output_units, activation_type='mixed', optimizer_type='adamw', strategy=None):
    """
    Build deep neural network model
    Parameters:
        input_shape: input feature dimensions
        output_units: number of output units
        activation_type: activation function type ('relu', 'elu', 'swish', 'gelu', 'softmax', 'mixed')
        optimizer_type: optimizer type ('adam', 'adamw')
        strategy: TensorFlow distribution strategy for multi-GPU training
    Returns:
        compiled model
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
        # Mixed activation functions: different layers use different activation functions
        activations = ['swish', 'elu', 'gelu', 'relu', 'softmax']
    else:
        activations = ['relu'] * 5
    activations[4] = 'softmax'
    
    # Build model within distribution strategy scope
    if strategy is not None:
        with strategy.scope():
            model = Sequential([
                Dense(64, activation=activations[0], input_shape=input_shape,
                    kernel_regularizer=l2(0.01), dtype='float16'),
                BatchNormalization(dtype='float16'),
                Dropout(0.3),
                
                Dense(24, activation=activations[1], kernel_regularizer=l2(0.01), dtype='float16'),
                BatchNormalization(dtype='float16'),
                Dropout(0.3),
                
                Dense(12, activation=activations[2], kernel_regularizer=l2(0.01), dtype='float16'),
                BatchNormalization(dtype='float16'),
                Dropout(0.2),
                
                Dense(8, activation=activations[3], kernel_regularizer=l2(0.01), dtype='float16'),
                BatchNormalization(dtype='float16'),
                Dropout(0.1),

                Dense(output_units, activation=activations[4], kernel_regularizer=l2(0.01), dtype='float16'),
            ])
            
            # Choose optimizer
            if optimizer_type == 'adamw':
                optimizer = AdamW(learning_rate=1e-3, weight_decay=0.01)
            else:
                optimizer = Adam(learning_rate=1e-3)
            
            # Use Huber loss
            huber_loss = Huber(delta=9.5)
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=huber_loss,
                metrics=['mae', 'mse']
            )
    else:
        # Build model without distribution strategy
        model = Sequential([
            Dense(64, activation=activations[0], input_shape=input_shape,
                kernel_regularizer=l2(0.01), dtype='float16'),
            BatchNormalization(dtype='float16'),
            Dropout(0.3),
            
            Dense(24, activation=activations[1], kernel_regularizer=l2(0.01), dtype='float16'),
            BatchNormalization(dtype='float16'),
            Dropout(0.3),
            
            Dense(12, activation=activations[2], kernel_regularizer=l2(0.01), dtype='float16'),
            BatchNormalization(dtype='float16'),
            Dropout(0.2),
            
            Dense(8, activation=activations[3], kernel_regularizer=l2(0.01), dtype='float16'),
            BatchNormalization(dtype='float16'),
            Dropout(0.1),

            Dense(output_units, activation=activations[4], kernel_regularizer=l2(0.01), dtype='float16'),
        ])
        
        # Choose optimizer
        if optimizer_type == 'adamw':
            optimizer = AdamW(learning_rate=1e-3, weight_decay=0.01)
        else:
            optimizer = Adam(learning_rate=1e-3)
        
        # Use Huber loss
        huber_loss = Huber(delta=9.5)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=huber_loss,
            metrics=['mae', 'mse']
        )
    
    return model

# 4. Advanced training process
def train_model_advanced(model, x_train, y_train, x_val, y_val, 
                        epochs=500, batch_size=128, lr_schedule='cosine'):
    """
    Advanced model training with dynamic learning rate scheduling and warmup
    Parameters:
        model: compiled model
        x_train, y_train: training data
        x_val, y_val: validation data
        epochs: number of training epochs
        batch_size: batch size
        lr_schedule: learning rate schedule type ('cosine', 'exponential', 'plateau')
    Returns:
        history: training history object
    """
    
    # Set learning rate scheduler
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
    
    # Callback settings
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
            patience=15,
            min_lr=1e-7,
            verbose=1
        ) if lr_schedule != 'plateau' else None
    ]
    
    # Remove None values
    callbacks = [cb for cb in callbacks if cb is not None]
    
    # Train model
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

# 5. Progressive training
def progressive_training(model, x_train, y_train, x_val, y_val, 
                        batch_sizes=[64, 128, 256], epochs_per_stage=100):
    """
    Progressive training: start with small batch size and gradually increase
    """
    history_list = []
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n{'='*50}")
        print(f"Training Stage {i+1}/{len(batch_sizes)}: Batch Size = {batch_size}")
        print(f"{'='*50}")
        
        # Adjust learning rate
        current_lr = model.optimizer.learning_rate.numpy()
        new_lr = current_lr * (0.8 ** i)  # Reduce learning rate each stage
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        
        # Train current stage
        history = train_model_advanced(
            model, x_train, y_train, x_val, y_val,
            epochs=epochs_per_stage,
            batch_size=batch_size,
            lr_schedule='plateau'
        )
        history_list.append(history)
    
    return history_list

# 6. Evaluation function
def evaluate_model(model, x_val, y_val):
    """
    Evaluate model performance
    Parameters:
        model: trained model
        x_val, y_val: validation data
    """
    print("\nModel Evaluation:")
    results = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation Loss (Huber): {results[0]:.4f}")
    print(f"Validation MAE: {results[1]:.4f}")
    print(f"Validation MSE: {results[2]:.4f}")
    return results

# 7. Visualize training process
def plot_training_history(history_list, title="Training History"):
    """Plot training history charts"""
    if not isinstance(history_list, list):
        history_list = [history_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, history in enumerate(history_list):
        # Loss curve
        axes[0, 0].plot(history.history['loss'], label=f'Training Loss (Stage {i+1})', alpha=0.7)
        axes[0, 0].plot(history.history['val_loss'], label=f'Validation Loss (Stage {i+1})', alpha=0.7)
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curve
        axes[0, 1].plot(history.history['mae'], label=f'Training MAE (Stage {i+1})', alpha=0.7)
        axes[0, 1].plot(history.history['val_mae'], label=f'Validation MAE (Stage {i+1})', alpha=0.7)
        axes[0, 1].set_title('MAE Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE curve
        axes[1, 0].plot(history.history['mse'], label=f'Training MSE (Stage {i+1})', alpha=0.7)
        axes[1, 0].plot(history.history['val_mse'], label=f'Validation MSE (Stage {i+1})', alpha=0.7)
        axes[1, 0].set_title('MSE Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label=f'Learning Rate (Stage {i+1})', alpha=0.7)
            axes[1, 1].set_title('Learning Rate Change')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# 8. Model warmup training
def warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=20, batch_size=64):
    """
    Model warmup training: use smaller learning rate for initial training
    """
    print(f"\n{'='*50}")
    print("Starting Model Warmup Training")
    print(f"{'='*50}")
    
    # Save original learning rate
    original_lr = model.optimizer.learning_rate.numpy()
    
    # Set warmup learning rate (smaller learning rate)
    warmup_lr = original_lr * 0.1
    tf.keras.backend.set_value(model.optimizer.learning_rate, warmup_lr)
    
    # Warmup training
    warmup_history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=warmup_epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )
    
    # Restore original learning rate
    tf.keras.backend.set_value(model.optimizer.learning_rate, original_lr)
    
    print(f"Warmup training completed, restored learning rate: {original_lr}")
    return warmup_history

# Main program
if __name__ == "__main__":
    # Check if CPU training is forced
    force_cpu = os.environ.get('FORCE_CPU', 'false').lower() == 'true'
    
    if force_cpu:
        print("CPU training forced by environment variable")
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        num_gpus = 0
    else:
        # Configure GPUs with fallback to CPU if needed
        try:
            strategy, num_gpus = configure_gpus()
        except Exception as e:
            print(f"GPU configuration failed: {e}")
            print("Forcing CPU training")
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            num_gpus = 0
    
    # Display GPU information
    display_gpu_info()
    
    # Monitor GPU memory
    monitor_gpu_memory()
    
    # Load data with float16 precision
    import h5py
    with h5py.File('training_data.h5', 'r') as f:
        X = np.array(f['X'][:], dtype=np.float16)
        Y = np.array(f['Y'][:], dtype=np.float16)

    # Auto transpose data to have samples as first dimension
    # X: (366, 17108) -> (17108, 366)
    # Y: (5, 17108) -> (17108, 5)
    if X.shape[0] != Y.shape[0]:
        X = X.T
        Y = Y.T

    # Convert to float16 for GPU performance
    Y = tf.cast(Y, tf.float16)
    Y = tf.clip_by_value(Y, clip_value_min=-90.0, clip_value_max=90.0)
    Y = Y.numpy()

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
    print(f"Mixed precision enabled: {tf.keras.mixed_precision.global_policy().name}")
    
    # 1. Data splitting
    x_train, x_val, y_train, y_val = split_data(X, Y)
    
    # 2. Build model (using AdamW optimizer with distribution strategy)
    model = build_model((x_train.shape[1],), y_train.shape[1], optimizer_type='adamw', strategy=strategy)
    print("\nModel Architecture Summary:")
    model.summary()
    
    # 3. Model warmup training
    # Optimize warmup batch size for multi-GPU training
    warmup_batch_size = optimize_batch_sizes(num_gpus, [64])[0]
    print(f"Optimized warmup batch size: {warmup_batch_size}")
    warmup_history = warmup_training(model, x_train, y_train, x_val, y_val, warmup_epochs=20, batch_size=warmup_batch_size)
    
    # 4. Progressive training
    print("\nStarting progressive training...")
    # Optimize batch sizes for multi-GPU training
    base_batch_sizes = [64, 128, 32]
    batch_sizes = optimize_batch_sizes(num_gpus, base_batch_sizes)
    print(f"Optimized batch sizes for {num_gpus} GPU(s): {batch_sizes}")
    
    training_history = progressive_training(
        model, x_train, y_train, x_val, y_val,
        batch_sizes=batch_sizes,
        epochs_per_stage=300
    )
    
    # 5. Final evaluation
    final_results = evaluate_model(model, x_val, y_val)
    
    # 6. Visualize training process
    all_history = [warmup_history] + training_history
    plot_training_history(all_history, "Complete Training History")
    
    # 7. Save final model
    model.save('final_advanced_model.h5')
    print("\nModel saved as 'final_advanced_model.h5'")
    
    # 8. Print final results
    print(f"\n{'='*60}")
    print("Final Training Results")
    print(f"{'='*60}")
    print(f"Training completed using {num_gpus} GPU(s)")
    print(f"Distribution strategy: {type(strategy).__name__}")
    print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")
    print(f"Final Loss: {final_results[0]:.4f}")
    print(f"Final MAE: {final_results[1]:.4f}")
    print(f"Final MSE: {final_results[2]:.4f}")
    
    # Final GPU memory check
    print(f"\n{'='*60}")
    print("Final GPU Memory Status")
    print(f"{'='*60}")
    monitor_gpu_memory()