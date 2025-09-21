import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import re
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CyclicLR
from torch.nn import functional as F
import warnings
import logging
import time
from collections import defaultdict
from error_handling import (
    TrainingError, DataError, ModelError, ConfigurationError,
    validate_tensor, validate_array, validate_positive,
    log_execution_time, handle_exception, check_gpu_memory
)
from config_validator import validate_training_config
warnings.filterwarnings('ignore')

# --- Checkpoint Management ---
def find_latest_checkpoint(model_name="optimized_model"):
    """
    Find the latest checkpoint file for the given model name.
    
    Args:
        model_name (str): Base name of the model
        
    Returns:
        tuple: (checkpoint_path, epoch_number) or (None, 0) if no checkpoint found
    """
    # Pattern to match checkpoint files: model_name_checkpoint_epoch_X.pth
    pattern = re.compile(rf'{model_name}_checkpoint_epoch_(\d+)\.pth')
    
    latest_checkpoint = None
    latest_epoch = 0
    
    # Search in current directory
    for filename in os.listdir('.'):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = filename
    
    if latest_checkpoint:
        print(f"Found latest checkpoint: {latest_checkpoint} (epoch {latest_epoch})")
        return latest_checkpoint, latest_epoch
    else:
        print("No checkpoint files found.")
        return None, 0

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load model and training state from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        device: Device to load tensors on
        
    Returns:
        dict: Training state including epoch, best_val_loss, patience_counter, etc.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        # New format with training state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        training_state = {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'patience_counter': checkpoint.get('patience_counter', 0),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', [])
        }
        
        print(f"Loaded training state from epoch {training_state['epoch']}")
        print(f"Best validation loss: {training_state['best_val_loss']:.6f}")
        
    else:
        # Old format - just model state dict
        model.load_state_dict(checkpoint)
        training_state = {
            'epoch': 0,
            'best_val_loss': float('inf'),
            'patience_counter': 0,
            'train_losses': [],
            'val_losses': []
        }
        print("Loaded model state only (old checkpoint format)")
    
    return training_state

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                   patience_counter, train_losses, val_losses,
                   model_name="optimized_model", save_after_epoch=0):
    """
    Save a complete checkpoint with model and training state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The scheduler to save
        epoch (int): Current epoch
        best_val_loss (float): Best validation loss so far
        patience_counter (int): Current patience counter
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        model_name (str): Base name for the checkpoint file
        save_after_epoch (int): Only save checkpoint after this epoch number
    """
    # Skip saving if we haven't reached save_after_epoch yet
    if epoch <= save_after_epoch:
        print(f"Skipping checkpoint save at epoch {epoch} (will start saving after epoch {save_after_epoch})")
        return
    # Get model parameters
    if isinstance(model, nn.DataParallel):
        model_instance = model.module
    else:
        model_instance = model

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'train_losses': train_losses,
        'val_losses': val_losses,
        # Add model parameters
        'model_params': {
            'input_dim': model_instance.input_dim,
            'output_dim': model_instance.output_dim,
            'hidden_size': model_instance.hidden_size,
            'num_layers': len(model_instance.layers)
        }
    }
    
    checkpoint_path = f'{model_name}_checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# --- Logging Setup ---
def setup_logging(model_name="optimized_model"):
    """Setup logging configuration."""
    logger = logging.getLogger(f"training_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(f'{model_name}_training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TrainingMetrics:
    """Track and log training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def update(self, epoch, train_loss, val_loss, lr, **kwargs):
        """Update metrics for current epoch."""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rate'].append(lr)
        
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_best_epoch(self):
        """Get epoch with best validation loss."""
        if not self.metrics['val_loss']:
            return -1
        return np.argmin(self.metrics['val_loss'])
    
    def get_training_time(self):
        """Get total training time."""
        return time.time() - self.start_time
    
    def log_summary(self, logger):
        """Log training summary."""
        if not self.metrics['val_loss']:
            return
        
        best_epoch = self.get_best_epoch()
        best_val_loss = min(self.metrics['val_loss'])
        total_time = self.get_training_time()
        
        logger.info(f"Training Summary:")
        logger.info(f"  Best epoch: {best_epoch + 1}")
        logger.info(f"  Best validation loss: {best_val_loss:.6f}")
        logger.info(f"  Total training time: {total_time:.2f} seconds")
        logger.info(f"  Final learning rate: {self.metrics['learning_rate'][-1]:.2e}")

# --- Configuration Management ---
class OptimizedTrainingConfig:
    """Optimized configuration for training parameters with noise regularization."""
    
    def __init__(self):
        # Model parameters
        self.hidden_size = None
        self.num_layers = None
        self.noise_std = None
        self.noise_decay = None
        self.min_noise_std = None
        
        # Training parameters
        self.batch_size = None
        self.gradient_accumulation_steps = None
        self._effective_batch_size = None  # Will be calculated on demand
        self.learning_rate = None
        self.weight_decay = None
        self.min_epochs = None
        self.max_epochs = None
        self.patience = None
        self.gradient_clip_norm = None
        self.mixup_alpha = None
        
        # Data parameters
        self.val_split = None
        self.test_split = None
        
        # Scheduler parameters
        self.scheduler_t0 = None
        self.scheduler_t_mult = None
        self.scheduler_eta_min = None
        
        # SWA parameters
        self.swa_start = None
        self.swa_lr = None
        self.swa_freq = None
        self.swa_anneal_epochs = None
        self.swa_anneal_strategy = None
        
        # Learning rate warmup parameters
        self.warmup_epochs = None
        self.warmup_start_lr = None
        
        # EMA parameters
        self.ema_decay = None
        self.ema_start = None
        
        # Adaptive noise parameters
        self.adaptive_noise = None
        self.noise_grad_threshold = None
        self.noise_scale_factor = None
        
        # Dynamic validation parameters
        self.dynamic_val_freq = None
        self.min_val_freq = None
        self.max_val_freq = None
        self.val_stability_threshold = None
        
        # Model saving
        self.save_checkpoint_every = None
        self.save_after_epoch = None
    
    def update(self, **kwargs):
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    @property
    def effective_batch_size(self):
        """Calculate effective batch size from batch_size and gradient_accumulation_steps."""
        return self.batch_size * self.gradient_accumulation_steps

    def validate(self):
        """Validate the configuration."""
        config_dict = {key: value for key, value in self.__dict__.items()
                      if not key.startswith('_')}
        validated_config = validate_training_config(config_dict)

        # Update with validated values
        for key, value in validated_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Additional validation for min_epochs
        if self.min_epochs >= self.max_epochs:
            raise ConfigurationError(f"min_epochs ({self.min_epochs}) must be less than max_epochs ({self.max_epochs})")
        if self.min_epochs < 1:
            raise ConfigurationError(f"min_epochs ({self.min_epochs}) must be at least 1")
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}

# --- GPU Setup and Memory Management ---
def setup_gpu():
    """Setup GPU and check availability."""
    if not torch.cuda.is_available():
        raise SystemError("GPU not found. This script requires a ROCm-enabled or CUDA-enabled GPU.")
    
    device = torch.device("cuda")
    print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    
    # Set memory fraction to avoid OOM - use adaptive fraction based on GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 16:
        memory_fraction = 0.9  # High-end GPUs
    elif gpu_memory_gb >= 8:
        memory_fraction = 0.8  # Mid-range GPUs
    else:
        memory_fraction = 0.7  # Lower-end GPUs
    
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    print(f"Set GPU memory fraction to {memory_fraction} for {gpu_memory_gb:.1f}GB GPU")
    
    return device

def show_gpu_memory_usage():
    """Show current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")

def clear_gpu_memory():
    """Clear GPU memory and garbage collect."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory before clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory after clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    import gc
    gc.collect()

# --- Data Loading and Preprocessing ---
@log_execution_time
@handle_exception
def load_and_preprocess_data(file_path='training_data.h5'):
    """
    Load data that has already been preprocessed.
    Data is loaded in Float16 format for memory efficiency.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Load X and Y, converting to float16 for memory efficiency
            X = np.array(f['X'][:], dtype=np.float32)
            Y = np.array(f['Y'][:], dtype=np.float32)
            T = np.array(f['T'][:], dtype=np.int32)

            # Transpose if the first dimension is smaller than the second
            if X.shape[0] != Y.shape[0]:
                print("Warning: X and Y have different number of samples. Attempting transpose.")
                X = X.T
                Y = Y.T
                print(f"Transposed X shape: {X.shape}, Y shape: {Y.shape}")
            assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
            if T.shape[0] != Y.shape[0]:
                print("Warning: T and Y have different number of samples. Attempting transpose.")
                T = T.T
                print(f"Transposed T shape: {T.shape}")
            assert T.shape[0] == Y.shape[0], "T and Y must have the same number of samples"

            # Validate the data
            validate_array(X, "X")
            validate_array(Y, "Y")
            validate_array(T, "T")
            
            print(f"X shape: {X.shape}, dtype: {X.dtype}")
            print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
            print(f"T shape: {T.shape}, dtype: {T.dtype}")
            print(f"X stats - mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")
            print(f"Y stats - mean: {np.mean(Y):.4f}, std: {np.std(Y):.4f}")
            
            return X, Y, T
        
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def prepare_optimized_data(X, Y, T, batch_size, val_split=0.05, test_split=0.05, device=None):
    """
    Prepare data in the optimized single-tensor format for maximum speed.
    This creates all data on GPU at once and uses slicing for batches.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors and move to GPU
    X_tensor = torch.from_numpy(X).to(device)
    Y_tensor = torch.from_numpy(Y).to(device)
    
    # Calculate split sizes
    total_size = len(X_tensor)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size
    
    # Create indices for splits
    indices = np.arange(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Calculate T timestamps
    assert T.shape[0] == total_size, "T must have the same number of samples as X and Y"
    train_last_ts = T[train_size - 1]
    val_last_ts = T[train_size + val_size - 1]
    test_last_ts = T[train_size + val_size + test_size - 1]
    
    # Split data using indices
    X_train = X_tensor[train_indices]
    Y_train = Y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    Y_val = Y_tensor[val_indices]
    X_test = X_tensor[test_indices]
    Y_test = Y_tensor[test_indices]
    
    print(f"Optimized data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"Train last timestamp: {train_last_ts}")
    print(f"Val last timestamp: {val_last_ts}")
    print(f"Test last timestamp: {test_last_ts}")
    print(f"All data tensors on GPU: {device}")
    
    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
        'batch_size': batch_size,
        'num_train_batches': train_size // batch_size,
        'num_val_batches': val_size // batch_size,
        'num_test_batches': test_size // batch_size,
        'train_last_ts': train_last_ts,
        'val_last_ts': val_last_ts,
        'test_last_ts': test_last_ts
    }

# --- Optimized Model Definition ---
class OptimizedModel(nn.Module):
    """
    Optimized neural network with noise regularization and advanced techniques to prevent overfitting.
    """
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, noise_std):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.training = True
        
        # Input projection with spectral normalization
        self.input_proj = nn.utils.spectral_norm(nn.Linear(input_dim, hidden_size))
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Hidden layers with residual connections and layer normalization
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
                nn.LayerNorm(hidden_size)
            )
            self.layers.append(layer)
        
        # Separate prediction and confidence heads
        self.pred_head = nn.utils.spectral_norm(nn.Linear(hidden_size, output_dim))
        
        # Confidence head with additional non-linearity
        self.conf_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size // 2)),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_size // 2, 1))
        )
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
        
        # Store dimensions for later use
        self.output_dim = output_dim
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass with residual connections and noise regularization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Input projection with noise
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        if self.training:
            # Add Gaussian noise during training
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Hidden layers with residual connections and noise
        for layer in self.layers:
            residual = x
            x = layer(x)
            
            if self.training:
                # Add layer-specific noise
                noise = torch.randn_like(x) * (self.noise_std / 2)  # Reduced noise in deeper layers
                x = x + noise
            
            # 增加残差连接的权重，加强信息流动
            x = 0.7 * x + 0.3 * residual  # 提高残差分支的权重
        
        # Apply minimal noise before heads
        if self.training:
            noise = torch.randn_like(x) * (self.noise_std / 4)  # Even smaller noise at output
            x = x + noise
        
        # Get predictions and confidence
        predictions = self.pred_head(x)
        confidence = self.conf_head(x)
        
        # Concatenate predictions and confidence
        return torch.cat([predictions, confidence], dim=1)

# --- Optimized Loss Functions ---
class ConfidenceWeightedLoss(nn.Module):
    """
    带有置信度加权的损失函数。
    模型输出的最后一个维度作为置信度分数，用于对整体预测进行加权。
    
    特点：
    1. 置信度不足时权重范围在[0.1,1]之间
    2. 引导置信度向目标值(0.2)靠近
    3. 平滑的权重过渡
    """
    def __init__(self, confidence_threshold, alpha, beta, target_confidence):
        super().__init__()
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.alpha = alpha  # 置信度正则化系数
        self.beta = beta   # 阈值损失系数
        self.target_confidence = target_confidence  # 目标置信度值
        self.base_criterion = nn.L1Loss(reduction='none')  # 基础损失函数
        
    def forward(self, outputs, targets):
        # 分离预测值和置信度
        predictions = outputs[:, :-1]  # 所有预测值
        confidence = torch.sigmoid(outputs[:, -1])  # 最后一个值作为置信度
        
        # 计算基础预测误差 (batch_size,)
        base_errors = self.base_criterion(predictions, targets).mean(dim=1)
        
        # 1. 动态权重计算：置信度不足时在[0.1,1]范围内
        # 使用sigmoid将置信度映射到[0.1,1]范围
        dynamic_weight = 0.1 + 0.9 * torch.sigmoid((confidence - self.confidence_threshold) * 5)
        prediction_loss = (dynamic_weight * base_errors).mean()
        
        # 2. 置信度惩罚：引导置信度向目标值靠近
        confidence_penalty = self.alpha * (confidence - self.target_confidence).abs().mean()
        
        # 3. 置信度分布正则化：防止模型输出极端置信度
        confidence_reg = -self.beta * (
            torch.log(confidence + 1e-7) + torch.log(1 - confidence + 1e-7)
        ).mean()
        
        # 4. 阈值损失：平滑过渡
        threshold_dist = torch.abs(confidence - self.target_confidence)
        smooth_weight = torch.exp(-5 * threshold_dist)  # 使用指数衰减创建平滑权重
        threshold_loss = self.beta * (smooth_weight * base_errors).mean()
        
        # 总损失
        total_loss = prediction_loss + confidence_penalty + confidence_reg + threshold_loss
        
        return total_loss

# --- Data Augmentation ---
def mixup_data(x, y, alpha=0.2, device=None):
    """
    Performs Mixup on the input data and labels.
    
    Args:
        x (torch.Tensor): Input data
        y (torch.Tensor): Target data
        alpha (float): Mixup interpolation coefficient
        device (torch.device): Device to use
        
    Returns:
        tuple: (mixed_x, mixed_y, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam

# --- Optimized Training Function ---
def train_model_optimized(data_dict, input_dim, output_dim, 
                         hidden_size=128, num_layers=8, 
                         model_name="optimized_model", config=None):
    """
    Optimized training function using single large tensors on GPU for maximum speed.
    Based on the fast reference code approach.
    """
    best_model_path = f'{model_name}_best_model.pth'

    # Use provided config or create default
    if config is None:
        config = OptimizedTrainingConfig()
        config.update(
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
    
    # Setup logging
    logger = setup_logging(model_name)
    metrics = TrainingMetrics()
    
    # Setup GPU
    device = setup_gpu()
    logger.info(f"Using device: {device}")
    show_gpu_memory_usage()
    
    # Create model with noise regularization
    model = OptimizedModel(input_dim, output_dim, config.hidden_size, config.num_layers, config.noise_std)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Create SWA model
    swa_model = AveragedModel(model)
    
    # Create EMA model
    ema_model = AveragedModel(model, avg_fn=lambda avg, new, num: config.ema_decay * avg + (1 - config.ema_decay) * new)
    
    # Track noise level and gradient norms
    current_noise_std = config.noise_std
    grad_norm_moving_avg = None
    val_freq = config.min_val_freq
    last_val_loss = float('inf')
    
    # Optimizer - Using Adam with more stable parameters
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True  # AMSGrad variant for better stability
    )
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.warmup_start_lr / config.learning_rate,
        total_iters=config.warmup_epochs
    )
    
    # Main learning rate scheduler
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_t_mult,
        eta_min=config.scheduler_eta_min
    )
    
    # Combined scheduler with warmup
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_epochs]
    )
    
    # SWA scheduler
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=config.swa_lr,
        anneal_epochs=config.swa_anneal_epochs,
        anneal_strategy=config.swa_anneal_strategy
    )
    
    # Loss function with confidence weighting
    criterion = ConfidenceWeightedLoss(
        confidence_threshold=args.confidence_threshold,
        alpha=args.confidence_alpha,
        beta=args.confidence_beta,
        target_confidence=args.target_confidence
    )
    
    
    # Check for existing checkpoints and load if found
    checkpoint_path, checkpoint_epoch = find_latest_checkpoint(model_name)
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    if checkpoint_path:
        logger.info(f"Found checkpoint: {checkpoint_path}")
        try:
            training_state = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
            start_epoch = checkpoint_epoch + 1
            best_val_loss = training_state['best_val_loss']
            patience_counter = training_state['patience_counter']
            train_losses = training_state['train_losses']
            val_losses = training_state['val_losses']
            
            logger.info(f"Resuming training from epoch {start_epoch + 1}")
            logger.info(f"Previous best validation loss: {best_val_loss:.6f}")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
    else:
        # Check for best model file (old format)
        if os.path.exists(best_model_path):
            logger.info(f"Found '{best_model_path}'. Loading pre-trained model state.")
            try:
                state_dict = torch.load(best_model_path, map_location=device)
                # Handle DataParallel saved models
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # Remove 'module.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                model.load_state_dict(new_state_dict)
                logger.info("Successfully loaded pre-trained model state.")
            except RuntimeError as e:
                logger.warning(f"Could not load existing model state: {e}")
                logger.info("Starting training from scratch.")
        else:
            logger.info("No checkpoint or best model found. Starting training from scratch.")
    
    # Extract data
    X_train, Y_train = data_dict['train']
    X_val, Y_val = data_dict['val']
    X_test, Y_test = data_dict['test']
    batch_size = data_dict['batch_size']
    num_train_batches = data_dict['num_train_batches']
    num_val_batches = data_dict['num_val_batches']
    num_test_batches = data_dict['num_test_batches']
    
    logger.info(f"Starting optimized training for {config.max_epochs} epochs...")
    logger.info(f"Minimum epochs before early stopping: {config.min_epochs}")
    logger.info(f"Batch size: {batch_size}, Train batches: {num_train_batches}, Val batches: {num_val_batches}")
    logger.info(f"Starting from epoch {start_epoch + 1}")
    
    try:
        for epoch in range(start_epoch, config.max_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            epoch_train_loss = 0
            optimizer.zero_grad()  # Zero gradients at the start of epoch
            
            for i in range(num_train_batches):
                # Slice the data from the single large tensor - this is extremely fast
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                inputs = X_train[start_idx:end_idx]
                targets = Y_train[start_idx:end_idx]
                
                # Apply Mixup augmentation during training
                if config.mixup_alpha > 0:
                    inputs, targets, _ = mixup_data(inputs, targets, config.mixup_alpha, device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                # Step optimization after accumulating gradients
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_train_loss += loss.item()
            
            # Validation phase (with dynamic frequency)
            if (epoch + 1) % val_freq == 0 or epoch == 0:
                model.eval()
                epoch_val_loss = 0
                
                with torch.no_grad():
                    for i in range(num_val_batches):
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size
                        inputs = X_val[start_idx:end_idx]
                        targets = Y_val[start_idx:end_idx]
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        epoch_val_loss += loss.item()
                
                # Calculate average losses
                avg_train_loss = epoch_train_loss / num_train_batches
                avg_val_loss = epoch_val_loss / num_val_batches
            else:
                # Skip validation, use previous validation loss
                avg_train_loss = epoch_train_loss / num_train_batches
                avg_val_loss = val_losses[-1] if val_losses else float('inf')
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Calculate gradient norm for adaptive noise
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            # Update gradient norm moving average
            if grad_norm_moving_avg is None:
                grad_norm_moving_avg = total_grad_norm
            else:
                grad_norm_moving_avg = 0.95 * grad_norm_moving_avg + 0.05 * total_grad_norm
            
            # Adaptive noise based on gradient norm
            if config.adaptive_noise:
                noise_scale = min(1.0, config.noise_grad_threshold / (total_grad_norm + 1e-8))
                current_noise_std = max(
                    config.min_noise_std,
                    config.noise_std * noise_scale * config.noise_scale_factor
                )
            else:
                current_noise_std = max(
                    config.min_noise_std,
                    current_noise_std * config.noise_decay
                )
            
            # Update model noise level
            if isinstance(model, nn.DataParallel):
                model.module.noise_std = current_noise_std
            else:
                model.noise_std = current_noise_std
            
            # Learning rate and SWA scheduling
            if epoch < config.swa_start:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            else:
                swa_scheduler.step()
                current_lr = config.swa_lr
                
                # Update SWA model
                if (epoch + 1) % config.swa_freq == 0:
                    swa_model.update_parameters(model)
            
            # Update EMA model
            if epoch >= config.ema_start:
                ema_model.update_parameters(model)
            
            # Dynamic validation frequency
            if config.dynamic_val_freq:
                val_loss_change = abs(avg_val_loss - last_val_loss)
                if val_loss_change < config.val_stability_threshold:
                    val_freq = min(val_freq + 1, config.max_val_freq)
                else:
                    val_freq = config.min_val_freq
                last_val_loss = avg_val_loss
            
            # Update metrics
            metrics.update(
                epoch + 1, 
                avg_train_loss, 
                avg_val_loss, 
                current_lr,
                noise_std=current_noise_std
            )
            
            # We must synchronize the GPU before stopping the timer for an accurate measurement
            torch.cuda.synchronize()
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            # Log when min_epochs is reached (early stopping becomes active)
            if epoch + 1 == config.min_epochs:
                logger.info(f"Epoch [{epoch+1:3d}/{config.max_epochs}]: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}, "
                          f"LR: {current_lr:.2e}, "
                          f"Time: {epoch_time:.2f}s, "
                          f"ETA: {int(epoch_time * (config.max_epochs - epoch - 1) // 3600)}h "
                          f"{int((epoch_time * (config.max_epochs - epoch - 1) % 3600) // 60)}m "
                          f"{int((epoch_time * (config.max_epochs - epoch - 1) % 60))}s")
                logger.info(f"  -> Reached minimum epochs ({config.min_epochs}). Early stopping is now active.")
            else:
                logger.info(f"Epoch [{epoch+1:3d}/{config.max_epochs}]: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}, "
                          f"LR: {current_lr:.2e}, "
                          f"Time: {epoch_time:.2f}s, "
                          f"ETA: {int(epoch_time * (config.max_epochs - epoch - 1) // 3600)}h "
                          f"{int((epoch_time * (config.max_epochs - epoch - 1) % 3600) // 60)}m "
                          f"{int((epoch_time * (config.max_epochs - epoch - 1) % 60))}s")
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Only save best model if we're past save_after_epoch
                if epoch + 1 > config.save_after_epoch:
                    # Save best model with parameters
                    if isinstance(model, nn.DataParallel):
                        model_instance = model.module
                    else:
                        model_instance = model
                        
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_params': {
                            'input_dim': model_instance.input_dim,
                            'output_dim': model_instance.output_dim,
                            'hidden_size': model_instance.hidden_size,
                            'num_layers': len(model_instance.layers)
                        }
                    }, best_model_path)
                    logger.info(f"  -> New best model saved! Val Loss: {best_val_loss:.6f}")
                else:
                    logger.info(f"  -> New best val loss: {best_val_loss:.6f} (not saving model until after epoch {config.save_after_epoch})")
            else:
                patience_counter += 1
                # Only apply early stopping after min_epochs have been reached
                if epoch + 1 >= config.min_epochs and patience_counter >= config.patience:
                    logger.info(f"  -> Early stopping triggered after {config.patience} epochs without improvement (min_epochs: {config.min_epochs} reached)")
                    break
            
            # Save checkpoint periodically after save_after_epoch
            if (epoch + 1) % config.save_checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_val_loss,
                    patience_counter, train_losses, val_losses, model_name,
                    save_after_epoch=config.save_after_epoch
                )
                if epoch + 1 > config.save_after_epoch:
                    logger.info(f"  -> Checkpoint saved for epoch {epoch + 1}")
                plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'], model_name)
        
        # Update batch normalization statistics for SWA model
        logger.info("Updating batch normalization statistics for SWA model...")
        swa_model.eval()
        torch.optim.swa_utils.update_bn(
            loader=[(X_train[i:i + batch_size], Y_train[i:i + batch_size]) 
                   for i in range(0, len(X_train), batch_size)],
            model=swa_model,
            device=device
        )
        
        # Save SWA model
        swa_model_path = f'{model_name}_swa_model.pth'
        torch.save(swa_model.state_dict(), swa_model_path)
        logger.info(f"SWA model saved to {swa_model_path}")
        
        # Save EMA model
        ema_model_path = f'{model_name}_ema_model.pth'
        torch.save(ema_model.state_dict(), ema_model_path)
        logger.info(f"EMA model saved to {ema_model_path}")
        
        # Load the best model for comparison
        logger.info("Loading the best model state for comparison.")
        state_dict = torch.load(best_model_path, map_location=device)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        
        # Final evaluation on test set for all models
        def evaluate_model(model, model_name):
            model.eval()
            test_loss = 0
            
            with torch.no_grad():
                for i in range(num_test_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    inputs = X_test[start_idx:end_idx]
                    targets = Y_test[start_idx:end_idx]
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
            
            return test_loss / num_test_batches
        
        # Evaluate all models
        regular_test_loss = evaluate_model(model, "Regular")
        swa_test_loss = evaluate_model(swa_model, "SWA")
        ema_test_loss = evaluate_model(ema_model, "EMA")
        
        logger.info(f"Final Test Losses:")
        logger.info(f"  Regular Model: {regular_test_loss:.6f}")
        logger.info(f"  SWA Model: {swa_test_loss:.6f}")
        logger.info(f"  EMA Model: {ema_test_loss:.6f}")
        
        # Use the best performing model
        final_test_loss = min(regular_test_loss, swa_test_loss, ema_test_loss)
        best_model_type = "Regular" if regular_test_loss == final_test_loss else ("SWA" if swa_test_loss == final_test_loss else "EMA")
        logger.info(f"Best performing model: {best_model_type} with loss: {final_test_loss:.6f}")
        
        # Log training summary
        metrics.log_summary(logger)
        
        return model, train_losses, val_losses, final_test_loss
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model...")
        torch.save(model.state_dict(), f'{model_name}_interrupted.pth')
        plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'], model_name)
        return model, train_losses, val_losses, None
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e

# --- Utility Functions ---

def predict_with_confidence(model, inputs, confidence_threshold=0.5):
    """
    使用模型进行预测，并根据置信度过滤结果。
    
    Args:
        model: 训练好的模型
        inputs: 输入数据
        confidence_threshold: 置信度阈值
        
    Returns:
        tuple: (filtered_predictions, confidences, mask)
        - filtered_predictions: 经过置信度过滤的预测值（低置信度的预测被置为0）
        - confidences: 预测的置信度值
        - mask: 置信度掩码（布尔值）
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs[:, :-1]  # 预测值
        confidences = torch.sigmoid(outputs[:, -1])  # 置信度
        
        # 创建置信度掩码
        mask = confidences > confidence_threshold
        
        # 将低置信度的预测置为0
        filtered_predictions = predictions * mask.unsqueeze(1)
        
        return filtered_predictions, confidences, mask

def plot_losses(train_losses, val_losses, train_last_ts, val_last_ts, test_last_ts, model_name="optimized_model"):
    """
    Enhanced plotting with better visualization.
    """
    # Remove first 100 elements if length is bigger than 200
    if len(train_losses) > 200:
        train_losses = train_losses[100:]
    else:
        train_losses = train_losses[round(len(train_losses)*2/3):]
    if len(val_losses) > 200:
        val_losses = val_losses[100:]
    else:
        val_losses = val_losses[round(len(val_losses)*2/3):]
    
    plt.figure(figsize=(12, 8))
    
    # Main loss plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label=f'Training Loss (till {train_last_ts})', alpha=0.8)
    plt.plot(val_losses, label=f'Validation Loss (till {val_last_ts})', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'train & val loss - {test_last_ts}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot
    plt.subplot(2, 2, 2)
    plt.semilogy(train_losses, label=f'Training Loss (till {train_last_ts})', alpha=0.8)
    plt.semilogy(val_losses, label=f'Validation Loss (till {val_last_ts})', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss in Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss difference
    plt.subplot(2, 2, 3)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(loss_diff, label='|Train - Val|', color='red', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Monitor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(2, 2, 4)
    window = min(10, len(val_losses) // 4)
    if window > 1:
        val_ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(val_losses)), val_ma, label=f'Val Loss (MA-{window})', alpha=0.8)
    plt.plot(val_losses, label=f'Validation Loss (till {val_last_ts})', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss with Moving Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_training_config(config, model_name):
    """Save training configuration to a JSON file."""
    import json
    
    config_dict = {key: value for key, value in config.__dict__.items() 
                   if not key.startswith('_')}
    
    config_file = f"{model_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {config_file}")

# --- Knowledge Distillation (Teacher -> Student) ---
def _parse_model_params_from_filename(filename):
    """Try to parse hidden_size and num_layers from filename. Returns (hs, nl) or (None, None)."""
    try:
        patterns = [
            r".*?(\d+)_layers_(\d+).*?\\.pth$",
            r".*?model_(\d+)_layers_(\d+).*?\\.pth$",
            r".*?(\d+)_(\d+).*?\\.pth$",
        ]
        for pat in patterns:
            m = re.match(pat, os.path.basename(filename))
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None, None

def _infer_arch_from_state_dict(state_dict, input_dim, output_dim):
    """Infer (hidden_size, num_layers) from an OptimizedModel state_dict."""
    hidden_size = None
    num_layers = 0
    # Remove 'module.' prefix if present for inspection only
    keys = list(state_dict.keys())
    if any(k.startswith('module.') for k in keys):
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        keys = list(state_dict.keys())
    # Infer hidden_size from input_proj
    if 'input_proj.weight' in state_dict:
        w = state_dict['input_proj.weight']
        # weight shape: [hidden_size, input_dim]
        if w.dim() == 2 and w.shape[1] == input_dim:
            hidden_size = int(w.shape[0])
    # Infer num_layers by counting distinct indices under 'layers.{i}.'
    layer_indices = set()
    for k in keys:
        if k.startswith('layers.'):
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if layer_indices:
        num_layers = max(layer_indices) + 1
    return hidden_size, num_layers

def _load_teacher_model(teacher_path, device, input_dim, output_dim):
    """Load a teacher OptimizedModel from path, inferring its architecture if needed."""
    raw = torch.load(teacher_path, map_location=device)
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise RuntimeError("Unsupported teacher checkpoint format")
    # Try filename parse first
    hs, nl = _parse_model_params_from_filename(teacher_path)
    if hs is None or nl is None:
        ihs, inl = _infer_arch_from_state_dict(state_dict, input_dim, output_dim)
        hs = hs or ihs
        nl = nl or inl
    if hs is None or nl is None:
        raise RuntimeError("Failed to infer teacher architecture (hidden_size/num_layers). Please follow filename pattern or supply compatible checkpoint.")
    teacher = OptimizedModel(input_dim, output_dim, hidden_size=hs, num_layers=nl, noise_std=0.0)
    # Strip DataParallel prefix if present
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state[k[7:]] = v
        else:
            cleaned_state[k] = v
    teacher.load_state_dict(cleaned_state, strict=False)
    teacher.to(device)
    teacher.eval()
    return teacher, hs, nl

def distill_model_optimized(
    data_dict,
    input_dim,
    output_dim,
    teacher_model_path,
    student_hidden_size=128,
    student_num_layers=2,
    temperature=3.0,
    alpha=0.7,
    model_name="distilled_student",
    base_config=None,
):
    """
    Distill a student model from a teacher using the optimized training pipeline.

    Args:
        data_dict: Output of prepare_optimized_data
        input_dim (int): Input feature dimension
        output_dim (int): Output dimension
        teacher_model_path (str): Path to teacher .pth file
        student_hidden_size (int): Student hidden size
        student_num_layers (int): Student number of layers
        temperature (float): Temperature for soft targets
        alpha (float): Weight for distillation loss vs ground-truth loss
        model_name (str): Base name for saved artifacts
        base_config (OptimizedTrainingConfig|None): Optional training config to reuse

    Returns:
        (student_model, train_losses, val_losses, final_val_loss)
    """
    best_model_path = f"{model_name}_best_model.pth"

    # Setup logging and device
    logger = setup_logging(model_name)
    metrics = TrainingMetrics()
    device = setup_gpu()
    logger.info(f"Using device: {device}")
    show_gpu_memory_usage()

    # Load teacher
    teacher, teacher_hs, teacher_nl = _load_teacher_model(teacher_model_path, device, input_dim, output_dim)
    logger.info(f"Loaded teacher model (hidden_size={teacher_hs}, num_layers={teacher_nl}) from {teacher_model_path}")

    # Build student
    config = base_config or OptimizedTrainingConfig()
    config.update(hidden_size=student_hidden_size, num_layers=student_num_layers)

    student = OptimizedModel(input_dim, output_dim, config.hidden_size, config.num_layers, config.noise_std)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for student!")
        student = nn.DataParallel(student)
    student.to(device)

    # SWA and EMA for student
    swa_model = AveragedModel(student)
    ema_model = AveragedModel(student, avg_fn=lambda avg, new, num: config.ema_decay * avg + (1 - config.ema_decay) * new)

    # Optimizer and schedulers
    optimizer = optim.Adam(
        student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8, amsgrad=True
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=config.warmup_start_lr / config.learning_rate, total_iters=config.warmup_epochs
    )
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler_t0, T_mult=config.scheduler_t_mult, eta_min=config.scheduler_eta_min
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_epochs])
    swa_scheduler = SWALR(
        optimizer, swa_lr=config.swa_lr, anneal_epochs=config.swa_anneal_epochs, anneal_strategy=config.swa_anneal_strategy
    )

    # Losses
    l1 = nn.L1Loss()

    # Extract data tensors
    X_train, Y_train = data_dict['train']
    X_val, Y_val = data_dict['val']
    batch_size = data_dict['batch_size']
    num_train_batches = data_dict['num_train_batches']
    num_val_batches = data_dict['num_val_batches']

    logger.info(
        f"Starting distillation: student(hs={config.hidden_size}, layers={config.num_layers}), temp={temperature}, alpha={alpha}"
    )

    current_noise_std = config.noise_std
    grad_norm_moving_avg = None
    val_freq = config.min_val_freq
    last_val_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    try:
        for epoch in range(0, config.max_epochs):
            epoch_start_time = time.time()
            student.train()
            epoch_train_loss = 0.0
            optimizer.zero_grad()

            for i in range(num_train_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                inputs = X_train[start_idx:end_idx]
                targets = Y_train[start_idx:end_idx]

                if config.mixup_alpha > 0:
                    inputs, targets, _ = mixup_data(inputs, targets, config.mixup_alpha, device)

                with torch.no_grad():
                    teacher_outputs = teacher(inputs)

                student_outputs = student(inputs)
                loss_gt = l1(student_outputs, targets)
                loss_kd = l1(student_outputs / temperature, teacher_outputs / temperature)
                loss = ((1 - alpha) * loss_gt + alpha * loss_kd) / config.gradient_accumulation_steps

                loss.backward()
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=config.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_train_loss += loss.item()

            # Validation
            if (epoch + 1) % val_freq == 0 or epoch == 0:
                student.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for i in range(num_val_batches):
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size
                        inputs = X_val[start_idx:end_idx]
                        targets = Y_val[start_idx:end_idx]
                        teacher_outputs = teacher(inputs)
                        student_outputs = student(inputs)
                        loss_gt = l1(student_outputs, targets)
                        loss_kd = l1(student_outputs / temperature, teacher_outputs / temperature)
                        loss = (1 - alpha) * loss_gt + alpha * loss_kd
                        epoch_val_loss += loss.item()

                avg_train_loss = epoch_train_loss / max(1, num_train_batches)
                avg_val_loss = epoch_val_loss / max(1, num_val_batches)
            else:
                avg_train_loss = epoch_train_loss / max(1, num_train_batches)
                avg_val_loss = val_losses[-1] if val_losses else float('inf')

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Gradient norm for adaptive noise
            total_grad_norm_sq = 0.0
            for p in student.parameters():
                if p.grad is not None:
                    total_grad_norm_sq += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm_sq ** 0.5
            if grad_norm_moving_avg is None:
                grad_norm_moving_avg = total_grad_norm
            else:
                grad_norm_moving_avg = 0.95 * grad_norm_moving_avg + 0.05 * total_grad_norm

            # Adaptive noise on student
            if config.adaptive_noise:
                noise_scale = min(1.0, config.noise_grad_threshold / (total_grad_norm + 1e-8))
                current_noise_std = max(config.min_noise_std, config.noise_std * noise_scale * config.noise_scale_factor)
            else:
                current_noise_std = max(config.min_noise_std, current_noise_std * config.noise_decay)
            if isinstance(student, nn.DataParallel):
                student.module.noise_std = current_noise_std
            else:
                student.noise_std = current_noise_std

            # LR scheduling and SWA
            if epoch < config.swa_start:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            else:
                swa_scheduler.step()
                current_lr = config.swa_lr
                if (epoch + 1) % config.swa_freq == 0:
                    swa_model.update_parameters(student)

            # EMA update
            if epoch >= config.ema_start:
                ema_model.update_parameters(student)

            # Dynamic val frequency
            if config.dynamic_val_freq:
                val_loss_change = abs(avg_val_loss - last_val_loss)
                if val_loss_change < config.val_stability_threshold:
                    val_freq = min(val_freq + 1, config.max_val_freq)
                else:
                    val_freq = config.min_val_freq
                last_val_loss = avg_val_loss

            # Metrics/logging
            metrics.update(epoch + 1, avg_train_loss, avg_val_loss, current_lr, noise_std=current_noise_std)
            torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"[KD] Epoch [{epoch+1:3d}/{config.max_epochs}] Train: {avg_train_loss:.6f} Val: {avg_val_loss:.6f} LR: {current_lr:.2e} Time: {epoch_time:.2f}s"
            )

            # Early stopping + save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(student.state_dict(), best_model_path)
                logger.info(f"  -> New best distilled student saved to {best_model_path}")
            else:
                patience_counter += 1
                if epoch + 1 >= config.min_epochs and patience_counter >= config.patience:
                    logger.info("  -> Early stopping for distillation")
                    break

        # Save SWA/EMA variants
        swa_model_path = f"{model_name}_swa_model.pth"
        torch.save(swa_model.state_dict(), swa_model_path)
        logger.info(f"SWA student model saved to {swa_model_path}")
        ema_model_path = f"{model_name}_ema_model.pth"
        torch.save(ema_model.state_dict(), ema_model_path)
        logger.info(f"EMA student model saved to {ema_model_path}")

        metrics.log_summary(logger)
        return student, train_losses, val_losses, best_val_loss

    except KeyboardInterrupt:
        logger.info("Distillation interrupted by user. Saving current student model...")
        torch.save(student.state_dict(), f"{model_name}_interrupted.pth")
        return student, train_losses, val_losses, None
    except Exception as e:
        logger.error(f"Distillation failed with error: {e}")
        raise e

# --- Main Execution ---
def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Optimized PyTorch Training')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=1536,
                      help='Hidden size for the model (default: 1536)')
    parser.add_argument('--num_layers', type=int, default=12,
                      help='Number of layers in the model (default: 12)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                      help='Number of gradient accumulation steps (default: 2)')
    parser.add_argument('--min_epochs', type=int, default=50,
                      help='Minimum number of epochs to train (default: 50)')
    parser.add_argument('--max_epochs', type=int, default=200,
                      help='Maximum number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                      help='Patience for early stopping (default: 20)')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                      help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                      help='Mixup interpolation factor (default: 0.2)')
    
    # Learning rate parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for optimizer (default: 1e-5)')
    
    # Data split parameters
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Validation split ratio (default: 0.1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                      help='Test split ratio (default: 0.1)')
    
    # Noise parameters
    parser.add_argument('--noise_std', type=float, default=0.03,
                      help='Initial noise standard deviation (default: 0.03)')
    parser.add_argument('--noise_decay', type=float, default=0.98,
                      help='Noise decay rate per epoch (default: 0.98)')
    parser.add_argument('--min_noise_std', type=float, default=0.0001,
                      help='Minimum noise standard deviation (default: 0.0001)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_t0', type=int, default=10,
                      help='Initial cycle length for cosine annealing (default: 10)')
    parser.add_argument('--scheduler_t_mult', type=int, default=2,
                      help='Cycle length multiplier (default: 2)')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6,
                      help='Minimum learning rate for scheduler (default: 1e-6)')
    
    # SWA parameters
    parser.add_argument('--swa_start', type=int, default=50,
                      help='Epoch to start SWA from (default: 50)')
    parser.add_argument('--swa_lr', type=float, default=1e-4,
                      help='SWA learning rate (default: 1e-4)')
    parser.add_argument('--swa_freq', type=int, default=5,
                      help='SWA model update frequency (default: 5)')
    parser.add_argument('--swa_anneal_epochs', type=int, default=10,
                      help='Number of epochs to anneal for SWA (default: 10)')
    parser.add_argument('--swa_anneal_strategy', type=str, default='cos',
                      help='SWA annealing strategy (default: cos)')
    
    # Learning rate warmup parameters
    parser.add_argument('--warmup_epochs', type=int, default=10,
                      help='Number of warmup epochs (default: 10)')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-7,
                      help='Starting learning rate for warmup (default: 1e-7)')
    
    # EMA parameters
    parser.add_argument('--ema_decay', type=float, default=0.999,
                      help='EMA decay rate (default: 0.999)')
    parser.add_argument('--ema_start', type=int, default=20,
                      help='Epoch to start EMA from (default: 20)')
    
    # Adaptive noise parameters
    parser.add_argument('--adaptive_noise', type=bool, default=True,
                      help='Enable adaptive noise scaling (default: True)')
    parser.add_argument('--noise_grad_threshold', type=float, default=1.0,
                      help='Gradient norm threshold for noise scaling (default: 1.0)')
    parser.add_argument('--noise_scale_factor', type=float, default=0.1,
                      help='Scaling factor for adaptive noise (default: 0.1)')
    
    # Dynamic validation parameters
    parser.add_argument('--dynamic_val_freq', type=bool, default=True,
                      help='Enable dynamic validation frequency (default: True)')
    parser.add_argument('--min_val_freq', type=int, default=1,
                      help='Minimum validation frequency (default: 1)')
    parser.add_argument('--max_val_freq', type=int, default=5,
                      help='Maximum validation frequency (default: 5)')
    parser.add_argument('--val_stability_threshold', type=float, default=0.01,
                      help='Threshold for considering training stable (default: 0.01)')
    
    # Model saving parameters
    parser.add_argument('--save_checkpoint_every', type=int, default=2,
                      help='Save checkpoint every N epochs (default: 2)')
    parser.add_argument('--save_after_epoch', type=int, default=0,
                      help='Start saving model weights after this epoch (default: 0)')
    
    # Model name
    parser.add_argument('--model_name', type=str, default="optimized_model",
                      help='Base name for saved model files (default: optimized_model)')
    
    # Confidence weighted loss parameters
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                      help='Confidence threshold for loss weighting (default: 0.6)')
    parser.add_argument('--confidence_alpha', type=float, default=0.2,
                      help='Alpha coefficient for confidence regularization (default: 0.2)')
    parser.add_argument('--confidence_beta', type=float, default=0.1,
                      help='Beta coefficient for threshold loss (default: 0.1)')
    parser.add_argument('--target_confidence', type=float, default=0.6,
                      help='Target confidence value (default: 0.6)')
    
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting Optimized PyTorch Training")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_args()
    
    # Load and prepare data
    X, Y, T = load_and_preprocess_data()
    
    # Determine dimensions
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Create configuration with validation
    config = OptimizedTrainingConfig()
    config.update(
        # Model parameters
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        noise_std=args.noise_std,
        noise_decay=args.noise_decay,
        min_noise_std=args.min_noise_std,
        
        # Training parameters
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm,
        mixup_alpha=args.mixup_alpha,
        
        # Data parameters
        val_split=args.val_split,
        test_split=args.test_split,
        
        # Scheduler parameters
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        scheduler_eta_min=args.scheduler_eta_min,
        
        # SWA parameters
        swa_start=args.swa_start,
        swa_lr=args.swa_lr,
        swa_freq=args.swa_freq,
        swa_anneal_epochs=args.swa_anneal_epochs,
        swa_anneal_strategy=args.swa_anneal_strategy,
        
        # Learning rate warmup parameters
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        
        # EMA parameters
        ema_decay=args.ema_decay,
        ema_start=args.ema_start,
        
        # Adaptive noise parameters
        adaptive_noise=args.adaptive_noise,
        noise_grad_threshold=args.noise_grad_threshold,
        noise_scale_factor=args.noise_scale_factor,
        
        # Dynamic validation parameters
        dynamic_val_freq=args.dynamic_val_freq,
        min_val_freq=args.min_val_freq,
        max_val_freq=args.max_val_freq,
        val_stability_threshold=args.val_stability_threshold,
        
        # Model saving parameters
        save_checkpoint_every=args.save_checkpoint_every,
        save_after_epoch=args.save_after_epoch
    )
    
    # Validate configuration
    config.validate()
    
    # Check GPU memory before training
    if not check_gpu_memory():
        print("Warning: Insufficient GPU memory detected. Consider reducing batch size.")
        config.batch_size = min(config.batch_size, 64)
    
    print(f"Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Original output dimension: {output_dim}")
    print(f"  Model output dimension: {output_dim} (direct predictions)")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Min epochs: {config.min_epochs}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Patience: {config.patience}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Noise std: {config.noise_std}")
    print(f"  Noise decay: {config.noise_decay}")
    
    # Setup GPU
    device = setup_gpu()
    
    # Prepare optimized data
    data_dict = prepare_optimized_data(
        X, Y, T,
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        device=device
    )
    data_dict['T'] = T
    
    # Train the model
    model, train_losses, val_losses, test_loss = train_model_optimized(
        data_dict, input_dim, output_dim,
        model_name=args.model_name,
        config=config
    )
    
    # Save configuration
    save_training_config(config, args.model_name)
    
    # Plot results
    plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'], model_name=args.model_name)

    print("\nOptimized training complete!")
    print("Files saved:")
    print(f"  - {args.model_name}_best_model.pth (best model)")
    print(f"  - {args.model_name}_config.json (configuration)")
    print(f"  - {args.model_name}_training.log (training log)")
    print(f"  - {args.model_name}_loss_analysis.png (loss curves)")
    if test_loss is not None:
        print(f"  - Final test loss: {test_loss:.6f}")
    
    # Clear GPU memory
    clear_gpu_memory()