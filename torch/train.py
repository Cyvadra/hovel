import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import functional as F
import warnings
from torch.cuda.amp import GradScaler, autocast
import logging
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

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
    """Optimized configuration for training parameters."""
    
    def __init__(self):
        # Model parameters
        self.hidden_size = 512
        self.num_layers = 4
        self.dropout_rate = 0.1
        
        # Training parameters - optimized for speed
        self.batch_size = 128  # Increased for better GPU utilization
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.max_epochs = 300
        self.patience = 30
        self.gradient_clip_norm = 1.0
        
        # Data parameters
        self.val_split = 0.05
        self.test_split = 0.05
        
        # Loss parameters
        self.mse_weight = 0.7
        self.huber_weight = 0.3
        self.huber_delta = 1.0
        
        # Scheduler parameters
        self.scheduler_t0 = 20
        self.scheduler_t_mult = 2
        self.scheduler_eta_min = 1e-6
        
        # Mixed precision
        self.use_mixed_precision = True
        
        # Model saving
        self.save_checkpoint_every = 50
    
    def update(self, **kwargs):
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")

# --- GPU Setup and Memory Management ---
def setup_gpu():
    """Setup GPU and check availability."""
    if not torch.cuda.is_available():
        raise SystemError("GPU not found. This script requires a ROCm-enabled or CUDA-enabled GPU.")
    
    device = torch.device("cuda")
    print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.9)
    
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
def load_and_preprocess_data(file_path='training_data.h5'):
    """
    Load and preprocess data with improved normalization and validation.
    """
    with h5py.File(file_path, 'r') as f:
        # Load X and Y, ensuring float32 type
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)
        T = np.array(f['T'][:], dtype=np.int32)

        # Transpose if the first dimension is smaller than the second
        if X.shape[0] != Y.shape[0]:
            print("Warning: X and Y have different number of samples. Attempting transpose.")
            X = X.T
            Y = Y.T
            print(f"Transposed X shape: {X.shape}, Y shape: {Y.shape}")
        if T.shape[0] != Y.shape[0]:
            print("Warning: T and Y have different number of samples. Attempting transpose.")
            T = T.T
            print(f"Transposed T shape: {T.shape}")

        # Improved preprocessing: Standardize X and apply robust scaling to Y
        # Standardize X (zero mean, unit variance)
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True)
        X_std = np.where(X_std == 0, 1.0, X_std)  # Avoid division by zero
        X = (X - X_mean) / X_std
        
        # Robust scaling for Y using median and IQR
        Y_median = np.median(Y, axis=0, keepdims=True).astype(np.float32)
        Y_q75, Y_q25 = np.percentile(Y, [75, 25], axis=0, keepdims=True).astype(np.float32)
        Y_iqr = Y_q75 - Y_q25
        Y_iqr = np.where(Y_iqr == 0, 1.0, Y_iqr).astype(np.float32)  # Avoid division by zero
        Y = ((Y - Y_median) / Y_iqr).astype(np.float32)

        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
        print(f"T shape: {T.shape}, dtype: {T.dtype}")
        print(f"X stats - mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")
        print(f"Y stats - mean: {np.mean(Y):.4f}, std: {np.std(Y):.4f}")
        
    return X, Y, T

def prepare_optimized_data(X, Y, T, batch_size, val_split=0.05, test_split=0.05, device=None):
    """
    Prepare data in the optimized single-tensor format for maximum speed.
    This creates all data on GPU at once and uses slicing for batches.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors and move to GPU
    X_tensor = torch.from_numpy(X).float().to(device)
    Y_tensor = torch.from_numpy(Y).float().to(device)
    
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
    Optimized neural network with modern best practices and speed optimizations.
    """
    def __init__(self, input_dim, output_dim, hidden_size=512, num_layers=4, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
        
        # Output projection: 3*output_dim (p1, p2, p3 for each output dimension)
        self.output_proj = nn.Linear(hidden_size, 3 * output_dim)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3*output_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Hidden layers with residual connections
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Output projection
        x = self.output_proj(x)
        
        return x

# --- Optimized Loss Functions ---
class HuberLoss(nn.Module):
    """
    Huber loss combines the best properties of L1 and L2 loss.
    More robust to outliers than MSE.
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)

class WeightedCombinedLoss(nn.Module):
    """
    Weighted combined loss function for 3*output_dim model outputs.
    """
    def __init__(self, output_dim, mse_weight=0.7, huber_weight=0.3, huber_delta=1.0):
        super().__init__()
        self.output_dim = output_dim
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        self.huber_loss = HuberLoss(delta=huber_delta)
        self.step_count = 0
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Model output of shape (batch_size, 3*output_dim)
            target (torch.Tensor): Target values of shape (batch_size, output_dim)
        
        Returns:
            torch.Tensor: Weighted loss value
        """
        batch_size = pred.shape[0]
        
        # Reshape predictions to separate p1, p2, p3 for each output dimension
        pred_reshaped = pred.view(batch_size, self.output_dim, 3)
        
        # Extract p1, p2, p3 for each output dimension
        p1 = pred_reshaped[:, :, 0]  # weight parameters ∈ [-1,1]
        p2 = pred_reshaped[:, :, 1]  # negative predictions
        p3 = pred_reshaped[:, :, 2]  # positive predictions
        
        # Calculate weights based on p1
        p1_clamped = torch.clamp(p1, -1.0, 1.0)
        weight_p2 = torch.abs(p1_clamped - 1) / 2
        weight_p3 = torch.abs(p1_clamped + 1) / 2
        
        # Calculate individual losses for p2 and p3
        loss_p2 = self.mse_loss(p2, target)
        loss_p3 = self.mse_loss(p3, target)
        
        # Apply weights and combine losses
        weighted_loss = weight_p2 * loss_p2 + weight_p3 * loss_p3
        
        # Take mean across batch and output dimensions
        final_loss = torch.mean(weighted_loss)
        
        # Print statistics every 1000 steps for monitoring
        self.step_count += 1
        if self.step_count % 1000 == 0:
            with torch.no_grad():
                p1_mean = torch.mean(p1).item()
                p1_std = torch.std(p1).item()
                p2_mean = torch.mean(p2).item()
                p3_mean = torch.mean(p3).item()
                weight_p2_mean = torch.mean(weight_p2).item()
                weight_p3_mean = torch.mean(weight_p3).item()
                
                print(f"Step {self.step_count} - p1: mean={p1_mean:.3f}, std={p1_std:.3f}, "
                      f"p2_mean={p2_mean:.3f}, p3_mean={p3_mean:.3f}, "
                      f"w2={weight_p2_mean:.3f}, w3={weight_p3_mean:.3f}")
        
        return final_loss

# --- Optimized Training Function ---
def train_model_optimized(data_dict, input_dim, output_dim, 
                         hidden_size=512, num_layers=4, dropout_rate=0.1, 
                         model_name="optimized_model", config=None):
    """
    Optimized training function using single large tensors on GPU for maximum speed.
    Based on the fast reference code approach.
    """
    # Use provided config or create default
    if config is None:
        config = OptimizedTrainingConfig()
        config.update(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
    
    # Setup logging
    logger = setup_logging(model_name)
    metrics = TrainingMetrics()
    
    # Setup GPU
    device = setup_gpu()
    logger.info(f"Using device: {device}")
    show_gpu_memory_usage()
    
    # Create model
    model = OptimizedModel(input_dim, output_dim, config.hidden_size, config.num_layers, config.dropout_rate)
    
    # Create model-specific file names
    best_model_path = f'{model_name}_best_model.pth'
    
    # Load existing model if available
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
        logger.info(f"No '{best_model_path}' found. Starting training from scratch.")

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_t_mult,
        eta_min=config.scheduler_eta_min
    )
    
    # Loss function
    criterion = WeightedCombinedLoss(
        output_dim, 
        mse_weight=config.mse_weight, 
        huber_weight=config.huber_weight, 
        huber_delta=config.huber_delta
    )
    
    # Mixed precision setup
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Extract data
    X_train, Y_train = data_dict['train']
    X_val, Y_val = data_dict['val']
    X_test, Y_test = data_dict['test']
    batch_size = data_dict['batch_size']
    num_train_batches = data_dict['num_train_batches']
    num_val_batches = data_dict['num_val_batches']
    num_test_batches = data_dict['num_test_batches']
    
    # Training parameters
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting optimized training for {config.max_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Train batches: {num_train_batches}, Val batches: {num_val_batches}")
    
    try:
        for epoch in range(config.max_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            epoch_train_loss = 0
            
            for i in range(num_train_batches):
                optimizer.zero_grad()
                
                # Slice the data from the single large tensor - this is extremely fast
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                inputs = X_train[start_idx:end_idx]
                targets = Y_train[start_idx:end_idx]
                
                if config.use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                    optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for i in range(num_val_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    inputs = X_val[start_idx:end_idx]
                    targets = Y_val[start_idx:end_idx]
                    
                    if config.use_mixed_precision and scaler is not None:
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    epoch_val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = epoch_train_loss / num_train_batches
            avg_val_loss = epoch_val_loss / num_val_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics
            metrics.update(epoch + 1, avg_train_loss, avg_val_loss, current_lr)
            
            # We must synchronize the GPU before stopping the timer for an accurate measurement
            torch.cuda.synchronize()
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            logger.info(f"Epoch [{epoch+1:3d}/{config.max_epochs}]: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, "
                      f"LR: {current_lr:.2e}, "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if epoch > 100:
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"  -> New best model saved! Val Loss: {best_val_loss:.6f}")
                else:
                    logger.info(f"  -> New best model! Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info(f"  -> Early stopping triggered after {config.patience} epochs without improvement")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % config.save_checkpoint_every == 0:
                checkpoint_path = f'{model_name}_checkpoint_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"  -> Checkpoint saved to {checkpoint_path}")
        
        # Load the best model for final evaluation
        logger.info("Loading the best model state for final evaluation.")
        state_dict = torch.load(best_model_path, map_location=device)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        
        # Final evaluation on test set
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for i in range(num_test_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                inputs = X_test[start_idx:end_idx]
                targets = Y_test[start_idx:end_idx]
                
                if config.use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                test_loss += loss.item()
        
        final_test_loss = test_loss / num_test_batches
        logger.info(f"Final Test Loss: {final_test_loss:.6f}")
        
        # Log training summary
        metrics.log_summary(logger)
        
        return model, train_losses, val_losses, final_test_loss
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model...")
        torch.save(model.state_dict(), f'{model_name}_interrupted.pth')
        return model, train_losses, val_losses, None
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e

# --- Utility Functions ---
def extract_final_predictions(model_output, output_dim):
    """
    Extract final predictions from 3*output_dim model output.
    """
    batch_size = model_output.shape[0]
    
    # Reshape to separate p1, p2, p3 for each output dimension
    pred_reshaped = model_output.view(batch_size, output_dim, 3)
    
    # Extract p1, p2, p3
    p1 = pred_reshaped[:, :, 0]  # weight parameters
    p2 = pred_reshaped[:, :, 1]  # negative predictions
    p3 = pred_reshaped[:, :, 2]  # positive predictions
    
    # Calculate weights (clamp p1 to ensure weights sum to 1)
    p1_clamped = torch.clamp(p1, -1.0, 1.0)
    weight_p2 = torch.abs(p1_clamped - 1) / 2
    weight_p3 = torch.abs(p1_clamped + 1) / 2
    
    # Weighted combination of p2 and p3
    final_predictions = weight_p2 * p2 + weight_p3 * p3
    
    return final_predictions

def plot_losses(train_losses, val_losses, train_last_ts, val_last_ts, test_last_ts, model_name="optimized_model"):
    """
    Enhanced plotting with better visualization.
    """
    # Remove first 100 elements if length is bigger than 200
    if len(train_losses) > 200:
        train_losses = train_losses[100:]
    if len(val_losses) > 200:
        val_losses = val_losses[100:]
    
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

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Optimized PyTorch Training")
    print("=" * 60)
    
    # Load and prepare data
    X, Y, T = load_and_preprocess_data()
    
    # Determine dimensions
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Create configuration
    config = OptimizedTrainingConfig()
    config.update(
        hidden_size=512,
        num_layers=4,
        dropout_rate=0.1,
        batch_size=128,  # Optimized batch size
        use_mixed_precision=True
    )
    
    print(f"Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Original output dimension: {output_dim}")
    print(f"  Model output dimension: {3 * output_dim} (3 values per output: p1, p2, p3)")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Dropout rate: {config.dropout_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    
    # Setup GPU
    device = setup_gpu()
    
    # Prepare optimized data
    data_dict = prepare_optimized_data(
        X, Y, 
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        device=device
    )
    data_dict['T'] = T
    
    # Train the model
    model, train_losses, val_losses, test_loss = train_model_optimized(
        data_dict, input_dim, output_dim,
        model_name="optimized_model",
        config=config
    )
    
    # Save configuration
    save_training_config(config, "optimized_model")
    
    # Plot results
    plot_losses(train_losses, val_losses, data_dict['train_last_ts'], data_dict['val_last_ts'], data_dict['test_last_ts'])

    print("\nOptimized training complete!")
    print("Files saved:")
    print("  - optimized_model_best_model.pth (best model)")
    print("  - optimized_model_config.json (configuration)")
    print("  - optimized_model_training.log (training log)")
    print("  - optimized_model_loss_analysis.png (loss curves)")
    if test_loss is not None:
        print(f"  - Final test loss: {test_loss:.6f}")
    
    # Clear GPU memory
    clear_gpu_memory() 