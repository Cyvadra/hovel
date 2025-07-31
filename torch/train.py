import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.nn import functional as F
import warnings
from torch.cuda.amp import GradScaler, autocast
import logging
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

# --- Logging Setup ---
def setup_logging(model_name="improved_model"):
    """Setup logging configuration."""
    # Create a logger specific to this model
    logger = logging.getLogger(f"training_{model_name}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(f'{model_name}_training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
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
class TrainingConfig:
    """Centralized configuration for training parameters."""
    
    def __init__(self):
        # Model parameters
        self.hidden_size = 512
        self.num_layers = 4
        self.dropout_rate = 0.1
        
        # Training parameters
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.max_epochs = 300
        self.patience = 30
        self.gradient_clip_norm = 1.0
        
        # Data parameters
        self.val_split = 0.1
        self.test_split = 0.1
        self.num_workers = 4
        
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

# --- 1. Data Loading and Preprocessing ---
def clear_gpu_memory():
    """Clear GPU memory and garbage collect."""
    if torch.cuda.is_available():
        # Show memory usage before clearing
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory before clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Clear memory cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Show memory usage after clearing
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory after clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    # Force garbage collection
    import gc
    gc.collect()

def show_gpu_memory_usage():
    """Show current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
    else:
        print("CUDA not available")

def load_state_dict_safely(file_path, device):
    """
    Load state dict safely, handling both DataParallel and non-DataParallel saved models.
    
    Args:
        file_path (str): Path to the saved model file.
        device (torch.device): Device to load the model on.
        
    Returns:
        dict: Cleaned state dict ready for loading.
    """
    state_dict = torch.load(file_path, map_location=device)
    
    # Handle DataParallel saved models (remove 'module.' prefix)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict

def load_and_preprocess_data(file_path='training_data.h5'):
    """
    Load and preprocess data with improved normalization and validation.
    """
    with h5py.File(file_path, 'r') as f:
        # Load X and Y, ensuring float32 type
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)

        # Transpose if the first dimension is smaller than the second
        if X.shape[0] != Y.shape[0]:
            print("Warning: X and Y have different number of samples. Attempting transpose.")
            X = X.T
            Y = Y.T
            print(f"Transposed X shape: {X.shape}, Y shape: {Y.shape}")

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
        print(f"X stats - mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")
        print(f"Y stats - mean: {np.mean(Y):.4f}, std: {np.std(Y):.4f}")
        
    return X, Y

# --- 2. Improved Model Definition ---
class ImprovedModel(nn.Module):
    """
    An improved neural network with modern best practices:
    - Residual connections
    - Layer normalization
    - Dropout for regularization
    - GELU activation (better than SiLU for deep networks)
    - Proper initialization
    - Output format: 3*output_dim (p1, p2, p3 for each output dimension)
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
                         For each output dimension i, we have:
                         - p1_i: weight parameter ∈ [-1,1]
                         - p2_i: negative prediction
                         - p3_i: positive prediction
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

# --- 3. Improved Training Setup ---
def setup_training(X, Y, batch_size=32, val_split=0.1, test_split=0.1, num_workers=4):
    """
    Prepares PyTorch DataLoaders with proper train/val/test split.

    Args:
        X (np.ndarray): Input features.
        Y (np.ndarray): Target values.
        batch_size (int): Size of batches for data loaders.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Convert numpy arrays to PyTorch tensors, ensuring float32 type
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float()
    
    # Create a TensorDataset from the tensors
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size
    
    # Split the dataset
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders with improved settings
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"Data loading with {num_workers} workers")
    
    return train_loader, val_loader, test_loader

# --- 4. Improved Loss Functions ---
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

class CombinedLoss(nn.Module):
    """
    Combined loss function using MSE and Huber loss.
    """
    def __init__(self, mse_weight=0.7, huber_weight=0.3, huber_delta=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.mse_loss = nn.MSELoss()
        self.huber_loss = HuberLoss(delta=huber_delta)
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        huber = self.huber_loss(pred, target)
        return self.mse_weight * mse + self.huber_weight * huber

class WeightedCombinedLoss(nn.Module):
    """
    Weighted combined loss function for 3*output_dim model outputs.
    
    For each output dimension i:
    - p1_i: weight parameter ∈ [-1,1] (controls which prediction to trust more)
    - p2_i: negative prediction (should be < 0)
    - p3_i: positive prediction (should be > 0)
    
    Loss formula: abs(p1-1)/2 * ori_loss(p2,y) + abs(p1+1)/2 * ori_loss(p3,y)
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
        # pred shape: (batch_size, 3*output_dim) -> (batch_size, output_dim, 3)
        pred_reshaped = pred.view(batch_size, self.output_dim, 3)
        
        # Extract p1, p2, p3 for each output dimension
        p1 = pred_reshaped[:, :, 0]  # weight parameters ∈ [-1,1]
        p2 = pred_reshaped[:, :, 1]  # negative predictions
        p3 = pred_reshaped[:, :, 2]  # positive predictions
        
        # Calculate weights based on p1
        # Clamp p1 to [-1, 1] range to ensure weights sum to 1
        p1_clamped = torch.clamp(p1, -1.0, 1.0)
        
        # When p1 < 0: weight_p2 = abs(p1-1)/2 = (1-p1)/2 (higher weight for p2)
        # When p1 > 0: weight_p3 = abs(p1+1)/2 = (1+p1)/2 (higher weight for p3)
        weight_p2 = torch.abs(p1_clamped - 1) / 2  # weight for p2 (negative prediction)
        weight_p3 = torch.abs(p1_clamped + 1) / 2  # weight for p3 (positive prediction)
        
        # Calculate individual losses for p2 and p3
        loss_p2 = self.mse_loss(p2, target)  # shape: (batch_size, output_dim)
        loss_p3 = self.mse_loss(p3, target)  # shape: (batch_size, output_dim)
        
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

# --- 5. Improved Training Function ---
def train_model(train_loader, val_loader, test_loader, input_dim, output_dim, 
                hidden_size=512, num_layers=4, dropout_rate=0.1, model_name="improved_model",
                enable_early_stop=True, fixed_epochs=300, config=None):
    """
    Improved training function with modern best practices:
    - Better learning rate scheduling
    - Early stopping with patience (optional)
    - Gradient clipping
    - Mixed precision training
    - Better monitoring and logging
    - Configuration management
    
    Args:
        enable_early_stop (bool): If True, use early stopping logic. If False, train for fixed_epochs.
        fixed_epochs (int): Number of epochs to train when early stopping is disabled.
        config (TrainingConfig): Configuration object. If None, uses default config.
    """
    # Use provided config or create default
    if config is None:
        config = TrainingConfig()
        config.update(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            max_epochs=fixed_epochs if not enable_early_stop else 300
        )
    
    # Setup logging
    logger = setup_logging(model_name)
    metrics = TrainingMetrics()
    
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    show_gpu_memory_usage()
    
    # Create model
    model = ImprovedModel(input_dim, output_dim, config.hidden_size, config.num_layers, config.dropout_rate)
    
    # Create model-specific file names
    best_model_path = f'{model_name}_best_model.pth'
    
    # Load existing model if available
    if os.path.exists(best_model_path):
        logger.info(f"Found '{best_model_path}'. Loading pre-trained model state.")
        try:
            new_state_dict = load_state_dict_safely(best_model_path, device)
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
    
    # Improved optimizer: AdamW with better hyperparameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Improved learning rate scheduler: Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_t_mult,
        eta_min=config.scheduler_eta_min
    )
    
    # Use weighted combined loss function for 3*output_dim model
    criterion = WeightedCombinedLoss(
        output_dim, 
        mse_weight=config.mse_weight, 
        huber_weight=config.huber_weight, 
        huber_delta=config.huber_delta
    )
    
    # Mixed precision setup
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Training parameters
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    num_epochs = config.max_epochs
    if enable_early_stop:
        logger.info(f"Training with early stopping enabled (max {num_epochs} epochs, patience: {config.patience})")
    else:
        logger.info(f"Training for fixed {num_epochs} epochs (early stopping disabled)")
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                try:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    
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
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"GPU OOM at batch {batch_idx}. Clearing memory and skipping batch.")
                        clear_gpu_memory()
                        continue
                    else:
                        raise e
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    if config.use_mixed_precision and scaler is not None:
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    epoch_val_loss += loss.item()
                    val_batches += 1
            
            # Calculate average losses
            avg_train_loss = epoch_train_loss / num_batches
            avg_val_loss = epoch_val_loss / val_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics
            metrics.update(epoch + 1, avg_train_loss, avg_val_loss, current_lr)
            
            logger.info(f"Epoch {epoch+1:3d}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  -> New best model saved! Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if enable_early_stop and patience_counter >= config.patience:
                    logger.info(f"  -> Early stopping triggered after {config.patience} epochs without improvement")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % config.save_checkpoint_every == 0:
                checkpoint_path = f'{model_name}_checkpoint_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"  -> Checkpoint saved to {checkpoint_path}")
        
        # Load the best model for final evaluation
        logger.info("Loading the best model state for final evaluation.")
        new_state_dict = load_state_dict_safely(best_model_path, device)
        model.load_state_dict(new_state_dict)
        
        # Final evaluation on test set
        model.eval()
        test_loss = 0
        test_batches = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if config.use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                test_batches += 1
        
        final_test_loss = test_loss / test_batches
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

# --- 6. Improved Plotting Functions ---
def check_existing_plots(model_name):
    """Check if plots for a specific model already exist."""
    plot_files = [
        f"{model_name}_loss_analysis.png",
        f"{model_name}_predictions.png"
    ]
    
    all_exist = all(os.path.exists(f) for f in plot_files)
    if all_exist:
        print(f"Plots for {model_name} already exist, skipping...")
    return all_exist

def check_existing_model_files(model_name):
    """Check if model files for a specific model already exist."""
    model_file = f"{model_name}_best_model.pth"
    exists = os.path.exists(model_file)
    if exists:
        print(f"Model file for {model_name} already exists, skipping...")
    return exists

def save_training_config(config, model_name):
    """Save training configuration to a JSON file."""
    import json
    
    config_dict = {key: value for key, value in config.__dict__.items() 
                   if not key.startswith('_')}
    
    config_file = f"{model_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {config_file}")

def load_training_config(model_name):
    """Load training configuration from a JSON file."""
    import json
    
    config_file = f"{model_name}_config.json"
    if not os.path.exists(config_file):
        return None
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    config = TrainingConfig()
    config.update(**config_dict)
    return config

def extract_final_predictions(model_output, output_dim):
    """
    Extract final predictions from 3*output_dim model output.
    
    Args:
        model_output (torch.Tensor): Model output of shape (batch_size, 3*output_dim)
        output_dim (int): Original output dimension
    
    Returns:
        torch.Tensor: Final predictions of shape (batch_size, output_dim)
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

def plot_losses(train_losses, val_losses, model_name="improved_model"):
    """
    Enhanced plotting with better visualization.
    """
    plt.figure(figsize=(12, 8))
    
    # Main loss plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot
    plt.subplot(2, 2, 2)
    plt.semilogy(train_losses, label='Training Loss', alpha=0.8)
    plt.semilogy(val_losses, label='Validation Loss', alpha=0.8)
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
    plt.plot(val_losses, label='Validation Loss', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss with Moving Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(model, val_loader, output_dim, stride=10, model_name="improved_model"):
    """
    Enhanced prediction plotting with confidence intervals and statistics.
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            raw_outputs = model(inputs)
            # Extract final predictions from 3*output_dim output
            final_outputs = extract_final_predictions(raw_outputs, output_dim).cpu().numpy()
            all_outputs.append(final_outputs)
            all_targets.append(targets.numpy())
    
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # Downsample for visualization
    indices = np.arange(0, len(all_targets), stride)
    downsampled_targets = all_targets[indices]
    downsampled_outputs = all_outputs[indices]
    
    # Calculate metrics
    mse = np.mean((all_targets - all_outputs) ** 2, axis=0)
    mae = np.mean(np.abs(all_targets - all_outputs), axis=0)
    r2_scores = []
    for i in range(all_targets.shape[1]):
        ss_res = np.sum((all_targets[:, i] - all_outputs[:, i]) ** 2)
        ss_tot = np.sum((all_targets[:, i] - np.mean(all_targets[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        r2_scores.append(r2)
    
    plt.figure(figsize=(16, 12))
    
    # Plot predictions for up to 6 output dimensions
    num_outputs = min(6, all_targets.shape[1])
    for i in range(num_outputs):
        plt.subplot(num_outputs, 2, 2*i + 1)
        plt.plot(downsampled_targets[:, i], 'b-', label='Actual', alpha=0.8, linewidth=1)
        plt.plot(downsampled_outputs[:, i], 'r--', label='Predicted', alpha=0.8, linewidth=1)
        plt.ylabel(f'Y_{i+1}')
        plt.title(f'Output {i+1} - MSE: {mse[i]:.4f}, MAE: {mae[i]:.4f}, R²: {r2_scores[i]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(num_outputs, 2, 2*i + 2)
        plt.scatter(all_targets[:, i], all_outputs[:, i], alpha=0.5, s=1)
        plt.plot([all_targets[:, i].min(), all_targets[:, i].max()], 
                [all_targets[:, i].min(), all_targets[:, i].max()], 'r--', alpha=0.8)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Scatter Plot - Output {i+1}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nPrediction Summary:")
    print(f"Average MSE: {np.mean(mse):.6f}")
    print(f"Average MAE: {np.mean(mae):.6f}")
    print(f"Average R²: {np.mean(r2_scores):.6f}")

# --- Main Execution ---
if __name__ == "__main__":
    # Load and prepare data
    X, Y = load_and_preprocess_data()
    
    # Determine dimensions
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Create configuration
    config = TrainingConfig()
    config.update(
        hidden_size=512,
        num_layers=4,
        dropout_rate=0.1,
        batch_size=64,
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
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_training(
        X, Y, 
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        num_workers=config.num_workers
    )
    
    # Train the model
    model, train_losses, val_losses, test_loss = train_model(
        train_loader, val_loader, test_loader, input_dim, output_dim,
        model_name="improved_model",
        enable_early_stop=True,
        config=config
    )
    
    # Save configuration
    save_training_config(config, "improved_model")
    
    # Plot results
    plot_losses(train_losses, val_losses)
    plot_predictions(model, val_loader, output_dim)

    print("\nTraining complete!")
    print("Files saved:")
    print("  - improved_model_best_model.pth (best model)")
    print("  - improved_model_config.json (configuration)")
    print("  - improved_model_training.log (training log)")
    print("  - improved_model_loss_analysis.png (loss curves)")
    print("  - improved_model_predictions.png (predictions)")
    if test_loss is not None:
        print(f"  - Final test loss: {test_loss:.6f}")
    
    # Clear GPU memory
    clear_gpu_memory()
