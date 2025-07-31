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
warnings.filterwarnings('ignore')

# --- 1. Data Loading and Preprocessing ---
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
        Y_median = np.median(Y, axis=0, keepdims=True)
        Y_q75, Y_q25 = np.percentile(Y, [75, 25], axis=0, keepdims=True)
        Y_iqr = Y_q75 - Y_q25
        Y_iqr = np.where(Y_iqr == 0, 1.0, Y_iqr)  # Avoid division by zero
        Y = (Y - Y_median) / Y_iqr

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
def setup_training(X, Y, batch_size=32, val_split=0.1, test_split=0.1):
    """
    Prepares PyTorch DataLoaders with proper train/val/test split.

    Args:
        X (np.ndarray): Input features.
        Y (np.ndarray): Target values.
        batch_size (int): Size of batches for data loaders.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    
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
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
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
        # When p1 < 0: weight_p2 = abs(p1-1)/2 = (1-p1)/2 (higher weight for p2)
        # When p1 > 0: weight_p3 = abs(p1+1)/2 = (1+p1)/2 (higher weight for p3)
        weight_p2 = torch.abs(p1 - 1) / 2  # weight for p2 (negative prediction)
        weight_p3 = torch.abs(p1 + 1) / 2  # weight for p3 (positive prediction)
        
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
                hidden_size=512, num_layers=4, dropout_rate=0.1, model_name="improved_model"):
    """
    Improved training function with modern best practices:
    - Better learning rate scheduling
    - Early stopping with patience
    - Gradient clipping
    - Mixed precision training
    - Better monitoring and logging
    """
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = ImprovedModel(input_dim, output_dim, hidden_size, num_layers, dropout_rate)
    
    # Create model-specific file names
    best_model_path = f'{model_name}_best_model.pth'
    
    # Load existing model if available
    if os.path.exists(best_model_path):
        print(f"Found '{best_model_path}'. Loading pre-trained model state.")
        try:
            new_state_dict = load_state_dict_safely(best_model_path, device)
            model.load_state_dict(new_state_dict)
            print("Successfully loaded pre-trained model state.")
        except RuntimeError as e:
            print(f"Warning: Could not load existing model state: {e}")
            print("Starting training from scratch.")
    else:
        print(f"No '{best_model_path}' found. Starting training from scratch.")

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Improved optimizer: AdamW with better hyperparameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-3,  # Higher initial learning rate
        weight_decay=1e-4,  # Stronger regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Improved learning rate scheduler: Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # Initial restart period
        T_mult=2,  # Multiply period by 2 after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Use weighted combined loss function for 3*output_dim model
    criterion = WeightedCombinedLoss(output_dim, mse_weight=0.7, huber_weight=0.3, huber_delta=1.0)
    
    # Training parameters
    best_val_loss = float('inf')
    patience = 30  # Early stopping patience
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    num_epochs = 200  # More epochs with early stopping
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
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
        
        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"LR: {current_lr:.2e}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f'{model_name}_checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Checkpoint saved to {checkpoint_path}")
    
    # Load the best model for final evaluation
    print("Loading the best model state for final evaluation.")
    new_state_dict = load_state_dict_safely(best_model_path, device)
    model.load_state_dict(new_state_dict)
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0
    test_batches = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_batches += 1
    
    final_test_loss = test_loss / test_batches
    print(f"Final Test Loss: {final_test_loss:.6f}")
    
    return model, train_losses, val_losses, final_test_loss

# --- 6. Improved Plotting Functions ---
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
    
    # Calculate weights
    weight_p2 = torch.abs(p1 - 1) / 2
    weight_p3 = torch.abs(p1 + 1) / 2
    
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
    
    print(f"Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Original output dimension: {output_dim}")
    print(f"  Model output dimension: {3 * output_dim} (3 values per output: p1, p2, p3)")
    print(f"  Hidden size: 512")
    print(f"  Number of layers: 4")
    print(f"  Dropout rate: 0.1")
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_training(X, Y, batch_size=64)
    
    # Train the model
    model, train_losses, val_losses, test_loss = train_model(
        train_loader, val_loader, test_loader, input_dim, output_dim,
        hidden_size=512, num_layers=4, dropout_rate=0.1
    )
    
    # Plot results
    plot_losses(train_losses, val_losses)
    plot_predictions(model, val_loader, output_dim)

    print("\nTraining complete!")
    print("Files saved:")
    print("  - improved_model_best_model.pth (best model)")
    print("  - improved_model_loss_analysis.png (loss curves)")
    print("  - improved_model_predictions.png (predictions)")
    print(f"  - Final test loss: {test_loss:.6f}")
