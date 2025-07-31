import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    with h5py.File(file_path, 'r') as f:
        # Load X and Y, ensuring float32 type
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)

        # Transpose if the first dimension is smaller than the second,
        # assuming samples are along the first dimension.
        if X.shape[0] != Y.shape[0]:
            print("Warning: X and Y have different number of samples. Attempting transpose.")
            X = X.T
            Y = Y.T
            print(f"Transposed X shape: {X.shape}, Y shape: {Y.shape}")

        # Apply signed log1p transformation to Y
        # This is useful for targets that span a wide range and include negative values.
        Y = np.sign(Y) * np.log1p(np.abs(Y))

        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
    return X, Y

# --- 2. Model Definition ---
class TinyModel(nn.Module):
    """
    A simple feed-forward neural network with SiLU activation, Dropout, and Softmax.
    Designed for regression tasks, but includes Softmax which is typically for classification.
    Consider removing Softmax if this is purely a regression problem.
    """
    def __init__(self, input_dim, output_dim, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(),
            nn.Linear(hidden_size, output_dim),
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the network.
        """
        return self.net(x)

# --- 3. Training Setup ---
def setup_training(X, Y, batch_size=32):
    """
    Prepares PyTorch DataLoaders for training and validation.

    Args:
        X (np.ndarray): Input features.
        Y (np.ndarray): Target values.
        batch_size (int): Size of batches for data loaders.

    Returns:
        tuple: train_loader and val_loader (PyTorch DataLoader objects).
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    
    # Create a TensorDataset from the tensors
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    # Split the dataset into training and validation sets (90% train, 10% validation)
    # Use sequential split instead of random split - take first 90% for training, last 10% for validation
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    # Create sequential splits
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, total_size))
    
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders for training and validation
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # Shuffle training data
    val_loader = DataLoader(val_set, batch_size=batch_size) # No need to shuffle validation data
    
    return train_loader, val_loader

# --- 4. Training Function ---
def train_model(train_loader, val_loader, input_dim, output_dim, hidden_size=512, model_name="model"):
    """
    Trains the TinyModel, handles multi-GPU, early stopping, and learning rate scheduling.
    Now loads existing model if 'best_model.pth' exists and saves checkpoints.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output targets.
        hidden_size (int): Size of hidden layers in the model.
        model_name (str): Name prefix for saving model files and plots.

    Returns:
        tuple: Trained model, list of training losses, list of validation losses.
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TinyModel(input_dim, output_dim, hidden_size)
    
    # Create model-specific file names
    best_model_path = f'{model_name}_best_model.pth'
    
    # --- MODIFICATION 1: Load existing model if best_model.pth exists ---
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

    # Multi-GPU setup using DataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device) # Move model to the selected device
    
    # Optimizer: AdamW with a small learning rate and weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    # Learning Rate Scheduler: Reduces learning rate when validation loss stops improving
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', # Monitor minimum validation loss
        factor=0.5, # Reduce LR by a factor of 0.5
        patience=50 # Wait for 20 epochs before reducing LR
    )
    
    criterion = nn.MSELoss() # Mean Squared Error Loss for regression
    
    # Training parameters
    best_val_loss = float('inf') # Initialize with infinity to ensure first loss is better
    last_save_epoch = -10 # Track when we last saved to enforce minimum interval
    
    train_losses = [] # To store training loss for each epoch
    val_losses = [] # To store validation loss for each epoch
    
    # --- MODIFICATION: Train for exactly 60 epochs ---
    num_epochs = 60
    for epoch in range(num_epochs):
        # Training phase
        model.train() # Set model to training mode
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device) # Move data to device
            
            optimizer.zero_grad() # Clear previous gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters
            
            epoch_train_loss += loss.item() * inputs.size(0) # Accumulate batch loss
        
        # Validation phase
        model.eval() # Set model to evaluation mode
        epoch_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        
        # Calculate average epoch losses
        epoch_train_loss /= len(train_loader.dataset)
        epoch_val_loss /= len(val_loader.dataset)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_val_loss:.6f}")
        
        # Learning rate scheduling step
        scheduler.step(epoch_val_loss)
        
        # Check for best validation loss and save with minimum interval
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Only save if at least 10 epochs have passed since last save
            if epoch - last_save_epoch >= 10:
                torch.save(model.state_dict(), best_model_path)
                last_save_epoch = epoch
                print(f"Saved new best model with validation loss: {best_val_loss:.6f} (epoch {epoch+1})")
            else:
                print(f"New best validation loss: {best_val_loss:.6f} (epoch {epoch+1}) - skipping save (min interval: 10 epochs)")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'{model_name}_checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    # Load the best model saved during training before returning
    print("Loading the best model state for final evaluation and plotting.")
    new_state_dict = load_state_dict_safely(best_model_path, device)
    model.load_state_dict(new_state_dict)
    return model, train_losses, val_losses

# --- 5. Plotting Functions ---
def plot_losses(train_losses, val_losses, model_name="model"):
    """
    Plots the training and validation loss curves.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        model_name (str): Name prefix for saving the plot file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss_curve.png') # Save the plot to a file
    plt.close() # Close the plot to free memory

def plot_predictions(model, val_loader, stride=10, model_name="model"):
    """
    Plots actual vs. predicted values for a subset of validation data with reduced density.

    Args:
        model (nn.Module): Trained PyTorch model.
        val_loader (DataLoader): DataLoader for validation data.
        stride (int): Step size for downsampling the data points (default: 10).
        model_name (str): Name prefix for saving the plot file.
    """
    # Get the device the model is currently on
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)  # Move inputs to device
            outputs = model(inputs).cpu().numpy()  # Get predictions and move to CPU as numpy array
            all_outputs.append(outputs)
            all_targets.append(targets.numpy())
    
    # Concatenate all collected targets and outputs
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # Downsample the data using the specified stride
    indices = np.arange(0, len(all_targets), stride)
    downsampled_targets = all_targets[indices]
    downsampled_outputs = all_outputs[indices]
    
    plt.figure(figsize=(14, 10))
    # Plot predictions for up to 5 output dimensions
    for i in range(min(5, all_targets.shape[1])):
        plt.subplot(min(5, all_targets.shape[1]), 1, i+1)  # Create subplots
        plt.plot(downsampled_targets[:, i], 'b-', label='Actual')  # Actual values in blue solid line
        plt.plot(downsampled_outputs[:, i], 'r--', label='Predicted', alpha=0.7)  # Predicted values in red dashed line
        plt.ylabel(f'Y_{i+1}')
        plt.legend()
    plt.xlabel(f'Samples (downsampled by {stride}x)')
    plt.suptitle(f'Validation Set: Actual vs Predicted (Downsampled) - {model_name}')  # Main title for the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.savefig(f'{model_name}_predictions.png')  # Save the plot
    plt.close()  # Close the plot

# --- Main Execution ---
if __name__ == "__main__":
    # Load and prepare data
    # Ensure 'training_data.h5' exists or is created by the load_and_preprocess_data function
    X, Y = load_and_preprocess_data()
    
    # Determine input and output dimensions from the loaded data
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Setup data loaders for training and validation
    train_loader, val_loader = setup_training(X, Y, batch_size=32)
    
    # Train the model (or continue training if best_model.pth exists)
    model, train_losses, val_losses = train_model(
        train_loader, val_loader, input_dim, output_dim
    )
    
    # Plot results after training completes
    plot_losses(train_losses, val_losses)
    plot_predictions(model, val_loader)

    print("Training complete. Loss curve saved to 'loss_curve.png' and predictions plot to 'predictions.png'.")
    print("Best model saved to 'best_model.pth'. Checkpoints saved every 100 epochs.")
