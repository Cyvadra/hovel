import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path='training_data.h5'):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)

        if X.shape[0] != Y.shape[0]:
            X = X.T
            Y = Y.T

        Y = np.sign(Y) * np.log1p(np.abs(Y))

        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
    return X, Y

# --- 2. Model Definition ---
class TinyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.SiLU(),
            nn.Linear(8, 24),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 24),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 12),
            nn.Softmax(),
            nn.Linear(12, output_dim),
        )
        
    def forward(self, x):
        return self.net(x)

# --- 3. Training Setup ---
def setup_training(X, Y, batch_size=32):
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    
    # Create dataset and split
    dataset = TensorDataset(X_tensor, Y_tensor)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    return train_loader, val_loader

# --- 4. Training Function ---
def train_model(train_loader, val_loader, input_dim, output_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TinyModel(input_dim, output_dim)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=2e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=20
    )
    criterion = nn.MSELoss()
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(300):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        
        # Calculate epoch losses
        epoch_train_loss /= len(train_loader.dataset)
        epoch_val_loss /= len(val_loader.dataset)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{300}: "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

# --- 5. Plotting Functions ---
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

def plot_predictions(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            all_outputs.append(outputs)
            all_targets.append(targets.numpy())
    
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    plt.figure(figsize=(14, 10))
    for i in range(min(5, all_targets.shape[1])):
        plt.subplot(5, 1, i+1)
        plt.plot(all_targets[:, i], 'b-', label='Actual')
        plt.plot(all_outputs[:, i], 'r--', label='Predicted', alpha=0.7)
        plt.ylabel(f'Y_{i+1}')
        plt.legend()
    plt.xlabel('Samples')
    plt.suptitle('Validation Set: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Load and prepare data
    X, Y = load_and_preprocess_data()
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Setup data loaders
    train_loader, val_loader = setup_training(X, Y, batch_size=32)
    
    # Train model
    model, train_losses, val_losses = train_model(
        train_loader, val_loader, input_dim, output_dim
    )
    
    # Plot results
    plot_losses(train_losses, val_losses)
    plot_predictions(model, val_loader)