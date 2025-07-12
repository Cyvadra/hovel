import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

# --- 1. Data Loading and Preprocessing (Adjusted for PyTorch) ---
def load_and_preprocess_data(file_path='training_data.h5'):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32) # Use float32 for PyTorch default
        Y = np.array(f['Y'][:], dtype=np.float32)

        if X.shape[0] != Y.shape[0]:
            X = X.T
            Y = Y.T

        # Equivalent of tf.clip_by_value in numpy/pytorch
        Y = np.clip(Y, a_min=-100.0, a_max=100.0)

        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
    return X, Y

# --- 2. Define the DNN Model ---
class Swish(nn.Module):
    '''
    Swish activation function. PyTorch also has nn.SiLU which is Swish.
    '''
    def forward(self, x):
        return x * torch.sigmoid(x)

class TinyDNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TinyDNN, self).__init__()
        # Extremely small model to prevent overfitting on limited samples
        self.fc1 = nn.Linear(input_size, 16) # Small hidden layer
        self.swish1 = Swish() # or nn.SiLU()
        self.dropout = nn.Dropout(0.2) # Dropout for regularization
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swish1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 3. Training Function ---
def train(rank, world_size, X_train, Y_train, X_val, Y_val, input_size, output_size, num_epochs=100, learning_rate=1e-5):
    print(f"Running DDP on rank {rank}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train).to(device), torch.tensor(Y_train).to(device))
    val_dataset = TensorDataset(torch.tensor(X_val).to(device), torch.tensor(Y_val).to(device))

    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Batch size needs to be adjusted per GPU, not global
    batch_size_per_gpu = 16 # Adjust based on your memory and dataset size
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler)

    # Initialize model
    model = TinyDNN(input_size, output_size).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Loss function and optimizer
    criterion = nn.MSELoss() # Common for regression
    # AdamW is an advanced optimizer, good for tiny learning rates
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) # L2 regularization

    # Learning rate scheduler: ReduceLROnPlateau
    # Reduces learning rate when a metric (e.g., validation loss) has stopped improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20 # Stop if validation loss doesn't improve for 20 epochs

    if rank == 0:
        print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch) # Important for DDP with shuffle=True
        running_loss = 0.0
        for i, (X_batch, Y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch_val, Y_batch_val in val_loader:
                outputs_val = model(X_batch_val)
                loss_val = criterion(outputs_val, Y_batch_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping check (only on rank 0)
        if rank == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.module.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                    break
    if rank == 0:
        print("Training finished.")
    dist.destroy_process_group() # Clean up DDP

# --- 4. Prediction and Plotting Function ---
def plot_predictions(model, X_val, Y_val, device, fig_path="prediction_plot.png"):
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        predictions = model(X_val_tensor).cpu().numpy()

    # Ensure Y_val is on CPU for plotting if it was passed as a tensor on GPU
    if isinstance(Y_val, torch.Tensor):
        Y_val = Y_val.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(Y_val, label='Actual Y Series', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted Y Series', color='red', linestyle='--', alpha=0.7)
    plt.title('Actual vs. Predicted Y Series on Validation Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Y Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)
    plt.close()
    print(f"Prediction plot saved to {fig_path}")

# --- Main execution block for DDP ---
def main_worker(rank, world_size, X, Y):
    # Split data into training and validation sets
    # For time series, often a chronological split is better
    # For simplicity, we'll use a random split here. If 'Y series' implies
    # a time-dependent sequence where future values depend on past,
    # then a chronological split should be used.
    # Given the description, it sounds like a regression task for each (X, Y) pair.
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    input_size = X_train.shape[1] if X_train.ndim > 1 else 1 # Handle 1D or 2D X
    output_size = Y_train.shape[1] if Y_train.ndim > 1 else 1 # Handle 1D or 2D Y

    train(rank, world_size, X_train, Y_train, X_val, Y_val, input_size, output_size)

    # After training, rank 0 will load the best model and plot
    if rank == 0:
        print("\nPlotting predictions...")
        # Load the best model saved by rank 0
        final_model = TinyDNN(input_size, output_size)
        final_model.load_state_dict(torch.load("best_model.pth"))
        final_model.to(f"cuda:0") # Move to GPU 0 for inference

        plot_predictions(final_model, X_val, Y_val, device=f"cuda:0")


if __name__ == '__main__':
    # Ensure all necessary environment variables are set for DDP
    # This part should be run using `torchrun` or `torch.multiprocessing.spawn`
    # Example for torchrun (recommended):
    # torchrun --nproc_per_node=4 your_script_name.py

    # Load data once
    X_data, Y_data = load_and_preprocess_data('training_data.h5')

    # Ensure Y is 2D if it's currently 1D, for consistent model output
    if Y_data.ndim == 1:
        Y_data = Y_data.reshape(-1, 1)

    world_size = 4 # Assuming 4 GPUs as per the request

    # Using torch.multiprocessing.spawn for demonstration if torchrun is not used
    # For production, torchrun is generally preferred as it handles process management
    # and environment variables more robustly.
    # To run this script: python -m torch.distributed.launch --nproc_per_node=4 your_script_name.py
    # or if you prefer torch.multiprocessing.spawn:
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # Choose a free port
        torch.multiprocessing.spawn(main_worker, args=(world_size, X_data, Y_data), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Error during DDP spawn: {e}")
        print("Please ensure your dataset 'training_data.h5' exists and contains 'X' and 'Y' datasets.")
        print("You might also want to run this script with `torchrun --nproc_per_node=4 your_script_name.py` for proper multi-GPU setup.")

    # Clean up the saved model after plotting
    if os.path.exists("best_model.pth"):
        os.remove("best_model.pth")
        print("Cleaned up best_model.pth")