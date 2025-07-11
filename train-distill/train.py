import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import warnings
import h5py
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataPreprocessor:
    """Handles data preprocessing including cleaning, clipping, and scaling"""
    
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
    def clean_and_clip_data(self, X, Y):
        """Clean and clip data to remove outliers and ensure Y is within [-100, 100]"""
        print("Cleaning and clipping data...")
        
        # Clip Y values to [-100, 100]
        Y_clipped = np.clip(Y, -100, 100)
        
        # Remove rows with NaN or infinite values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | 
                      np.isnan(Y_clipped).any(axis=1) | np.isinf(Y_clipped).any(axis=1))
        
        X_clean = X[valid_mask]
        Y_clean = Y_clipped[valid_mask]
        
        print(f"Removed {len(X) - len(X_clean)} rows with invalid data")
        print(f"Final data shape: X={X_clean.shape}, Y={Y_clean.shape}")
        
        return X_clean, Y_clean
    
    def scale_data(self, X, Y, fit=True):
        """Scale features and targets"""
        if fit:
            X_scaled = self.scaler_X.fit_transform(X)
            Y_scaled = self.scaler_Y.fit_transform(Y)
        else:
            X_scaled = self.scaler_X.transform(X)
            Y_scaled = self.scaler_Y.transform(Y)
        
        return X_scaled, Y_scaled
    
    def inverse_transform_Y(self, Y_scaled):
        """Inverse transform scaled Y back to original scale"""
        return self.scaler_Y.inverse_transform(Y_scaled)

class TeacherModel:
    """Large teacher model with advanced architecture"""
    
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        
    def build_model(self):
        """Build a large and deep teacher model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # First layer - wide
        x = layers.Dense(512, activation='swish')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Multiple hidden layers with decreasing width
        layer_sizes = [256, 128, 64, 32]
        for size in layer_sizes:
            x = layers.Dense(size, activation='swish')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.output_shape, activation='linear')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """Compile the teacher model with safe training settings"""
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        print("Teacher model compiled successfully")
        print(f"Total parameters: {self.model.count_params():,}")

class StudentModel:
    """Small student model for distillation"""
    
    def __init__(self, input_shape, output_shape, first_layer_width=24):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.first_layer_width = first_layer_width
        self.model = None
        
    def build_model(self):
        """Build a small student model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # First layer with specified width (24)
        x = layers.Dense(self.first_layer_width, activation='swish')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Additional small layers
        x = layers.Dense(16, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(8, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(self.output_shape, activation='linear')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """Compile the student model"""
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        print("Student model compiled successfully")
        print(f"Total parameters: {self.model.count_params():,}")

class DistillationTrainer:
    """Handles the distillation training process"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss vs ground truth loss
        
    def distillation_loss(self, y_true, y_pred, teacher_predictions):
        """Custom distillation loss combining ground truth and teacher knowledge"""
        # Ground truth loss
        ground_truth_loss = keras.losses.mean_absolute_error(y_true, y_pred)
        
        # Distillation loss (soft targets)
        distillation_loss = keras.losses.mean_absolute_error(
            teacher_predictions / self.temperature,
            y_pred / self.temperature
        )
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * ground_truth_loss
        return total_loss
    
    def create_distillation_model(self):
        """Create a model that combines teacher and student for distillation"""
        student_inputs = self.student_model.input
        student_outputs = self.student_model.output
        
        # Get teacher predictions
        teacher_outputs = self.teacher_model(student_inputs)
        
        # Create distillation model
        distillation_model = keras.Model(
            inputs=student_inputs,
            outputs=[student_outputs, teacher_outputs]
        )
        
        return distillation_model

def load_data():
    with h5py.File('training_data.h5', 'r') as f:
        X = np.array(f['X'][:], dtype=np.float32)
        Y = np.array(f['Y'][:], dtype=np.float32)

    Y = tf.clip_by_value(Y, clip_value_min=-90.0, clip_value_max=90.0)
    Y = Y.numpy()
    
    return X, Y

def create_callbacks():
    """Create training callbacks for safe training"""
    callbacks_list = [
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint to save best model
        callbacks.ModelCheckpoint(
            'best_teacher_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks_list

def plot_training_history(history, title):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title(f'{title} - MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("Starting TensorFlow Model Training with Distillation")
    print("=" * 60)
    
    # Load data
    X, Y = load_data()
    
    # Data preprocessing
    preprocessor = DataPreprocessor()
    X_clean, Y_clean = preprocessor.clean_and_clip_data(X, Y)
    
    # Split data: 90% training, 10% validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_clean, Y_clean, test_size=0.1, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Scale the data
    X_train_scaled, Y_train_scaled = preprocessor.scale_data(X_train, Y_train, fit=True)
    X_val_scaled, Y_val_scaled = preprocessor.scale_data(X_val, Y_val, fit=False)
    
    # Build and train teacher model
    print("\n" + "="*40)
    print("TRAINING TEACHER MODEL")
    print("="*40)
    
    teacher = TeacherModel(input_shape=(X_train.shape[1],), output_shape=Y_train.shape[1])
    teacher_model = teacher.build_model()
    teacher.compile_model(learning_rate=1e-4)
    
    # Train teacher model
    teacher_history = teacher_model.fit(
        X_train_scaled, Y_train_scaled,
        validation_data=(X_val_scaled, Y_val_scaled),
        epochs=100,
        batch_size=32,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Plot teacher training history
    plot_training_history(teacher_history, "Teacher Model")
    
    # Evaluate teacher model
    teacher_val_loss, teacher_val_mae, teacher_val_mse = teacher_model.evaluate(
        X_val_scaled, Y_val_scaled, verbose=0
    )
    print(f"\nTeacher Model Validation Results:")
    print(f"Loss: {teacher_val_loss:.4f}")
    print(f"MAE: {teacher_val_mae:.4f}")
    print(f"MSE: {teacher_val_mse:.4f}")
    
    # Build and train student model
    print("\n" + "="*40)
    print("TRAINING STUDENT MODEL")
    print("="*40)
    
    student = StudentModel(
        input_shape=(X_train.shape[1],), 
        output_shape=Y_train.shape[1], 
        first_layer_width=24
    )
    student_model = student.build_model()
    student.compile_model(learning_rate=1e-4)
    
    # Train student model
    student_history = student_model.fit(
        X_train_scaled, Y_train_scaled,
        validation_data=(X_val_scaled, Y_val_scaled),
        epochs=100,
        batch_size=32,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Plot student training history
    plot_training_history(student_history, "Student Model")
    
    # Evaluate student model
    student_val_loss, student_val_mae, student_val_mse = student_model.evaluate(
        X_val_scaled, Y_val_scaled, verbose=0
    )
    print(f"\nStudent Model Validation Results:")
    print(f"Loss: {student_val_loss:.4f}")
    print(f"MAE: {student_val_mae:.4f}")
    print(f"MSE: {student_val_mse:.4f}")
    
    # Model distillation
    print("\n" + "="*40)
    print("MODEL DISTILLATION")
    print("="*40)
    
    # Get teacher predictions for distillation
    teacher_predictions = teacher_model.predict(X_train_scaled)
    
    # Create distillation trainer
    distillation_trainer = DistillationTrainer(
        teacher_model, student_model, temperature=3.0, alpha=0.7
    )
    
    # Create a custom training loop for distillation
    def distillation_training_step(x_batch, y_batch, teacher_pred_batch):
        with tf.GradientTape() as tape:
            student_pred = student_model(x_batch, training=True)
            loss = distillation_trainer.distillation_loss(
                y_batch, student_pred, teacher_pred_batch
            )
        
        gradients = tape.gradient(loss, student_model.trainable_variables)
        student_model.optimizer.apply_gradients(
            zip(gradients, student_model.trainable_variables)
        )
        return loss
    
    # Distillation training
    print("Starting distillation training...")
    distillation_epochs = 50
    batch_size = 32
    
    for epoch in range(distillation_epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train_scaled), batch_size):
            x_batch = X_train_scaled[i:i+batch_size]
            y_batch = Y_train_scaled[i:i+batch_size]
            teacher_pred_batch = teacher_predictions[i:i+batch_size]
            
            loss = distillation_training_step(x_batch, y_batch, teacher_pred_batch)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if epoch % 10 == 0:
            print(f"Distillation Epoch {epoch+1}/{distillation_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate distilled student model
    distilled_val_loss, distilled_val_mae, distilled_val_mse = student_model.evaluate(
        X_val_scaled, Y_val_scaled, verbose=0
    )
    print(f"\nDistilled Student Model Validation Results:")
    print(f"Loss: {distilled_val_loss:.4f}")
    print(f"MAE: {distilled_val_mae:.4f}")
    print(f"MSE: {distilled_val_mse:.4f}")
    
    # Save models
    teacher_model.save('teacher_model.h5')
    student_model.save('student_model.h5')
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Teacher Model Parameters: {teacher_model.count_params():,}")
    print(f"Student Model Parameters: {student_model.count_params():,}")
    print(f"Compression Ratio: {teacher_model.count_params() / student_model.count_params():.1f}x")
    print(f"\nFinal Validation MAE:")
    print(f"  Teacher: {teacher_val_mae:.4f}")
    print(f"  Student (before distillation): {student_val_mae:.4f}")
    print(f"  Student (after distillation): {distilled_val_mae:.4f}")
    
    print("\nTraining completed successfully!")
    print("Models saved as 'teacher_model.h5' and 'student_model.h5'")

if __name__ == "__main__":
    main()
