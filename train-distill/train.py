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
        # Ensure all inputs are tensors with the same shape and dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        teacher_predictions = tf.cast(teacher_predictions, tf.float32)
        
        # Ensure all tensors have the same shape
        if y_true.shape != y_pred.shape:
            print(f"Warning: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
            # Try to broadcast if possible
            y_pred = tf.broadcast_to(y_pred, y_true.shape)
        
        if teacher_predictions.shape != y_pred.shape:
            print(f"Warning: teacher_predictions shape {teacher_predictions.shape} != y_pred shape {y_pred.shape}")
            # Try to broadcast if possible
            teacher_predictions = tf.broadcast_to(teacher_predictions, y_pred.shape)
        
        # Ground truth loss
        ground_truth_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Distillation loss (soft targets)
        distillation_loss = tf.reduce_mean(tf.abs(
            teacher_predictions / self.temperature - y_pred / self.temperature
        ))
        
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

def prepare_data():
    """Prepare and preprocess the data"""
    print("Preparing data...")
    
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
    
    return (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, preprocessor)

def train_teacher_model(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled):
    """Train the teacher model"""
    print("\n" + "="*40)
    print("TRAINING TEACHER MODEL")
    print("="*40)
    
    teacher = TeacherModel(input_shape=(X_train_scaled.shape[1],), output_shape=Y_train_scaled.shape[1])
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
    
    # Save teacher model
    teacher_model.save('teacher_model.h5')
    
    return teacher_model, teacher_val_mae

def train_student_model(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, first_layer_width=16):
    """Train the student model"""
    print("\n" + "="*40)
    print("TRAINING STUDENT MODEL")
    print("="*40)
    
    student = StudentModel(
        input_shape=(X_train_scaled.shape[1],), 
        output_shape=Y_train_scaled.shape[1], 
        first_layer_width=first_layer_width
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
    
    return student_model, student_val_mae

def run_distillation(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, 
                    teacher_model, student_model, temperature=3.0, alpha=0.7, epochs=50):
    """Run model distillation"""
    print("\n" + "="*40)
    print("MODEL DISTILLATION")
    print("="*40)
    
    print(f"Teacher model type: {type(teacher_model)}")
    print(f"Student model type: {type(student_model)}")
    
    # Get teacher predictions for distillation
    print("Getting teacher predictions...")
    teacher_predictions = teacher_model.predict(X_train_scaled)
    print(f"Teacher predictions shape: {teacher_predictions.shape}")
    print(f"Y_train_scaled shape: {Y_train_scaled.shape}")
    
    # Create distillation trainer
    distillation_trainer = DistillationTrainer(
        teacher_model, student_model, temperature=temperature, alpha=alpha
    )
    
    # Test distillation loss with a small batch to catch shape issues early
    print("Testing distillation loss with sample batch...")
    test_batch_size = min(10, len(X_train_scaled))
    test_x = X_train_scaled[:test_batch_size]
    test_y = Y_train_scaled[:test_batch_size]
    test_teacher_pred = teacher_predictions[:test_batch_size]
    
    test_student_pred = student_model.predict(test_x)
    print(f"Test student predictions shape: {test_student_pred.shape}")
    
    try:
        test_loss = distillation_trainer.distillation_loss(test_y, test_student_pred, test_teacher_pred)
        print(f"Test distillation loss: {test_loss}")
        print("Distillation loss test passed ✓")
    except Exception as e:
        print(f"Error in distillation loss test: {e}")
        raise
    
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
        return tf.reduce_mean(loss)  # Ensure we return a scalar
    
    # Distillation training
    print("Starting distillation training...")
    batch_size = 32
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X_train_scaled), batch_size):
            x_batch = X_train_scaled[i:i+batch_size]
            y_batch = Y_train_scaled[i:i+batch_size]
            teacher_pred_batch = teacher_predictions[i:i+batch_size]
            
            # Convert to tensors to ensure consistency
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            teacher_pred_batch = tf.convert_to_tensor(teacher_pred_batch, dtype=tf.float32)
            
            loss = distillation_training_step(x_batch, y_batch, teacher_pred_batch)
            total_loss += float(loss)  # Convert to Python float
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if epoch % 10 == 0:
            print(f"Distillation Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate distilled student model
    distilled_val_loss, distilled_val_mae, distilled_val_mse = student_model.evaluate(
        X_val_scaled, Y_val_scaled, verbose=0
    )
    print(f"\nDistilled Student Model Validation Results:")
    print(f"Loss: {distilled_val_loss:.4f}")
    print(f"MAE: {distilled_val_mae:.4f}")
    print(f"MSE: {distilled_val_mse:.4f}")
    
    # Save distilled student model
    student_model.save('student_model_distilled.h5')
    
    return distilled_val_mae

def load_pretrained_models():
    """Load pretrained teacher and student models"""
    print("Loading pretrained models...")
    
    if not os.path.exists('teacher_model.h5'):
        raise FileNotFoundError("Teacher model 'teacher_model.h5' not found!")
    
    teacher_model = keras.models.load_model('teacher_model.h5')
    print(f"Teacher model loaded: {type(teacher_model)}")

    if os.path.exists('student_model.h5'):
        student_model = keras.models.load_model('student_model.h5')
        print("Student model loaded from 'student_model.h5'")
    elif os.path.exists('student_model_distilled.h5'):
        student_model = keras.models.load_model('student_model_distilled.h5')
        print("Student model loaded from 'student_model_distilled.h5'")
    else:
        print("Building new student model...")
        # build new student model
        student_builder = StudentModel(
            input_shape=(teacher_model.input_shape[1],), 
            output_shape=teacher_model.output_shape[1], 
            first_layer_width=16
        )
        student_model = student_builder.build_model()
        student_builder.compile_model(learning_rate=1e-4)
        # Get the compiled model from the builder
        student_model = student_builder.model
        print("New student model built and compiled")
    
    print(f"Student model type: {type(student_model)}")
    
    # Test that the student model is callable
    try:
        test_input = np.random.randn(1, teacher_model.input_shape[1])
        test_output = student_model(test_input)
        print("Student model is callable ✓")
    except Exception as e:
        print(f"Error testing student model: {e}")
        raise
    
    print("Models loaded successfully!")
    return teacher_model, student_model

def main():
    """Main training function - runs complete pipeline"""
    print("Starting TensorFlow Model Training with Distillation")
    print("=" * 60)
    
    # Prepare data
    X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, preprocessor = prepare_data()
    
    # Train teacher model
    teacher_model, teacher_val_mae = train_teacher_model(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled)
    
    # Train student model
    student_model, student_val_mae = train_student_model(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled)
    
    # Run distillation
    distilled_val_mae = run_distillation(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, 
                                       teacher_model, student_model)
    
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
    print("Models saved as 'teacher_model.h5' and 'student_model_distilled.h5'")

def run_distillation_only():
    """Run only the distillation process with pretrained models"""
    print("Running Model Distillation Only")
    print("=" * 60)
    
    # Prepare data
    X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, preprocessor = prepare_data()
    
    # Load pretrained models
    teacher_model, student_model = load_pretrained_models()
    
    # Run distillation
    distilled_val_mae = run_distillation(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, 
                                       teacher_model, student_model)
    
    # Summary
    print("\n" + "="*60)
    print("DISTILLATION SUMMARY")
    print("="*60)
    print(f"Teacher Model Parameters: {teacher_model.count_params():,}")
    print(f"Student Model Parameters: {student_model.count_params():,}")
    print(f"Compression Ratio: {teacher_model.count_params() / student_model.count_params():.1f}x")
    print(f"Distilled Student MAE: {distilled_val_mae:.4f}")
    
    print("\nDistillation completed successfully!")
    print("Distilled model saved as 'student_model_distilled.h5'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--distillation-only":
        run_distillation_only()
    else:
        main()
