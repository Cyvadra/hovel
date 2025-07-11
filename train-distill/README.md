# TensorFlow Model Training with Distillation

This program implements a comprehensive TensorFlow training pipeline with model distillation for regression tasks.

## Features

- **Data Preprocessing**: Automatic data cleaning, clipping, and scaling
- **Teacher Model**: Large, deep neural network with advanced architecture
- **Student Model**: Compact model with configurable first layer width (default: 24)
- **Model Distillation**: Knowledge transfer from teacher to student model
- **Safe Training**: Early stopping, learning rate scheduling, and monitoring
- **Advanced Activations**: Swish activation functions throughout
- **Visualization**: Training history plots and performance metrics

## Data Requirements

- **Input (X)**: Shape (18000, 366) - 18,000 samples with 366 features
- **Target (Y)**: Shape (18000, 5) - 18,000 samples with 5 target values
- **Y Range**: Values are automatically clipped to [-100, 100]

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training program:
```bash
python train.py
```

## Program Structure

### DataPreprocessor
- Cleans data by removing NaN and infinite values
- Clips Y values to [-100, 100] range
- Scales features and targets using StandardScaler
- Splits data: 90% training, 10% validation

### TeacherModel
- Large architecture: 512 → 256 → 128 → 64 → 32 → 5
- Uses Swish activation functions
- Includes BatchNormalization and Dropout for regularization
- ~200K+ parameters

### StudentModel
- Compact architecture: 24 → 16 → 8 → 5
- Configurable first layer width (default: 24)
- Uses Swish activation functions
- ~1K+ parameters
- ~200x compression ratio

### DistillationTrainer
- Implements knowledge distillation with temperature scaling
- Combines ground truth loss with teacher knowledge loss
- Custom training loop for distillation process

## Training Process

1. **Data Preparation**: Clean, clip, and scale the data
2. **Teacher Training**: Train the large teacher model
3. **Student Training**: Train the small student model
4. **Distillation**: Transfer knowledge from teacher to student
5. **Evaluation**: Compare performance of all models

## Safe Training Features

- **Early Stopping**: Prevents overfitting with patience=15
- **Learning Rate Scheduling**: Reduces LR on plateau with patience=8
- **Model Checkpointing**: Saves best model weights
- **TensorBoard Logging**: For training monitoring
- **Small Learning Rate**: Starts with 1e-4 for stable training

## Output Files

- `teacher_model.h5`: Trained teacher model
- `student_model.h5`: Trained and distilled student model
- `best_teacher_model.h5`: Best teacher model weights
- `teacher_model_training_history.png`: Teacher training plots
- `student_model_training_history.png`: Student training plots
- `./logs/`: TensorBoard logs

## Customization

### Load Your Own Data
Replace the `generate_sample_data()` call in `main()` with your data loading:

```python
# Load your data here
X = your_data_loading_function()
Y = your_target_loading_function()
```

### Modify Model Architecture
- **Teacher**: Modify `TeacherModel.build_model()`
- **Student**: Modify `StudentModel.build_model()` and `first_layer_width` parameter

### Adjust Training Parameters
- **Learning Rate**: Change in `compile_model(learning_rate=...)`
- **Batch Size**: Modify in `model.fit(batch_size=...)`
- **Epochs**: Change in `model.fit(epochs=...)`
- **Distillation**: Adjust `temperature` and `alpha` in `DistillationTrainer`

## Performance Metrics

The program tracks and reports:
- **MAE (Mean Absolute Error)**: Primary loss function
- **MSE (Mean Squared Error)**: Additional metric
- **Compression Ratio**: Teacher vs Student parameter count
- **Training History**: Loss and MAE plots

## Example Output

```
Starting TensorFlow Model Training with Distillation
============================================================
Generating sample data...
Cleaning and clipping data...
Final data shape: X=(18000, 366), Y=(18000, 5)
Training set: (16200, 366)
Validation set: (1800, 366)

========================================
TRAINING TEACHER MODEL
========================================
Teacher model compiled successfully
Total parameters: 234,501

========================================
TRAINING STUDENT MODEL
========================================
Student model compiled successfully
Total parameters: 1,157

========================================
MODEL DISTILLATION
========================================
Starting distillation training...

============================================================
TRAINING SUMMARY
============================================================
Teacher Model Parameters: 234,501
Student Model Parameters: 1,157
Compression Ratio: 202.7x

Final Validation MAE:
  Teacher: 0.1234
  Student (before distillation): 0.1567
  Student (after distillation): 0.1345
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy 1.21+
- scikit-learn 1.0+
- matplotlib 3.5+
- pandas 1.3+

## Notes

- The program uses random seeds for reproducibility
- All models use MAE as the loss function as requested
- Swish activation functions are used throughout for better performance
- The student model's first layer width is set to 24 as specified
- Training uses the safest methods with small learning rates and comprehensive monitoring 