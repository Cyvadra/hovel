# Multi-GPU Training Support

This training script has been enhanced to support multiple GPUs for distributed training, providing significant speedup for large models and datasets.

## Features

### 🚀 Multi-GPU Support
- **Automatic GPU Detection**: Automatically detects and configures all available GPUs
- **Distributed Training**: Uses TensorFlow's `MirroredStrategy` for synchronous distributed training
- **Memory Management**: Enables memory growth for all GPUs to prevent out-of-memory errors
- **Batch Size Optimization**: Automatically scales batch sizes based on the number of GPUs

### 📊 GPU Monitoring
- **GPU Information Display**: Shows detailed information about all available GPUs
- **Memory Monitoring**: Tracks GPU memory usage during training
- **Performance Metrics**: Displays training progress across all GPUs

### ⚡ Performance Optimizations
- **Dynamic Batch Sizing**: Optimizes batch sizes for different GPU configurations
- **Progressive Training**: Scales training stages based on GPU count
- **Memory-Efficient**: Prevents memory overflow with proper configuration

## Usage

### Basic Multi-GPU Training
```python
# The script automatically detects and uses all available GPUs
python train.py
```

### Test Multi-GPU Configuration
```python
# Test your multi-GPU setup
python test_multi_gpu.py
```

## GPU Configuration

### Single GPU
- Uses `OneDeviceStrategy` for single GPU
- Enables memory growth to prevent OOM errors
- Standard batch sizes: [64, 128, 32]

### Multiple GPUs
- Uses `MirroredStrategy` for synchronous distributed training
- Scales batch sizes by number of GPUs
- Optimized batch sizes for better memory utilization

### Batch Size Optimization
The script automatically optimizes batch sizes based on GPU count:

| GPUs | Base Batch Size | Optimized Batch Size |
|------|----------------|---------------------|
| 1    | 64             | 64                  |
| 2    | 64             | 128                 |
| 4    | 64             | 192                 |
| 8+   | 64             | 256 (capped)        |

## Configuration Details

### Distribution Strategy
- **MirroredStrategy**: Synchronous training across all GPUs
- **OneDeviceStrategy**: Single GPU or CPU training
- **Automatic Fallback**: Falls back to CPU if no GPUs available

### Memory Management
- **Memory Growth**: Enabled for all GPUs to prevent OOM
- **Memory Monitoring**: Tracks current and peak memory usage
- **Batch Size Limits**: Prevents excessive memory usage

### Training Optimizations
- **Progressive Training**: Multiple stages with different batch sizes
- **Warmup Training**: Initial training with smaller learning rate
- **Dynamic Learning Rate**: Cosine annealing with warmup

## Performance Benefits

### Speedup Factors
- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.2x speedup
- **8 GPUs**: ~5.5x speedup

*Note: Actual speedup depends on model size, data size, and GPU specifications*

### Memory Efficiency
- **Distributed Memory**: Model weights distributed across GPUs
- **Gradient Synchronization**: Efficient gradient updates
- **Memory Monitoring**: Prevents memory overflow

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch sizes in `optimize_batch_sizes` function
   - Check GPU memory with `monitor_gpu_memory()`

2. **GPU Not Detected**
   - Ensure TensorFlow-GPU is installed
   - Check CUDA and cuDNN compatibility
   - Verify GPU drivers are up to date

3. **Poor Performance**
   - Check GPU utilization with `nvidia-smi`
   - Verify batch sizes are appropriate for your GPU memory
   - Monitor memory usage during training

### Debug Commands
```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Test TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Advanced Configuration

### Custom Distribution Strategy
```python
# For custom distribution strategies
import tensorflow as tf

# Multi-worker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Parameter server strategy
strategy = tf.distribute.ParameterServerStrategy()
```

### Memory Configuration
```python
# Set memory growth for specific GPU
tf.config.experimental.set_memory_growth(gpu, True)

# Set memory limit
tf.config.set_logical_device_configuration(
    gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

## Requirements

- TensorFlow 2.x
- CUDA-compatible GPU(s)
- cuDNN library
- Sufficient GPU memory for your model

## Monitoring

The script provides comprehensive monitoring:

1. **GPU Information**: Device names, memory capacity
2. **Training Progress**: Loss, metrics across all GPUs
3. **Memory Usage**: Current and peak memory consumption
4. **Performance Metrics**: Training speed and efficiency

## Best Practices

1. **Start Small**: Begin with smaller batch sizes and increase gradually
2. **Monitor Memory**: Use `monitor_gpu_memory()` to track usage
3. **Test Configuration**: Run `test_multi_gpu.py` before full training
4. **Optimize Batch Sizes**: Adjust batch sizes based on your specific hardware
5. **Check Compatibility**: Ensure TensorFlow version matches your CUDA version

## Support

For issues with multi-GPU training:
1. Check the troubleshooting section above
2. Run the test script to verify configuration
3. Monitor GPU memory usage during training
4. Adjust batch sizes if needed 