# 模型尺寸测试代码优化说明

## 优化内容

### 1. 显存清理功能

**新增功能：**
- 每次运行完一个尺寸的模型训练后，自动清理GPU显存
- 显示清理前后的显存使用情况
- 强制垃圾回收

**实现细节：**
```python
def clear_gpu_memory():
    """Clear GPU memory and garbage collect."""
    if torch.cuda.is_available():
        # 显示清理前的显存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory before clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # 清理显存缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 显示清理后的显存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory after clearing: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    # 强制垃圾回收
    gc.collect()
```

### 2. 跳过已存在图表功能

**新增功能：**
- 训练前检查模型文件是否已存在
- 训练前检查图表文件是否已存在
- 如果文件已存在，跳过该模型的训练

**实现细节：**
```python
def check_existing_plots(model_name):
    """检查特定模型的图表是否已存在"""
    plot_files = [
        f"{model_name}_loss_curve.png",
        f"{model_name}_predictions.png"
    ]
    
    all_exist = all(os.path.exists(f) for f in plot_files)
    if all_exist:
        print(f"Plots for {model_name} already exist, skipping...")
    return all_exist

def check_existing_model_files(model_name):
    """检查特定模型的模型文件是否已存在"""
    model_file = f"{model_name}_best_model.pth"
    exists = os.path.exists(model_file)
    if exists:
        print(f"Model file for {model_name} already exists, skipping...")
    return exists
```

### 3. 显存监控功能

**新增功能：**
- 显示当前GPU显存使用情况
- 在训练前显示显存状态
- 监控显存分配和缓存情况

**实现细节：**
```python
def show_gpu_memory_usage():
    """显示当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
    else:
        print("CUDA not available")
```

### 4. 比较图表和结果文件跳过功能

**新增功能：**
- 检查比较图表是否已存在，如果存在则跳过生成
- 检查结果JSON文件是否已存在，如果存在则跳过保存

## 使用方法

### 运行完整测试
```python
from test_model_sizes import test_model_sizes

# 运行完整的模型尺寸测试
hidden_sizes = [128, 1024, 2048]
num_layers = [4, 8, 16]
results = test_model_sizes(hidden_sizes, num_layers)
```

### 运行测试脚本
```bash
python test_optimized_model_sizes.py
```

## 优化效果

1. **显存管理：** 避免显存累积，防止OOM错误
2. **时间节省：** 跳过已完成的模型训练，节省重复计算时间
3. **资源监控：** 实时监控显存使用情况，便于调试和优化
4. **断点续传：** 支持中断后继续运行，不会重复已完成的工作

## 文件结构

```
torch/
├── test_model_sizes.py              # 优化后的主测试文件
├── test_optimized_model_sizes.py    # 测试脚本
├── OPTIMIZATION_README.md           # 本说明文件
└── train.py                         # 原始训练文件
```

## 注意事项

1. 确保有足够的磁盘空间存储模型文件和图表
2. 显存清理功能需要CUDA支持
3. 跳过功能基于文件存在性检查，删除文件可重新训练
4. 建议在运行前检查GPU显存是否充足 