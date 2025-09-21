"""
Error handling module for the PyTorch training pipeline.

This module provides custom exceptions, validation functions, and utility decorators
for error handling and logging in the training process.
"""

import logging
import time
import functools
import numpy as np
import torch
from typing import Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass

class DataError(TrainingError):
    """Exception raised for data-related errors."""
    pass

class ModelError(TrainingError):
    """Exception raised for model-related errors."""
    pass

class ConfigurationError(TrainingError):
    """Exception raised for configuration-related errors."""
    pass

# --- Validation Functions ---
def validate_tensor(tensor: torch.Tensor, name: str, allow_none: bool = False) -> None:
    """
    Validate a PyTorch tensor.
    
    Args:
        tensor: The tensor to validate
        name: Name of the tensor for error messages
        allow_none: Whether None is acceptable
    
    Raises:
        DataError: If validation fails
    """
    if tensor is None:
        if allow_none:
            return
        raise DataError(f"{name} cannot be None")
    
    if not isinstance(tensor, torch.Tensor):
        raise DataError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if torch.isnan(tensor).any():
        raise DataError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise DataError(f"{name} contains infinite values")

def validate_array(array: np.ndarray, name: str, allow_none: bool = False) -> None:
    """
    Validate a numpy array.
    
    Args:
        array: The array to validate
        name: Name of the array for error messages
        allow_none: Whether None is acceptable
    
    Raises:
        DataError: If validation fails
    """
    if array is None:
        if allow_none:
            return
        raise DataError(f"{name} cannot be None")
    
    if not isinstance(array, np.ndarray):
        raise DataError(f"{name} must be a numpy.ndarray, got {type(array)}")
    
    if np.isnan(array).any():
        raise DataError(f"{name} contains NaN values")
    
    if np.isinf(array).any():
        raise DataError(f"{name} contains infinite values")

def validate_positive(value: Union[int, float], name: str) -> None:
    """
    Validate that a value is positive.
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")
    
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

# --- Utility Decorators ---
def log_execution_time(func):
    """
    Decorator to log the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

def handle_exception(func):
    """
    Decorator to handle and log exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TrainingError as e:
            logger.error(f"Training error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise TrainingError(f"Error in {func.__name__}: {str(e)}") from e
    return wrapper

# --- Memory Management ---
def check_gpu_memory() -> Optional[dict]:
    """
    Check GPU memory usage.
    
    Returns:
        dict: Memory statistics if GPU is available, None otherwise
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
        max_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        memory_stats = {
            'device': device,
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': max_mem,
            'free_gb': max_mem - allocated
        }
        
        logger.info(f"GPU Memory Stats: "
                   f"Allocated: {allocated:.2f}GB, "
                   f"Cached: {cached:.2f}GB, "
                   f"Free: {(max_mem - allocated):.2f}GB")
        
        return memory_stats
        
    except Exception as e:
        logger.warning(f"Failed to check GPU memory: {str(e)}")
        return None
