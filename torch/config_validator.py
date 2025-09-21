"""
Configuration validation module for the PyTorch training pipeline.

This module provides validation for training configurations to ensure all parameters
are within acceptable ranges and have appropriate types.
"""

import json
import os
from typing import Dict, Any, Union, Optional
from error_handling import ConfigurationError, validate_positive

class ConfigValidator:
    """
    Validator class for training configurations.
    Ensures all parameters are within acceptable ranges and have appropriate types.
    """
    
    # Define parameter constraints
    PARAM_CONSTRAINTS = {
        # Model parameters
        'num_layers': {'type': int, 'min': 1, 'max': 32},
        'hidden_size': {'type': int, 'min': 32, 'max': 4096},
        'noise_std': {'type': float, 'min': 0.0, 'max': 1.0},
        'noise_decay': {'type': float, 'min': 0.9, 'max': 1.0},
        'min_noise_std': {'type': float, 'min': 0.0, 'max': 0.1},
        
        # Training parameters
        'batch_size': {'type': int, 'min': 1, 'max': 1024},
        'gradient_accumulation_steps': {'type': int, 'min': 1, 'max': 32},
        'learning_rate': {'type': float, 'min': 1e-7, 'max': 1.0},
        'weight_decay': {'type': float, 'min': 0.0, 'max': 0.1},
        'min_epochs': {'type': int, 'min': 1, 'max': 1000},
        'max_epochs': {'type': int, 'min': 1, 'max': 10000},
        'patience': {'type': int, 'min': 1, 'max': 200},
        'gradient_clip_norm': {'type': float, 'min': 0.0, 'max': 10.0},
        'mixup_alpha': {'type': float, 'min': 0.0, 'max': 1.0},
        
        # Data parameters
        'val_split': {'type': float, 'min': 0.0, 'max': 0.3},
        'test_split': {'type': float, 'min': 0.0, 'max': 0.3},
        
        # Scheduler parameters
        'scheduler_t0': {'type': int, 'min': 1, 'max': 100},
        'scheduler_t_mult': {'type': int, 'min': 1, 'max': 10},
        'scheduler_eta_min': {'type': float, 'min': 0.0, 'max': 0.1},
        
        # SWA parameters
        'swa_start': {'type': int, 'min': 1, 'max': 1000},
        'swa_lr': {'type': float, 'min': 1e-7, 'max': 0.1},
        'swa_freq': {'type': int, 'min': 1, 'max': 50},
        'swa_anneal_epochs': {'type': int, 'min': 1, 'max': 50},
        'swa_anneal_strategy': {'type': str, 'choices': ['cos', 'linear']},
        
        # Warmup parameters
        'warmup_epochs': {'type': int, 'min': 0, 'max': 100},
        'warmup_start_lr': {'type': float, 'min': 1e-10, 'max': 0.1},
        
        # EMA parameters
        'ema_decay': {'type': float, 'min': 0.9, 'max': 1.0},
        'ema_start': {'type': int, 'min': 0, 'max': 100},
        
        # Adaptive noise parameters
        'adaptive_noise': {'type': bool},
        'noise_grad_threshold': {'type': float, 'min': 0.0, 'max': 10.0},
        'noise_scale_factor': {'type': float, 'min': 0.0, 'max': 1.0},
        
        # Dynamic validation parameters
        'dynamic_val_freq': {'type': bool},
        'min_val_freq': {'type': int, 'min': 1, 'max': 20},
        'max_val_freq': {'type': int, 'min': 1, 'max': 50},
        'val_stability_threshold': {'type': float, 'min': 0.0, 'max': 0.1},
        
        # Model saving
        'save_checkpoint_every': {'type': int, 'min': 1, 'max': 100}
    }
    
    @classmethod
    def validate_value(cls, name: str, value: Any, constraints: Dict[str, Any]) -> Any:
        """
        Validate a single configuration value against its constraints.
        
        Args:
            name: Parameter name
            value: Parameter value
            constraints: Dictionary of constraints for the parameter
            
        Returns:
            The validated value (may be coerced to correct type)
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Check type
        expected_type = constraints['type']
        try:
            if expected_type == bool and isinstance(value, int):
                value = bool(value)
            elif not isinstance(value, expected_type):
                value = expected_type(value)
        except (ValueError, TypeError):
            raise ConfigurationError(
                f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}"
            )
        
        # Check choices for enum-like parameters
        if 'choices' in constraints and value not in constraints['choices']:
            raise ConfigurationError(
                f"Parameter '{name}' must be one of {constraints['choices']}, got {value}"
            )
        
        # Check numeric bounds
        if 'min' in constraints and value < constraints['min']:
            raise ConfigurationError(
                f"Parameter '{name}' must be >= {constraints['min']}, got {value}"
            )
        if 'max' in constraints and value > constraints['max']:
            raise ConfigurationError(
                f"Parameter '{name}' must be <= {constraints['max']}, got {value}"
            )
        
        return value

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an entire configuration dictionary.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            Dictionary with validated and potentially coerced values
            
        Raises:
            ConfigurationError: If validation fails
        """
        validated = {}
        
        # Validate known parameters
        for name, value in config.items():
            if name in cls.PARAM_CONSTRAINTS:
                validated[name] = cls.validate_value(
                    name, value, cls.PARAM_CONSTRAINTS[name]
                )
            else:
                # Keep unknown parameters as-is but warn
                print(f"Warning: Unknown configuration parameter '{name}'")
                validated[name] = value
        
        # Cross-validate parameters
        if 'min_epochs' in validated and 'max_epochs' in validated:
            if validated['min_epochs'] >= validated['max_epochs']:
                raise ConfigurationError(
                    f"min_epochs ({validated['min_epochs']}) must be less than "
                    f"max_epochs ({validated['max_epochs']})"
                )
        
        if 'min_val_freq' in validated and 'max_val_freq' in validated:
            if validated['min_val_freq'] > validated['max_val_freq']:
                raise ConfigurationError(
                    f"min_val_freq ({validated['min_val_freq']}) must be <= "
                    f"max_val_freq ({validated['max_val_freq']})"
                )
        
        return validated

def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a training configuration dictionary.
    
    Args:
        config: Dictionary of configuration parameters
        
    Returns:
        Dictionary with validated and potentially coerced values
        
    Raises:
        ConfigurationError: If validation fails
    """
    return ConfigValidator.validate_config(config)

def save_training_config(config: Dict[str, Any], model_name: str) -> None:
    """
    Save a training configuration to a JSON file.
    
    Args:
        config: Configuration dictionary (will be validated before saving)
        model_name: Base name for the model (used in filename)
    """
    # Validate config before saving
    config = validate_training_config(config)
    
    # Convert config to dictionary if it's an object
    if hasattr(config, 'to_dict'):
        config = config.to_dict()
    
    # Save to file
    filename = f"{model_name}_config.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_training_config(model_name: str) -> Dict[str, Any]:
    """
    Load a training configuration from a JSON file.
    
    Args:
        model_name: Base name for the model (used in filename)
        
    Returns:
        Dictionary with validated configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigurationError: If config is invalid
    """
    filename = f"{model_name}_config.json"
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    with open(filename, 'r') as f:
        config = json.load(f)
    
    # Validate loaded config
    return validate_training_config(config)
