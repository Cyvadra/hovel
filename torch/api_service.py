#!/usr/bin/env python3
"""
API Service for the Optimized PyTorch Model

This script loads a trained model and serves it as a REST API using FastAPI.
The model expects input data and returns predictions.

Usage:
    python api_service.py --model_path path/to/model.pth --port 8000
"""

import argparse
import logging
import sys
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from contextlib import asynccontextmanager
import time
import json

# Import the model class from the training script
from train import OptimizedModel, extract_final_predictions

# --- Logging Setup ---
def setup_logging():
    """Setup logging configuration for the API service."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('api_service.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Model Management ---
class ModelManager:
    """Manages the loaded model and provides prediction functionality."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.input_dim = None
        self.output_dim = None
        self.hidden_size = None
        self.num_layers = None
        self.is_loaded = False
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(device)
            logger.info(f"Using device: {device}")
        
        return device
    
    def _parse_model_params_from_filename(self, filename: str) -> tuple:
        """
        Parse hidden_size and num_layers from filename.
        
        Expected format: model_{hidden_size}_layers_{num_layers}_best_model.pth
        Examples:
        - model_1024_layers_16_best_model.pth -> (1024, 16)
        - model_512_layers_4_best_model.pth -> (512, 4)
        - optimized_model_best_model.pth -> (512, 4) [default]
        
        Returns:
            tuple: (hidden_size, num_layers)
        """
        # Default values
        default_hidden_size = 512
        default_num_layers = 4
        
        # Extract filename without path
        basename = os.path.basename(filename)
        
        # Pattern to match: model_{hidden_size}_layers_{num_layers}_best_model.pth
        pattern = r'model_(\d+)_layers_(\d+)_best_model\.pth'
        match = re.search(pattern, basename)
        
        if match:
            hidden_size = int(match.group(1))
            num_layers = int(match.group(2))
            logger.info(f"Parsed from filename: hidden_size={hidden_size}, num_layers={num_layers}")
            return hidden_size, num_layers
        else:
            # Try alternative patterns
            patterns = [
                r'model_(\d+)_(\d+)_best_model\.pth',  # model_1024_16_best_model.pth
                r'model_(\d+)_layers_(\d+)\.pth',      # model_1024_layers_16.pth
                r'model_(\d+)_(\d+)\.pth',             # model_1024_16.pth
            ]
            
            for pattern in patterns:
                match = re.search(pattern, basename)
                if match:
                    hidden_size = int(match.group(1))
                    num_layers = int(match.group(2))
                    logger.info(f"Parsed from filename (alt pattern): hidden_size={hidden_size}, num_layers={num_layers}")
                    return hidden_size, num_layers
            
            # If no pattern matches, use defaults
            logger.info(f"Could not parse model parameters from filename '{basename}', using defaults: hidden_size={default_hidden_size}, num_layers={default_num_layers}")
            return default_hidden_size, default_num_layers
    
    def load_model(self) -> bool:
        """Load the model from the specified path."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Parse model parameters from filename
            self.hidden_size, self.num_layers = self._parse_model_params_from_filename(self.model_path)
            
            # Load the model state
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Handle DataParallel saved models
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[key] = value
                else:
                    new_state_dict[key] = value
            
            # Try to infer model dimensions from the state dict
            # Look for input_proj.weight to get input_dim
            if 'input_proj.weight' in new_state_dict:
                self.input_dim = new_state_dict['input_proj.weight'].shape[1]
            else:
                # Default values if we can't infer
                self.input_dim = 512
                logger.warning("Could not infer input_dim from model, using default: 512")
            
            # Look for output_proj.weight to get output_dim
            if 'output_proj.weight' in new_state_dict:
                # The output is 3*output_dim, so divide by 3
                self.output_dim = new_state_dict['output_proj.weight'].shape[0] // 3
            else:
                # Default values if we can't infer
                self.output_dim = 1
                logger.warning("Could not infer output_dim from model, using default: 1")
            
            # Create the model with parsed parameters
            self.model = OptimizedModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_rate=0.1
            )
            
            # Load the state dict
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully!")
            logger.info(f"Input dimension: {self.input_dim}")
            logger.info(f"Output dimension: {self.output_dim}")
            logger.info(f"Hidden size: {self.hidden_size}")
            logger.info(f"Number of layers: {self.num_layers}")
            logger.info(f"Model output dimension: {3 * self.output_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            input_data: Input data of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        
        if input_data.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {input_data.shape[1]}")
        
        try:
            # Convert to tensor and move to device
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
            
            # Make prediction
            with torch.no_grad():
                model_output = self.model(input_tensor)
                
                # Extract final predictions from the 3*output_dim output
                predictions = extract_final_predictions(model_output, self.output_dim)
                
                # Convert back to numpy
                predictions_np = predictions.cpu().numpy()
                
                return predictions_np
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": str(self.device),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "model_output_dim": 3 * self.output_dim if self.output_dim else None,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

# --- Pydantic Models for API ---
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[List[float]] = Field(..., description="Input data as a list of lists")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        
        # Check if all rows have the same length
        if len(set(len(row) for row in v)) > 1:
            raise ValueError("All rows must have the same length")
        
        return v

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[List[float]] = Field(..., description="Model predictions")
    input_shape: List[int] = Field(..., description="Shape of input data")
    output_shape: List[int] = Field(..., description="Shape of output data")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Dict[str, Any] = Field(..., description="Model information")

# --- Global Variables ---
model_manager: Optional[ModelManager] = None

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global model_manager
    
    # Startup
    logger.info("Starting API service...")
    if model_manager:
        success = model_manager.load_model()
        if not success:
            logger.error("Failed to load model during startup")
            sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API service...")

app = FastAPI(
    title="Optimized PyTorch Model API",
    description="API service for the optimized PyTorch model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Optimized PyTorch Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global model_manager
    
    if model_manager is None:
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_info={"error": "Model manager not initialized"}
        )
    
    model_info = model_manager.get_model_info()
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "error",
        model_loaded=model_manager.is_loaded,
        model_info=model_info
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the loaded model."""
    global model_manager
    
    if model_manager is None or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request data to numpy array
        input_data = np.array(request.data, dtype=np.float32)
        
        # Record processing time
        start_time = time.time()
        
        # Make prediction
        predictions = model_manager.predict(input_data)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            input_shape=list(input_data.shape),
            output_shape=list(predictions.shape),
            processing_time=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about the loaded model."""
    global model_manager
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return model_manager.get_model_info()

@app.post("/model/reload")
async def reload_model():
    """Reload the model."""
    global model_manager
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        success = model_manager.load_model()
        if success:
            return {"message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")

# --- Utility Functions ---
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="API Service for Optimized PyTorch Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the API service on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the API service to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the model on (default: auto)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    return parser.parse_args()

# --- Main Function ---
def main():
    """Main function to run the API service."""
    global model_manager
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Initialize model manager
    model_manager = ModelManager(args.model_path, args.device)
    
    # Load model
    if not model_manager.load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Start the server
    logger.info(f"Starting API service on {args.host}:{args.port}")
    logger.info(f"Model loaded from: {args.model_path}")
    logger.info(f"API documentation available at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 