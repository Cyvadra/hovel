# Optimized PyTorch Model API Service

This directory contains an API service for serving the optimized PyTorch model as a REST API using FastAPI.

## Files

- `api_service.py` - Main API service script
- `api_requirements.txt` - Python dependencies for the API service
- `test_api_client.py` - Test client to demonstrate API usage
- `API_README.md` - This documentation file

## Installation

1. Install the required dependencies:
```bash
pip install -r api_requirements.txt
```

2. Make sure you have a trained model file (`.pth` format) from the training script.

## Usage

### Starting the API Service

```bash
python api_service.py --model_path path/to/your/model.pth --port 8000
```

#### Command Line Arguments

- `--model_path` (required): Path to the trained model file (.pth)
- `--port` (optional): Port to run the service on (default: 8000)
- `--host` (optional): Host to bind to (default: 0.0.0.0)
- `--device` (optional): Device to run on - "auto", "cpu", or "cuda" (default: auto)
- `--reload` (optional): Enable auto-reload for development

#### Examples

```bash
# Basic usage
python api_service.py --model_path optimized_model_best_model.pth

# Custom port and device
python api_service.py --model_path optimized_model_best_model.pth --port 8080 --device cuda

# Development mode with auto-reload
python api_service.py --model_path optimized_model_best_model.pth --reload
```

### API Endpoints

Once the service is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

#### Available Endpoints

1. **GET /** - Root endpoint with basic information
2. **GET /health** - Health check endpoint
3. **POST /predict** - Make predictions
4. **GET /model/info** - Get model information
5. **POST /model/reload** - Reload the model

### Making Predictions

#### Request Format

```json
{
  "data": [
    [0.1, 0.2, 0.3, ...],  // Sample 1
    [0.4, 0.5, 0.6, ...],  // Sample 2
    ...
  ]
}
```

#### Response Format

```json
{
  "predictions": [
    [0.123, 0.456, ...],  // Prediction for sample 1
    [0.789, 0.012, ...],  // Prediction for sample 2
    ...
  ],
  "input_shape": [2, 512],
  "output_shape": [2, 1],
  "processing_time": 0.00123
}
```

### Testing the API

Use the provided test client:

```bash
python test_api_client.py
```

This will:
1. Check the health of the API service
2. Get model information
3. Generate test data
4. Make predictions
5. Test different batch sizes

### Example API Usage with curl

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.6, 0.7, 0.8, 0.9, 1.0]
    ]
  }'
```

### Example Python Client

```python
import requests
import numpy as np

# Generate test data
input_dim = 512  # Adjust based on your model
test_data = np.random.randn(5, input_dim).tolist()

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"data": test_data}
)

if response.status_code == 200:
    result = response.json()
    print(f"Predictions: {result['predictions']}")
    print(f"Processing time: {result['processing_time']:.3f}s")
else:
    print(f"Error: {response.text}")
```

## Model Requirements

The API service expects a model file that:

1. Was trained using the `train.py` script
2. Contains an `OptimizedModel` with the expected architecture
3. Has the correct input and output dimensions

The service will automatically:
- Detect the input and output dimensions from the model weights
- Handle DataParallel models (removes 'module.' prefix)
- Load the model on the specified device (GPU/CPU)

## Error Handling

The API service includes comprehensive error handling:

- **400 Bad Request**: Invalid input data format or dimensions
- **503 Service Unavailable**: Model not loaded or service not ready
- **500 Internal Server Error**: Unexpected errors during prediction

## Performance Considerations

- The service uses PyTorch's `torch.no_grad()` for inference to save memory
- GPU inference is automatically used if available
- Batch processing is supported for efficient predictions
- Processing time is measured and returned with each prediction

## Logging

The API service logs to both console and file (`api_service.log`):
- Model loading status
- Prediction requests and errors
- Service startup/shutdown events

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure the model path is correct
   - Check file permissions

2. **CUDA out of memory**
   - Use `--device cpu` to run on CPU
   - Reduce batch size in requests

3. **Input dimension mismatch**
   - Check the model's expected input dimension with `/model/info`
   - Ensure your input data matches this dimension

4. **Service won't start**
   - Check if the port is already in use
   - Verify all dependencies are installed
   - Check the logs for specific error messages

### Getting Help

1. Check the API documentation at `/docs`
2. Review the service logs
3. Use the health check endpoint to verify service status
4. Test with the provided test client

## Security Notes

- The service runs on `0.0.0.0` by default (accessible from any IP)
- For production use, consider:
  - Using a reverse proxy (nginx)
  - Adding authentication
  - Limiting CORS origins
  - Running behind a firewall 