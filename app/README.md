# MedGemma CXR Triage - Demo App

A containerized demo application for chest X-ray urgency triage using MedGemma.

## Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Build the image
cd docker
docker build -t medgemma-demo .

# Run with GPU
docker run --gpus all -p 7860:7860 medgemma-demo

# Or run on CPU (slower)
docker run -p 7860:7860 -e MEDGEMMA_DEVICE=cpu medgemma-demo
```

Then open http://localhost:7860 in your browser.

### Option 2: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio UI
cd ui
python gradio_app.py

# Or run FastAPI server
cd service
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (Gradio) |
| `/predict` | POST | Triage prediction (FastAPI) |
| `/health` | GET | Health check (FastAPI) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDGEMMA_MODEL_ID` | `google/medgemma-4b-it` | HuggingFace model ID |
| `MEDGEMMA_QUANTIZE` | `true` | Use 4-bit quantization |
| `MEDGEMMA_DEVICE` | `auto` | Device (`auto`, `cuda`, `cpu`) |
| `MOCK_MODEL` | `false` | Use mock model for testing |

## API Usage

### Predict Endpoint

```bash
curl -X POST http://localhost:7860/predict \
  -F "image=@chest_xray.png" \
  -F "prior_report=Previous study showed no abnormalities"
```

Response:
```json
{
  "id": "req_1234567890",
  "label": "Non-Urgent",
  "score": 0.15,
  "explanation": "Normal chest X-ray with clear lung fields",
  "crops": [],
  "provenance": [
    {"source": "model_analysis", "snippet": "No abnormalities detected"}
  ],
  "inference_time_ms": 1523.4,
  "disclaimer": "[WARNING] This is for clinical decision support only."
}
```

## Testing

Run smoke tests:
```bash
# Start the app first, then:
./tests/smoke_test.sh
```

## Project Structure

```
demo_app/
├── service/
│   ├── app.py           # FastAPI server
│   ├── model_loader.py  # Model loading utilities
│   └── predict.py       # Prediction pipeline
├── ui/
│   └── gradio_app.py    # Gradio web interface
├── docker/
│   └── Dockerfile       # Container definition
├── tests/
│   └── smoke_test.sh    # End-to-end tests
├── onnx/
│   └── build_onnx.sh    # ONNX export script
├── requirements.txt
└── README.md
```

## Disclaimer

This application is for **clinical decision support only**. It is NOT a diagnostic device and should NOT replace professional medical judgment. All AI-generated outputs require verification by qualified healthcare professionals.
