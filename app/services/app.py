"""
MedGemma CXR Triage — FastAPI Service (V3.0)
=============================================

Production-grade REST API for chest X-ray urgency triage.

Endpoints:
  POST /predict     ← Upload CXR + optional prior report → Triage result
  GET  /health      ← Service health check
  GET  /version     ← Model and pipeline version info
"""

import io
import os
import time
import traceback
import uuid
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from model_loader import load_model, check_model_health
from predict import run_triage_prediction, VERSION


# ─── FastAPI Application ────────────────────────────────────────────

app = FastAPI(
    title="MedGemma CXR Triage API",
    description=(
        "Chest X-ray urgency triage using Google's MedGemma model. "
        "Clinical decision support only — all predictions require "
        "verification by a qualified radiologist."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response Models ────────────────────────────────────────────────

class CropRegion(BaseModel):
    label: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0


class ProvenanceItem(BaseModel):
    source: str
    type: str
    details: object = None
    note: str = ""


class FindingItem(BaseModel):
    finding: str
    source: str = "model"


class TriageResponse(BaseModel):
    id: str = Field(description="Unique request ID")
    label: str = Field(description="Urgency label: 'Urgent' or 'Non-Urgent'")
    score: float = Field(description="Confidence score [0, 1]")
    model_confidence: str = Field(description="Confidence level: High/Medium/Low")
    explanation: str = Field(description="One-line clinical explanation")
    key_findings: List[str] = Field(default_factory=list)
    uncertainty: float = Field(description="Uncertainty estimate [0, 1]")
    verified: bool = Field(description="Stage-2 verification result")
    abstain: bool = Field(description="Whether model abstained")
    provenance: List[ProvenanceItem] = Field(default_factory=list)
    crops: List[CropRegion] = Field(default_factory=list)
    inference_time_ms: float = Field(description="Total inference time in ms")
    ood_flag: bool = Field(default=False, description="Out-of-distribution flag")
    disclaimer: str = (
        "[WARNING] This is for clinical decision support only. "
        "Not a diagnostic tool. All findings require radiologist verification."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    pipeline_version: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    uptime_s: float


class VersionResponse(BaseModel):
    api_version: str
    pipeline_version: str
    model_id: str
    torch_version: str
    cuda_available: bool


# ─── State ──────────────────────────────────────────────────────────

_model = None
_processor = None
_start_time = time.time()
_model_id = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")


# ─── Startup ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_load_model():
    """Load MedGemma model on application startup."""
    global _model, _processor
    print("=" * 60)
    print("MedGemma CXR Triage API -- Starting")
    print("=" * 60)

    try:
        _model, _processor = load_model()
        is_healthy = check_model_health(_model, _processor)
        if is_healthy:
            print("[OK] Model loaded and healthy")
        else:
            print("[WARN] Model loaded but health check failed")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        print("  API will respond with 503 until model is available")
        _model = None
        _processor = None


# --- Endpoints ----------------------------------------------------

@app.post("/predict", response_model=TriageResponse)
async def predict(
    image: UploadFile = File(..., description="Chest X-ray image (PNG, JPEG, or DICOM)"),
    prior_report: Optional[str] = Form(None, description="Prior radiology report text"),
):
    """
    Analyze a chest X-ray and return urgency triage assessment.

    The prediction pipeline includes:
    1. Image quality & OOD checks
    2. MedGemma inference
    3. Structured response parsing
    4. Uncertainty estimation
    5. Verification (for urgent cases)
    6. Audit logging
    """
    if _model is None or _processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization or check logs.",
        )

    # Read and validate image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {e}",
        )

    # Run prediction
    try:
        result = run_triage_prediction(
            _model, _processor, img, prior_report=prior_report
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    # Convert attention regions to crops
    crops = []
    attention = result.get("model_attention", {})
    for region in attention.get("top_regions", []):
        px = region.get("pixel_region", {})
        if px:
            crops.append(CropRegion(
                label=f"attention_region (var={region.get('relative_attention', 0):.2f})",
                x=px.get("x", 0),
                y=px.get("y", 0),
                width=px.get("w", 0),
                height=px.get("h", 0),
                confidence=region.get("relative_attention", 0),
            ))

    # Build provenance
    prov_items = []
    for p in result.get("provenance", []):
        prov_items.append(ProvenanceItem(
            source=p.get("source", ""),
            type=p.get("type", ""),
            details=p.get("details"),
            note=p.get("note", ""),
        ))

    return TriageResponse(
        id=result.get("request_id", str(uuid.uuid4())[:8]),
        label=result["label"],
        score=result["score"],
        model_confidence=result.get("model_confidence", "Unknown"),
        explanation=result.get("explanation", ""),
        key_findings=result.get("key_findings", []),
        uncertainty=result.get("uncertainty", 0),
        verified=result.get("verified", False),
        abstain=result.get("abstain", False),
        provenance=prov_items,
        crops=crops,
        inference_time_ms=result.get("model_metadata", {}).get("inference_time_ms", 0),
        ood_flag=result.get("ood_flag", False),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_id=_model_id,
        pipeline_version=VERSION,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        uptime_s=round(time.time() - _start_time, 1),
    )


@app.get("/version", response_model=VersionResponse)
async def version():
    """Version information."""
    return VersionResponse(
        api_version="3.0.0",
        pipeline_version=VERSION,
        model_id=_model_id,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
