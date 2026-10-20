"""
Model Serving Benchmark API
============================
FastAPI server for ONNX Runtime model inference with:
- Async inference offloading (non-blocking event loop)
- Prometheus metrics instrumentation
- Health checks and model metadata
"""

import asyncio
import os
from contextlib import asynccontextmanager
from time import perf_counter

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
from starlette.responses import Response
from transformers import AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

MODEL_PATHS = {
    "fp32": "models/fp32/model.onnx",
    "int8": "models/int8/model.onnx",
}

LABELS = {
    0: "NEGATIVE",
    1: "POSITIVE",
}

MODEL_VARIANT = os.getenv("MODEL_VARIANT", "int8").lower()

if MODEL_VARIANT not in MODEL_PATHS:
    raise ValueError(
        f"Invalid MODEL_VARIANT: {MODEL_VARIANT}. "
        f"Use 'fp32' or 'int8'."
    )


# ---------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------

MODEL_INFO = Info(
    "model_serving",
    "Model serving metadata",
)

INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Time spent on model inference (ONNX session.run) in seconds",
    labelnames=["model_variant"],
    buckets=(
        0.001, 0.0025, 0.005, 0.0075, 0.01,
        0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0,
    ),
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Total HTTP request duration in seconds",
    labelnames=["method", "endpoint", "status_code"],
    buckets=(
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
        0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
    ),
)

ACTIVE_REQUESTS = Gauge(
    "model_serving_active_requests",
    "Number of requests currently being processed",
)

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of predictions made",
    labelnames=["model_variant", "predicted_label"],
)

TOKENIZATION_LATENCY = Histogram(
    "tokenization_duration_seconds",
    "Time spent on tokenization in seconds",
    labelnames=["model_variant"],
    buckets=(
        0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
    ),
)


# ---------------------------------------------------------
# Model loading & lifespan
# ---------------------------------------------------------

tokenizer = None
session = None


def _load_model():
    """Load tokenizer and ONNX session at startup."""
    global tokenizer, session

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

    print(f"Loading {MODEL_VARIANT.upper()} model...")
    session = ort.InferenceSession(
        MODEL_PATHS[MODEL_VARIANT],
        providers=["CPUExecutionProvider"],
    )

    print(
        f"Loaded {MODEL_VARIANT.upper()}: "
        f"{MODEL_PATHS[MODEL_VARIANT]}"
    )

    # Publish model info to Prometheus
    MODEL_INFO.info({
        "model_name": MODEL_NAME,
        "variant": MODEL_VARIANT,
        "onnx_path": MODEL_PATHS[MODEL_VARIANT],
        "runtime": "onnxruntime",
    })


def _warmup_model():
    """Run warmup inference passes to stabilize latency."""
    print("Warming up model...")

    encoded = tokenizer(
        "This is a warmup request.",
        return_tensors="np",
        truncation=True,
        padding=True,
    )

    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    for i in range(5):
        start = perf_counter()
        session.run(None, inputs)
        end = perf_counter()
        print(f"Warm-up {i + 1}: {(end - start) * 1000:.3f} ms")

    print("Warm-up complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    _load_model()
    _warmup_model()
    print("API initialization complete.")
    yield
    print("API shutting down.")


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="Model Serving Benchmark API",
    description="ONNX Runtime model serving benchmark with Prometheus metrics",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Prometheus middleware
# ---------------------------------------------------------

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    """Track HTTP request duration and active request count."""
    ACTIVE_REQUESTS.inc()
    start = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code="500",
        ).observe(perf_counter() - start)
        ACTIVE_REQUESTS.dec()
        raise

    duration = perf_counter() - start
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).observe(duration)
    ACTIVE_REQUESTS.dec()

    return response


# ---------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    model: str
    label: str
    confidence: float
    latency_ms: float


# ---------------------------------------------------------
# Synchronous inference (runs in thread pool)
# ---------------------------------------------------------

def _run_inference(text: str) -> dict:
    """
    Tokenize + run ONNX inference synchronously.
    This function is offloaded to a thread pool so it
    does not block the async event loop.
    """
    # Tokenization
    tok_start = perf_counter()
    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
    )
    tok_end = perf_counter()

    TOKENIZATION_LATENCY.labels(
        model_variant=MODEL_VARIANT,
    ).observe(tok_end - tok_start)

    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    # Model inference
    inf_start = perf_counter()
    outputs = session.run(None, inputs)
    inf_end = perf_counter()

    inference_seconds = inf_end - inf_start

    INFERENCE_LATENCY.labels(
        model_variant=MODEL_VARIANT,
    ).observe(inference_seconds)

    # Post-processing
    logits = outputs[0][0]
    probabilities = np.exp(logits - np.max(logits))
    probabilities = probabilities / np.sum(probabilities)
    predicted_class = int(np.argmax(probabilities))

    label = LABELS[predicted_class]

    PREDICTION_COUNTER.labels(
        model_variant=MODEL_VARIANT,
        predicted_label=label,
    ).inc()

    return {
        "model": MODEL_VARIANT,
        "label": label,
        "confidence": float(probabilities[predicted_class]),
        "latency_ms": round(inference_seconds * 1000, 3),
    }


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "Model Serving Benchmark API",
        "status": "running",
        "model": MODEL_VARIANT,
        "version": "2.0.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_VARIANT,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Run sentiment analysis inference.

    The blocking ONNX session.run() call is offloaded to
    a thread pool via asyncio.to_thread(), keeping the
    FastAPI event loop responsive under concurrent load.
    """
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    result = await asyncio.to_thread(
        _run_inference, request.text
    )

    return result


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
