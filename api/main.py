from time import perf_counter

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

MODEL_PATHS = {
    "fp32": "models/fp32/model.onnx",
    "int8": "models/int8/model.onnx",
}

LABELS = {
    0: "NEGATIVE",
    1: "POSITIVE",
}


app = FastAPI(
    title="Model Serving Benchmark API",
    description="ONNX Runtime model serving benchmark",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str


# Model is selected through an environment variable.
# Default = INT8
import os

MODEL_NAME_SELECTED = os.getenv("MODEL_VARIANT", "int8").lower()

if MODEL_NAME_SELECTED not in MODEL_PATHS:
    raise ValueError(
        f"Invalid MODEL_VARIANT: {MODEL_NAME_SELECTED}. "
        f"Use 'fp32' or 'int8'."
    )


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "models/tokenizer"
)

print(f"Loading {MODEL_NAME_SELECTED.upper()} model...")

session = ort.InferenceSession(
    MODEL_PATHS[MODEL_NAME_SELECTED],
    providers=["CPUExecutionProvider"],
)

print(
    f"Loaded {MODEL_NAME_SELECTED.upper()}: "
    f"{MODEL_PATHS[MODEL_NAME_SELECTED]}"
)

print("API initialization complete.")

def warmup_model():
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

        print(
            f"Warm-up {i + 1}: "
            f"{(end - start) * 1000:.3f} ms"
        )

    print("Warm-up complete.")

warmup_model()

@app.get("/")
def root():
    return {
        "service": "Model Serving Benchmark API",
        "status": "running",
        "model": MODEL_NAME_SELECTED,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_NAME_SELECTED,
    }


@app.post("/predict")
def prediction(request: PredictionRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    encoded = tokenizer(
        request.text,
        return_tensors="np",
        truncation=True,
        padding=True,
    )

    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    start = perf_counter()

    outputs = session.run(None, inputs)

    end = perf_counter()

    logits = outputs[0][0]

    probabilities = np.exp(
        logits - np.max(logits)
    )

    probabilities = (
        probabilities / np.sum(probabilities)
    )

    predicted_class = int(
        np.argmax(probabilities)
    )

    return {
        "model": MODEL_NAME_SELECTED,
        "label": LABELS[predicted_class],
        "confidence": float(
            probabilities[predicted_class]
        ),
        "latency_ms": round(
            (end - start) * 1000,
            3,
        ),
    }