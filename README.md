# ⚙️ Model Serving Benchmark

**Production-grade ML model serving & optimization lab** — benchmarking DistilBERT inference through ONNX Runtime with FP32/INT8 quantization, async FastAPI serving, Prometheus observability, and a Streamlit dashboard.

**Live Demo - https://model-serving-benchmarking.streamlit.app/ **
---

## Architecture

```
                    ┌─────────────────────────┐
                    │   Streamlit Dashboard    │
                    │       (Port 8501)        │
                    └────────┬────────┬────────┘
                             │        │
                   HTTP POST │        │ HTTP POST
                    /predict │        │ /predict
                             ▼        ▼
              ┌──────────────┐  ┌──────────────┐
              │  FastAPI     │  │  FastAPI      │
              │  FP32 Server │  │  INT8 Server  │
              │  (Port 8000) │  │  (Port 8001)  │
              └──────┬───────┘  └──────┬────────┘
                     │                 │
                     │  /metrics       │  /metrics
                     ▼                 ▼
              ┌────────────────────────────────┐
              │         Prometheus              │
              │         (Port 9090)             │
              └────────────────────────────────┘
```

### Key Components

| Component | Technology | Port | Description |
|-----------|-----------|------|-------------|
| **API Server (FP32)** | FastAPI + ONNX Runtime | 8000 | Serves the full-precision ONNX model |
| **API Server (INT8)** | FastAPI + ONNX Runtime | 8001 | Serves the quantized INT8 ONNX model |
| **Dashboard** | Streamlit | 8501 | Interactive benchmarking & comparison UI |
| **Metrics** | Prometheus | 9090 | Scrapes and stores inference metrics |

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (v2+)
- ~1 GB disk space for model files

### Option A: Single Command via Python (Local Development)

Run all services together locally without Docker using the built-in fleet launcher:

```bash
python run.py
```
*(or on PowerShell: `./start.ps1`)*

This starts:
- **FP32 FastAPI API** on `http://localhost:8000`
- **INT8 FastAPI API** on `http://localhost:8001`
- **Streamlit Dashboard** on `http://localhost:8501`

Press `Ctrl+C` to cleanly stop all services.

---

### Option B: Single Command via Docker Compose (Production Fleet)

Launch the containerized fleet including Prometheus:

```bash
docker compose up --build
```

This builds and starts all four services:

| Service | URL |
|---------|-----|
| FP32 API | [http://localhost:8000](http://localhost:8000) |
| INT8 API | [http://localhost:8001](http://localhost:8001) |
| Dashboard | [http://localhost:8501](http://localhost:8501) |
| Prometheus | [http://localhost:9090](http://localhost:9090) |

### Stop Everything

```bash
docker compose down
```

To also remove the Prometheus data volume:

```bash
docker compose down -v
```

---

## API Endpoints

### `GET /` — Service Info

```bash
curl http://localhost:8000/
```

```json
{
  "service": "Model Serving Benchmark API",
  "status": "running",
  "model": "fp32",
  "version": "2.0.0"
}
```

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": "fp32"
}
```

### `POST /predict` — Run Inference

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is excellent!"}'
```

```json
{
  "model": "fp32",
  "label": "POSITIVE",
  "confidence": 0.9998,
  "latency_ms": 12.345
}
```

### `GET /metrics` — Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Returns Prometheus-formatted metrics (see [Observability](#observability) below).

---

## Observability

### Prometheus Metrics Exposed

The `/metrics` endpoint on each API server exposes the following metrics:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_request_duration_seconds` | Histogram | `method`, `endpoint`, `status_code` | Total HTTP request duration |
| `model_inference_duration_seconds` | Histogram | `model_variant` | ONNX `session.run()` execution time |
| `tokenization_duration_seconds` | Histogram | `model_variant` | Tokenization time |
| `model_serving_active_requests` | Gauge | — | Currently in-flight requests |
| `model_predictions_total` | Counter | `model_variant`, `predicted_label` | Cumulative prediction count |
| `model_serving_info` | Info | `model_name`, `variant`, `onnx_path`, `runtime` | Static model metadata |

### Example Prometheus Queries (PromQL)

**P95 inference latency (last 5 minutes):**

```promql
histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))
```

**Request throughput (requests/sec):**

```promql
rate(model_predictions_total[1m])
```

**FP32 vs INT8 median latency comparison:**

```promql
histogram_quantile(0.5, rate(model_inference_duration_seconds_bucket{model_variant="fp32"}[5m]))
histogram_quantile(0.5, rate(model_inference_duration_seconds_bucket{model_variant="int8"}[5m]))
```

---

## Async Inference Architecture

The FastAPI server uses **async inference offloading** to keep the event loop responsive under concurrent load:

```
Incoming Request
      │
      ▼
 FastAPI async handler
      │
      ├─── Middleware: start timer, increment active_requests gauge
      │
      ▼
 asyncio.to_thread(_run_inference)
      │
      ├─── Thread Pool ──────────────────────────┐
      │    │  Tokenize input                     │
      │    │  session.run() (CPU-bound, blocking) │
      │    │  Softmax + argmax                    │
      │    │  Record Prometheus metrics           │
      │    └──────────────────────────────────────┘
      │
      ▼
 Return JSON response
      │
      ├─── Middleware: record duration, decrement active_requests
      ▼
 Client
```

**Why this matters:** Without offloading, a single `session.run()` call (~5–15ms) blocks the entire Python async event loop. Under 50+ concurrent requests, this causes severe head-of-line blocking and P99 latency spikes. `asyncio.to_thread()` delegates the CPU-bound work to a thread pool, allowing FastAPI to accept and queue new requests concurrently.

---

## Project Structure

```
model-serving-benchmark/
├── api/
│   └── main.py                 # FastAPI app (async + Prometheus)
├── benchmark/
│   └── benchmark_api.py        # Sequential API benchmark script
├── dashboard/
│   └── app.py                  # Streamlit optimization lab
├── models/
│   ├── fp32/model.onnx         # Full-precision ONNX model
│   ├── int8/model.onnx         # INT8 quantized ONNX model
│   └── tokenizer/              # HuggingFace tokenizer files
├── prometheus/
│   └── prometheus.yml          # Prometheus scrape configuration
├── src/
│   ├── export_onnx.py          # PyTorch → ONNX export
│   ├── quantize_onnx.py        # FP32 → INT8 dynamic quantization
│   ├── evaluate_onnx.py        # Accuracy evaluation on SST-2
│   ├── evaluate_pytorch.py     # PyTorch baseline evaluation
│   ├── benchmark_batch1.py     # Batch-size-1 latency benchmark
│   ├── benchmark_memory.py     # Memory profiling benchmark
│   ├── benchmark_onnx.py       # ONNX inference benchmark
│   ├── compare_onnx.py         # FP32 vs INT8 comparison
│   └── compare_int8.py         # INT8 analysis
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # API server image
├── Dockerfile.dashboard        # Streamlit dashboard image
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Running Individual Services

### API Server Only (Development)

```bash
# Using the local venv
pip install -r requirements.txt

# FP32 model
MODEL_VARIANT=fp32 uvicorn api.main:app --host 0.0.0.0 --port 8000

# INT8 model (in a separate terminal)
MODEL_VARIANT=int8 uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Dashboard Only (Development)

```bash
streamlit run dashboard/app.py
```

### Single Docker Container

```bash
# Build
docker build -t model-serving-benchmark .

# Run INT8 model
docker run --rm -p 8000:8000 -e MODEL_VARIANT=int8 model-serving-benchmark

# Run FP32 model
docker run --rm -p 8001:8000 -e MODEL_VARIANT=fp32 model-serving-benchmark
```

---

## Model Pipeline

The optimization pipeline converts a HuggingFace model to optimized ONNX:

```
HuggingFace Hub
      │
      ▼
  PyTorch Model (DistilBERT SST-2)
      │
      │  src/export_onnx.py
      ▼
  FP32 ONNX Model (~255 MB)
      │
      │  src/quantize_onnx.py
      ▼
  INT8 ONNX Model (~64 MB)     ← 4x smaller
```

### Re-export Models

```bash
# Export PyTorch → ONNX (FP32)
python src/export_onnx.py

# Quantize FP32 → INT8
python src/quantize_onnx.py
```

---

## Benchmarking Scripts

| Script | Purpose |
|--------|---------|
| `benchmark/benchmark_api.py` | End-to-end API latency (sequential HTTP requests) |
| `src/benchmark_batch1.py` | Raw ONNX batch-size-1 latency (no HTTP overhead) |
| `src/benchmark_onnx.py` | ONNX inference throughput benchmark |
| `src/benchmark_memory.py` | Process RSS memory profiling |
| `src/evaluate_onnx.py` | Accuracy & F1 on SST-2 validation set |
| `src/compare_onnx.py` | FP32 vs INT8 side-by-side comparison |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | DistilBERT (SST-2 sentiment classification) |
| Runtime | ONNX Runtime (CPU) |
| API Framework | FastAPI + Uvicorn |
| Async Offloading | `asyncio.to_thread()` |
| Metrics | Prometheus (`prometheus-client`) |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| Quantization | ONNX Runtime Dynamic INT8 |

---

## License

This project is for educational and benchmarking purposes.

