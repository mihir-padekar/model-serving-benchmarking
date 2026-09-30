import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

MODELS = {
    "FP32": "models/fp32/model.onnx",
    "INT8": "models/int8/model.onnx",
}

TEXT = "This product is excellent and works perfectly."

WARMUP_RUNS = 50
BENCHMARK_RUNS = 500


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def benchmark_model(model_name, model_path, inputs):

    print("\n" + "=" * 70)
    print(f"Batch Size 1 Benchmark: {model_name}")
    print("=" * 70)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    print(f"Model size: {get_model_size_mb(model_path):.2f} MB")

    # Warmup
    print(f"Running {WARMUP_RUNS} warmup runs...")

    for _ in range(WARMUP_RUNS):
        session.run(None, inputs)

    # Benchmark
    print(f"Running {BENCHMARK_RUNS} benchmark runs...")

    latencies = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        session.run(None, inputs)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    latencies = np.array(latencies)

    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    throughput = 1000 / mean_latency

    print("\nResults")
    print("-" * 50)

    print(f"Mean latency: {mean_latency:.2f} ms")
    print(f"P50 latency:  {p50_latency:.2f} ms")
    print(f"P95 latency:  {p95_latency:.2f} ms")
    print(f"P99 latency:  {p99_latency:.2f} ms")
    print(f"Min latency:  {min_latency:.2f} ms")
    print(f"Max latency:  {max_latency:.2f} ms")
    print(f"Throughput:   {throughput:.2f} requests/sec")

    return {
        "model": model_name,
        "model_size_mb": get_model_size_mb(model_path),
        "mean": mean_latency,
        "p50": p50_latency,
        "p95": p95_latency,
        "p99": p99_latency,
        "min": min_latency,
        "max": max_latency,
        "throughput": throughput,
    }


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenizer(
        TEXT,
        return_tensors="np",
        truncation=True,
        padding=True,
    )

    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    results = {}

    for model_name, model_path in MODELS.items():

        results[model_name] = benchmark_model(
            model_name,
            model_path,
            inputs,
        )

    fp32 = results["FP32"]
    int8 = results["INT8"]

    print("\n\n")
    print("=" * 80)
    print("BATCH SIZE 1 COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<25}"
        f"{'FP32':>15}"
        f"{'INT8':>15}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    metrics = [
        ("Model size (MB)", "model_size_mb"),
        ("Mean latency (ms)", "mean"),
        ("P50 latency (ms)", "p50"),
        ("P95 latency (ms)", "p95"),
        ("P99 latency (ms)", "p99"),
        ("Throughput (req/s)", "throughput"),
    ]

    for label, key in metrics:

        fp32_value = fp32[key]
        int8_value = int8[key]

        change = ((int8_value - fp32_value) / fp32_value) * 100

        print(
            f"{label:<25}"
            f"{fp32_value:>15.2f}"
            f"{int8_value:>15.2f}"
            f"{change:>14.2f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()