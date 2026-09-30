import os
import time
import statistics
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

MODELS = {
    "FP32": "models/fp32/model.onnx",
    "INT8": "models/int8/model.onnx",
}

WARMUP_RUNS = 50
BENCHMARK_RUNS = 500

TEXTS = [
    "This product is excellent and works perfectly.",
    "I am very disappointed with the quality.",
    "The service was fast and reliable.",
    "The package arrived damaged and late.",
    "Absolutely fantastic experience.",
    "The product is okay, nothing special.",
    "I would definitely recommend this to everyone.",
    "This was a terrible experience.",
    "The quality is acceptable for the price.",
    "I regret buying this product.",
]


def load_inputs():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenizer(
        TEXTS,
        return_tensors="np",
        truncation=True,
        padding=True,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def get_model_size_mb(model_path):
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def benchmark_model(model_name, model_path, inputs):
    print("\n" + "=" * 70)
    print(f"Benchmarking {model_name}")
    print("=" * 70)

    print("Loading ONNX Runtime session...")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    print("Running warm-up...")

    for _ in range(WARMUP_RUNS):
        session.run(None, inputs)

    print(f"Running {BENCHMARK_RUNS} benchmark iterations...")

    latencies_ms = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        session.run(None, inputs)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)

    latencies_ms = np.array(latencies_ms)

    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)

    mean_latency = np.mean(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)

    total_time_seconds = np.sum(latencies_ms) / 1000

    throughput = BENCHMARK_RUNS / total_time_seconds

    model_size_mb = get_model_size_mb(model_path)

    results = {
        "model": model_name,
        "model_size_mb": model_size_mb,
        "mean_ms": mean_latency,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "throughput": throughput,
    }

    print("\nResults:")
    print(f"Model size:       {model_size_mb:.2f} MB")
    print(f"Mean latency:     {mean_latency:.2f} ms")
    print(f"P50 latency:      {p50:.2f} ms")
    print(f"P95 latency:      {p95:.2f} ms")
    print(f"P99 latency:      {p99:.2f} ms")
    print(f"Min latency:      {min_latency:.2f} ms")
    print(f"Max latency:      {max_latency:.2f} ms")
    print(f"Throughput:       {throughput:.2f} runs/sec")

    return results


def print_comparison(fp32, int8):
    print("\n\n")
    print("=" * 80)
    print("FP32 vs INT8 COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<20}"
        f"{'FP32':>15}"
        f"{'INT8':>15}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    metrics = [
        ("Model size (MB)", "model_size_mb"),
        ("Mean latency (ms)", "mean_ms"),
        ("P50 latency (ms)", "p50_ms"),
        ("P95 latency (ms)", "p95_ms"),
        ("P99 latency (ms)", "p99_ms"),
        ("Throughput (runs/s)", "throughput"),
    ]

    for label, key in metrics:
        fp32_value = fp32[key]
        int8_value = int8[key]

        change_percent = (
            (int8_value - fp32_value)
            / fp32_value
        ) * 100

        print(
            f"{label:<20}"
            f"{fp32_value:>15.2f}"
            f"{int8_value:>15.2f}"
            f"{change_percent:>14.2f}%"
        )

    print("=" * 80)


def main():
    print("Loading benchmark inputs...")

    inputs = load_inputs()

    print("\nInput shape:")
    print(f"input_ids:      {inputs['input_ids'].shape}")
    print(f"attention_mask: {inputs['attention_mask'].shape}")

    fp32_results = benchmark_model(
        "FP32",
        MODELS["FP32"],
        inputs,
    )

    int8_results = benchmark_model(
        "INT8",
        MODELS["INT8"],
        inputs,
    )

    print_comparison(
        fp32_results,
        int8_results,
    )


if __name__ == "__main__":
    main()