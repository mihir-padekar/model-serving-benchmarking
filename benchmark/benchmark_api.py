import requests
import time
import statistics
import numpy as np
URL = "http://localhost:8001/predict"

MODEL = "int8"   # Change to "fp32" for FP32

TEXTS = [
    "This product is excellent.",
    "I regret buying this product.",
    "The quality is amazing and I love it.",
    "This was a terrible purchase.",
    "Absolutely fantastic experience.",
]

WARMUP_REQUESTS = 20
BENCHMARK_REQUESTS = 200
ROUNDS = 3
STARTUP_WARMUPS = 5


def run_round():

    latencies = []
    model_latencies = []
    predictions = {
        "POSITIVE": 0,
        "NEGATIVE": 0
    }

    print(
        f"Server startup warm-up: {STARTUP_WARMUPS} requests "
        f"(handled by FastAPI)"
    )

    print(
        f"Benchmark warm-up: {WARMUP_REQUESTS} requests "
        f"(excluded from measurements)"
    )

    for i in range(WARMUP_REQUESTS):
        text = TEXTS[i % len(TEXTS)]

        response = requests.post(
            URL,
            json={"text": text}
        )

        response.raise_for_status()


    print(f"Running {BENCHMARK_REQUESTS} benchmark requests...")
    benchmark_start = time.perf_counter()
    for i in range(BENCHMARK_REQUESTS):

        text = TEXTS[i % len(TEXTS)]

        start = time.perf_counter()

        response = requests.post(
            URL,
            json={"text": text}
        )

        end = time.perf_counter()

        response.raise_for_status()

        data = response.json()

        api_latency = (end - start) * 1000

        model_latency = data["latency_ms"]

        latencies.append(api_latency)
        model_latencies.append(model_latency)

        predictions[data["label"]] += 1

    benchmark_end = time.perf_counter()

    total_benchmark_time = benchmark_end - benchmark_start

    latencies_sorted = sorted(latencies)

    mean_latency = statistics.mean(latencies)

    p50 = np.percentile(latencies, 50)

    p95 = np.percentile(latencies, 95)

    p99 = np.percentile(latencies, 99)

    mean_model_latency = statistics.mean(model_latencies)

    throughput = BENCHMARK_REQUESTS / total_benchmark_time

    return {
        "mean_api": mean_latency,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean_model": mean_model_latency,
        "throughput": throughput,
        "predictions": predictions
    }


print("=" * 70)
print(f"API Benchmark: {MODEL.upper()}")
print("=" * 70)

results = []

for round_number in range(1, ROUNDS + 1):

    print()
    print(f"ROUND {round_number}/{ROUNDS}")
    print("-" * 50)

    result = run_round()

    results.append(result)

    print()
    print(f"Mean API latency:    {result['mean_api']:.2f} ms")
    print(f"P50 API latency:     {result['p50']:.2f} ms")
    print(f"P95 API latency:     {result['p95']:.2f} ms")
    print(f"P99 API latency:     {result['p99']:.2f} ms")
    print(f"Mean model latency:  {result['mean_model']:.2f} ms")
    print(f"Throughput:          {result['throughput']:.2f} requests/sec")


def avg(key):
    return statistics.mean(r[key] for r in results)


def std(key):
    return statistics.stdev(r[key] for r in results)


print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\nModel: {MODEL.upper()}")
print(f"Rounds: {ROUNDS}")
print(f"Requests per round: {BENCHMARK_REQUESTS}")
print(f"Benchmark warm-up requests: {WARMUP_REQUESTS}")
print("Server startup warm-up: 5 requests")
print("Warm-up requests excluded from measurements")
print()
print("Metric                  Mean       Std Dev")
print("-" * 50)

print(
    f"API latency             "
    f"{avg('mean_api'):8.2f} ms   "
    f"{std('mean_api'):8.2f}"
)

print(
    f"P50 latency             "
    f"{avg('p50'):8.2f} ms   "
    f"{std('p50'):8.2f}"
)

print(
    f"P95 latency             "
    f"{avg('p95'):8.2f} ms   "
    f"{std('p95'):8.2f}"
)

print(
    f"P99 latency             "
    f"{avg('p99'):8.2f} ms   "
    f"{std('p99'):8.2f}"
)

print(
    f"Model latency           "
    f"{avg('mean_model'):8.2f} ms   "
    f"{std('mean_model'):8.2f}"
)

print(
    f"Throughput              "
    f"{avg('throughput'):8.2f} req/s   "
    f"{std('throughput'):8.2f}"
)

print()
print("Predictions per round:")

for i, result in enumerate(results, 1):

    print(
        f"Round {i}: "
        f"POSITIVE={result['predictions']['POSITIVE']}, "
        f"NEGATIVE={result['predictions']['NEGATIVE']}"
    )

print()
print("=" * 70)