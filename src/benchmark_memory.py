import os
import time
import gc
import psutil
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

MODELS = {
    "FP32": "models/fp32/model.onnx",
    "INT8": "models/int8/model.onnx",
}

TEXT = "This product is excellent and works perfectly."


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_model(model_name, model_path, inputs):

    print("\n" + "=" * 70)
    print(f"Memory benchmark: {model_name}")
    print("=" * 70)

    gc.collect()

    memory_before = get_process_memory_mb()

    print(f"Memory before loading: {memory_before:.2f} MB")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    memory_after_load = get_process_memory_mb()

    print(f"Memory after loading:  {memory_after_load:.2f} MB")

    # Warm-up
    for _ in range(20):
        session.run(None, inputs)

    memory_after_warmup = get_process_memory_mb()

    print(f"Memory after warmup:   {memory_after_warmup:.2f} MB")

    # Run inference repeatedly
    peak_memory = memory_after_warmup

    for _ in range(100):
        session.run(None, inputs)

        current_memory = get_process_memory_mb()

        peak_memory = max(
            peak_memory,
            current_memory,
        )

    print(f"Peak observed memory:  {peak_memory:.2f} MB")

    model_size = get_model_size_mb(model_path)

    print(f"Model file size:       {model_size:.2f} MB")

    model_memory_increase = (
        memory_after_load - memory_before
    )

    print(
        f"Process memory increase: "
        f"{model_memory_increase:.2f} MB"
    )

    return {
        "model": model_name,
        "model_size_mb": model_size,
        "memory_before_mb": memory_before,
        "memory_after_load_mb": memory_after_load,
        "memory_after_warmup_mb": memory_after_warmup,
        "peak_memory_mb": peak_memory,
        "model_memory_increase_mb": model_memory_increase,
    }


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

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

        results[model_name] = measure_model(
            model_name,
            model_path,
            inputs,
        )

    fp32 = results["FP32"]
    int8 = results["INT8"]

    print("\n\n")
    print("=" * 80)
    print("MEMORY COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<30}"
        f"{'FP32':>15}"
        f"{'INT8':>15}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    metrics = [
        ("Model size (MB)", "model_size_mb"),
        ("Memory after load (MB)", "memory_after_load_mb"),
        ("Peak process memory (MB)", "peak_memory_mb"),
        ("Model memory increase (MB)", "model_memory_increase_mb"),
    ]

    for label, key in metrics:

        fp32_value = fp32[key]
        int8_value = int8[key]

        change = (
            (int8_value - fp32_value)
            / fp32_value
        ) * 100

        print(
            f"{label:<30}"
            f"{fp32_value:>15.2f}"
            f"{int8_value:>15.2f}"
            f"{change:>14.2f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()