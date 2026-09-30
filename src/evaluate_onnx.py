import time
import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

MODELS = {
    "FP32": "models/fp32/model.onnx",
    "INT8": "models/int8/model.onnx",
}

BATCH_SIZE = 16


def evaluate_model(model_name, model_path, dataset, tokenizer):
    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    predictions = []
    actual_labels = []

    total_samples = len(dataset)

    start_time = time.perf_counter()

    for start in range(0, total_samples, BATCH_SIZE):
        batch = dataset[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch["sentence"],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128,
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        outputs = session.run(
            None,
            onnx_inputs,
        )

        logits = outputs[0]

        batch_predictions = np.argmax(
            logits,
            axis=-1,
        )

        predictions.extend(batch_predictions.tolist())
        actual_labels.extend(batch["label"])

    end_time = time.perf_counter()

    accuracy = accuracy_score(
        actual_labels,
        predictions,
    )

    f1 = f1_score(
        actual_labels,
        predictions,
    )

    total_time = end_time - start_time
    samples_per_second = total_samples / total_time

    print(f"Samples:             {total_samples}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Evaluation time:     {total_time:.2f} seconds")
    print(f"Evaluation throughput: {samples_per_second:.2f} samples/sec")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "evaluation_time": total_time,
        "throughput": samples_per_second,
    }


def print_comparison(fp32, int8):
    print("\n\n")
    print("=" * 70)
    print("ACCURACY / F1 COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<25}"
        f"{'FP32':>15}"
        f"{'INT8':>15}"
        f"{'Difference':>15}"
    )

    print("-" * 70)

    accuracy_difference = (
        int8["accuracy"] - fp32["accuracy"]
    ) * 100

    f1_difference = (
        int8["f1"] - fp32["f1"]
    )

    print(
        f"{'Accuracy':<25}"
        f"{fp32['accuracy'] * 100:>14.2f}%"
        f"{int8['accuracy'] * 100:>14.2f}%"
        f"{accuracy_difference:>14.2f} pp"
    )

    print(
        f"{'F1 Score':<25}"
        f"{fp32['f1']:>15.4f}"
        f"{int8['f1']:>15.4f}"
        f"{f1_difference:>15.4f}"
    )

    print("=" * 70)


def main():
    print("Loading SST-2 validation dataset...")

    dataset = load_dataset(
    "nyu-mll/glue",
    "sst2",
    split="validation",
    )

    print(f"Dataset size: {len(dataset)}")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    fp32_results = evaluate_model(
        "FP32",
        MODELS["FP32"],
        dataset,
        tokenizer,
    )

    int8_results = evaluate_model(
        "INT8",
        MODELS["INT8"],
        dataset,
        tokenizer,
    )

    print_comparison(
        fp32_results,
        int8_results,
    )


if __name__ == "__main__":
    main()