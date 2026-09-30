import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
FP32_MODEL = "models/fp32/model.onnx"
INT8_MODEL = "models/int8/model.onnx"


def main():
    print("Loading tokenizer and reference model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    reference_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    print("Loading FP32 ONNX model...")
    fp32_session = ort.InferenceSession(
        FP32_MODEL,
        providers=["CPUExecutionProvider"],
    )

    print("Loading INT8 ONNX model...")
    int8_session = ort.InferenceSession(
        INT8_MODEL,
        providers=["CPUExecutionProvider"],
    )

    texts = [
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

    print("\nComparing FP32 vs INT8...\n")

    fp32_correct = 0
    int8_correct = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        fp32_logits = fp32_session.run(
            None,
            onnx_inputs
        )[0]

        int8_logits = int8_session.run(
            None,
            onnx_inputs
        )[0]

        fp32_prediction = int(np.argmax(fp32_logits, axis=-1)[0])
        int8_prediction = int(np.argmax(int8_logits, axis=-1)[0])

        fp32_label = reference_model.config.id2label[fp32_prediction]
        int8_label = reference_model.config.id2label[int8_prediction]

        max_difference = np.max(
            np.abs(fp32_logits - int8_logits)
        )

        prediction_match = fp32_prediction == int8_prediction

        if prediction_match:
            fp32_correct += 1
            int8_correct += 1

        print(f"Text: {text}")
        print(f"FP32 prediction: {fp32_label}")
        print(f"INT8 prediction: {int8_label}")
        print(f"Max logit difference: {max_difference:.6f}")
        print(f"Prediction match: {prediction_match}")
        print("-" * 70)

    print("\nSummary")
    print("=" * 70)
    print(f"FP32/INT8 matching predictions: {int8_correct}/{len(texts)}")
    print(
        f"Agreement: "
        f"{(int8_correct / len(texts)) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()