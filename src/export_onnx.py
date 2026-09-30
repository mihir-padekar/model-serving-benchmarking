import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "models/fp32"


def main():
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text = "This product is excellent and works perfectly."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    output_path = os.path.join(OUTPUT_DIR, "model.onnx")

    print("Exporting model to ONNX...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
            ),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "logits": {
                    0: "batch_size",
                },
            },
            opset_version=18,
            dynamo=False,
        )

    print("\nModel exported successfully:")
    print(output_path)


if __name__ == "__main__":
    main()