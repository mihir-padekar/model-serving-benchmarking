import numpy as np
import onnxruntime as ort
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_MODEL = "models/fp32/model.onnx"


def main():
    print("Loading tokenizer and PyTorch model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pytorch_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )
    pytorch_model.eval()

    print("Loading ONNX model...")

    session = ort.InferenceSession(
        ONNX_MODEL,
        providers=["CPUExecutionProvider"],
    )

    print("\nONNX inputs:")
    for input_info in session.get_inputs():
        print(
            f"  {input_info.name}: "
            f"shape={input_info.shape}, "
            f"type={input_info.type}"
        )

    print("\nONNX outputs:")
    for output_info in session.get_outputs():
        print(
            f"  {output_info.name}: "
            f"shape={output_info.shape}, "
            f"type={output_info.type}"
        )

    texts = [
        "This product is excellent and works perfectly.",
        "I am very disappointed with the quality.",
        "The service was fast and reliable.",
        "The package arrived damaged and late.",
        "Absolutely fantastic experience.",
    ]

    print("\nComparing predictions...\n")

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        # -------------------------
        # PyTorch inference
        # -------------------------

        with torch.no_grad():
            pytorch_output = pytorch_model(**inputs)

        pytorch_logits = pytorch_output.logits.numpy()
        pytorch_prediction = int(np.argmax(pytorch_logits, axis=-1)[0])

        # -------------------------
        # ONNX inference
        # -------------------------

        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }

        onnx_output = session.run(
            None,
            onnx_inputs,
        )

        onnx_logits = onnx_output[0]
        onnx_prediction = int(np.argmax(onnx_logits, axis=-1)[0])

        # -------------------------
        # Compare
        # -------------------------

        max_difference = np.max(
            np.abs(pytorch_logits - onnx_logits)
        )

        pytorch_label = pytorch_model.config.id2label[
            pytorch_prediction
        ]

        onnx_label = pytorch_model.config.id2label[
            onnx_prediction
        ]

        print(f"Text: {text}")
        print(f"PyTorch prediction: {pytorch_label}")
        print(f"ONNX prediction:    {onnx_label}")
        print(f"Max logit difference: {max_difference:.8f}")

        if pytorch_prediction == onnx_prediction:
            print("Result: MATCH")
        else:
            print("Result: MISMATCH")

        print("-" * 70)


if __name__ == "__main__":
    main()