import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main():
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()

    texts = [
        "This product is excellent and works perfectly.",
        "I am very disappointed with the quality.",
        "The service was fast and reliable.",
        "The package arrived damaged and late.",
        "Absolutely fantastic experience.",
    ]

    print("\nRunning inference...\n")

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            start = time.perf_counter()

            outputs = model(**inputs)

            end = time.perf_counter()

            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

            label = model.config.id2label[predicted_class]

            latency_ms = (end - start) * 1000

            print(f"Text: {text}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Latency: {latency_ms:.2f} ms")
            print("-" * 60)


if __name__ == "__main__":
    main()