import os
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "models/fp32/model.onnx"
OUTPUT_DIR = "models/int8"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "model.onnx")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting INT8 dynamic quantization...")
    print(f"Input model:  {INPUT_MODEL}")
    print(f"Output model: {OUTPUT_MODEL}")

    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QInt8,
    )

    print("\nINT8 quantization completed successfully.")
    print(f"Quantized model: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()