import time

import requests
import streamlit as st
import pandas as pd
import statistics
import numpy as np

FP32_API_URL = "http://localhost:8000"
INT8_API_URL = "http://localhost:8001"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.title("⚙️ Model Optimization Lab")
st.caption(
    "Interactive benchmarking and optimization analysis for "
    "DistilBERT inference using ONNX Runtime."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def check_health(api_url):
    """Check whether a FastAPI model server is running."""

    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=3,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def predict(api_url, text):
    """Send text to a model server."""

    response = requests.post(
        f"{api_url}/predict",
        json={"text": text},
        timeout=30,
    )

    response.raise_for_status()

    return response.json()

def run_benchmark(api_url, requests_count):
    """Run a live sequential benchmark against a model server."""

    texts = [
        "This product is excellent.",
        "I regret buying this product.",
        "The quality is amazing and I love it.",
        "This was a terrible purchase.",
        "Absolutely fantastic experience.",
    ]

    # Warm-up requests
    for i in range(10):
        predict(api_url, texts[i % len(texts)])

    latencies = []

    benchmark_start = time.perf_counter()

    progress = st.progress(0)

    for i in range(requests_count):
        text = texts[i % len(texts)]

        start = time.perf_counter()

        result = predict(api_url, text)

        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        progress.progress((i + 1) / requests_count)

    benchmark_end = time.perf_counter()

    total_time = benchmark_end - benchmark_start

    return {
        "mean": statistics.mean(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "throughput": requests_count / total_time,
        "latencies": latencies,
    }

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("⚡ Model Optimization Lab")

st.markdown(
    """
Compare an FP32 and INT8 DistilBERT model running through
ONNX Runtime, FastAPI and Docker.

This dashboard performs **real inference against the running API**.
"""
)


# ---------------------------------------------------------
# Server status
# ---------------------------------------------------------

health = check_health(FP32_API_URL)

if health:

    model_loaded = health.get(
        "model_loaded",
        "unknown"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("● API ONLINE")

    with col2:
        st.metric(
            "Model Loaded",
            model_loaded.upper(),
        )

    with col3:
        st.metric(
            "Endpoint",
            "/predict",
        )

else:

    st.error(
        "API server is offline. Start the Docker container on port 8000."
    )

    st.code(
        "docker run --rm -p 8000:8000 "
        "-e MODEL_VARIANT=int8 model-serving-benchmark",
        language="powershell",
    )

    st.stop()


st.divider()


# ---------------------------------------------------------
# Live inference
# ---------------------------------------------------------

st.header("🧪 Live Inference")

st.write(
    "Enter any sentence and send it through the actual "
    f"{model_loaded.upper()} model."
)

text_input = st.text_area(
    "Input text",
    value="This product is surprisingly good.",
    height=120,
)


if st.button(
    "Run Inference",
    type="primary",
    use_container_width=True,
):

    if not text_input.strip():

        st.warning("Please enter some text.")

    else:

        try:

            result = predict(
            FP32_API_URL,
            text_input,
        )

            label = result.get(
                "label",
                "UNKNOWN"
            )

            confidence = result.get(
                "confidence",
                0
            )

            model_latency = result.get(
                "latency_ms",
                0
            )

            client_latency = result.get(
                "client_latency_ms",
                0
            )


            # Store request history

            st.session_state.history.append(
                {
                    "timestamp": pd.Timestamp.now(),
                    "model": model_loaded.upper(),
                    "text": text_input,
                    "prediction": label,
                    "confidence": confidence,
                    "model_latency_ms": model_latency,
                    "client_latency_ms": client_latency,
                }
            )


            # Results

            st.subheader("Prediction")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Prediction",
                    label,
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%",
                )

            with col3:

                st.metric(
                    "Model Latency",
                    f"{model_latency:.2f} ms",
                )

            with col4:

                st.metric(
                    "Client Latency",
                    f"{client_latency:.2f} ms",
                )


            st.success(
                f"{label} — {confidence * 100:.2f}% confidence"
            )


        except requests.RequestException as error:

            st.error(
                f"API request failed: {error}"
            )


st.divider()

# ---------------------------------------------------------
# FP32 vs INT8 comparison
# ---------------------------------------------------------

st.header("⚡ FP32 vs INT8 Live Comparison")

st.write(
    "Send the exact same input to both model-serving containers "
    "and compare their predictions and latency."
)

comparison_text = st.text_area(
    "Comparison input",
    value="This product is surprisingly good.",
    height=120,
    key="comparison_text",
)


if st.button(
    "Compare FP32 vs INT8",
    type="primary",
    use_container_width=True,
):

    if not comparison_text.strip():

        st.warning("Please enter some text.")

    else:

        fp32_health = check_health(FP32_API_URL)
        int8_health = check_health(INT8_API_URL)

        if not fp32_health:

            st.error(
                "FP32 server is not available on port 8000."
            )

        elif not int8_health:

            st.error(
                "INT8 server is not available on port 8001."
            )

        else:

            try:

                fp32_result = predict(
                    FP32_API_URL,
                    comparison_text,
                )

                int8_result = predict(
                    INT8_API_URL,
                    comparison_text,
                )


                # -------------------------------------------------
                # Results
                # -------------------------------------------------

                col1, col2 = st.columns(2)


                with col1:

                    st.subheader("FP32")

                    st.metric(
                        "Prediction",
                        fp32_result["label"],
                    )

                    st.metric(
                        "Confidence",
                        f"{fp32_result['confidence'] * 100:.2f}%",
                    )

                    st.metric(
                        "Model Latency",
                        f"{fp32_result['latency_ms']:.2f} ms",
                    )


                with col2:

                    st.subheader("INT8")

                    st.metric(
                        "Prediction",
                        int8_result["label"],
                    )

                    st.metric(
                        "Confidence",
                        f"{int8_result['confidence'] * 100:.2f}%",
                    )

                    st.metric(
                        "Model Latency",
                        f"{int8_result['latency_ms']:.2f} ms",
                    )


                # -------------------------------------------------
                # Comparison
                # -------------------------------------------------

                fp32_latency = fp32_result["latency_ms"]
                int8_latency = int8_result["latency_ms"]


                latency_change = (
                    (fp32_latency - int8_latency)
                    / fp32_latency
                    * 100
                )


                st.subheader("Optimization Result")


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Latency Difference",
                        f"{abs(fp32_latency - int8_latency):.2f} ms",
                    )


                with col2:

                    st.metric(
                        "INT8 Latency Change",
                        f"{latency_change:.1f}%",
                    )


                with col3:

                    if fp32_result["label"] == int8_result["label"]:

                        st.success(
                            "Predictions Match"
                        )

                    else:

                        st.warning(
                            "Predictions Differ"
                        )


                # Confidence difference

                confidence_difference = abs(
                    fp32_result["confidence"]
                    - int8_result["confidence"]
                )


                st.write(
                    f"Confidence difference: "
                    f"{confidence_difference * 100:.4f} percentage points"
                )


            except requests.RequestException as error:

                st.error(
                    f"Comparison failed: {error}"
                )
# ---------------------------------------------------------
# Model analysis / optimization pipeline
# ---------------------------------------------------------

st.divider()

st.header("🔬 Model Analysis & Optimization Pipeline")

st.write(
    "This section shows how the original FP32 model was transformed "
    "into an optimized INT8 ONNX model and validated before serving."
)

# Model overview
st.subheader("Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Base Model", "DistilBERT")

with col2:
    st.metric("Task", "Sentiment Analysis")

with col3:
    st.metric("Classes", "2")

with col4:
    st.metric("Runtime", "ONNX Runtime")


# Model size comparison
st.subheader("Model Size")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "FP32 Model",
        "255.53 MB",
    )

with col2:
    st.metric(
        "INT8 Model",
        "64.24 MB",
    )

with col3:
    st.metric(
        "Storage Reduction",
        "74.86%",
    )


# Optimization pipeline
st.subheader("Optimization Workflow")

pipeline = [
    ("1", "Hugging Face Model", "DistilBERT SST-2"),
    ("2", "PyTorch Baseline", "Original FP32 model"),
    ("3", "ONNX Export", "FP32 ONNX model"),
    ("4", "FP32 Validation", "PyTorch vs ONNX"),
    ("5", "INT8 Quantization", "Dynamic weight quantization"),
    ("6", "INT8 Validation", "FP32 vs INT8 predictions"),
    ("7", "SST-2 Evaluation", "Accuracy and F1"),
    ("8", "Model Serving", "FastAPI + ONNX Runtime"),
    ("9", "Deployment", "Docker"),
]

for step, title, description in pipeline:
    col1, col2, col3 = st.columns([1, 3, 5])

    with col1:
        st.markdown(f"### {step}")

    with col2:
        st.markdown(f"**{title}**")

    with col3:
        st.write(description)


# Validation results
st.subheader("Validation Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### FP32")

    st.metric("Accuracy", "91.06%")
    st.metric("F1 Score", "0.9137")

with col2:
    st.markdown("### INT8")

    st.metric("Accuracy", "90.48%")
    st.metric("F1 Score", "0.9087")


st.info(
    "INT8 preserved the same sentiment predictions on the sampled "
    "validation checks while reducing model size and inference cost. "
    "On the full SST-2 validation set, INT8 accuracy was 0.57 percentage "
    "points lower than FP32."
)


# Optimization tradeoff
st.subheader("Optimization Tradeoff")

tradeoff_df = pd.DataFrame(
    {
        "Metric": [
            "Model Size",
            "Mean Model Latency",
            "API Throughput",
            "Validation Accuracy",
            "Validation F1",
        ],
        "FP32": [
            "255.53 MB",
            "15.32 ms",
            "39.09 req/s",
            "91.06%",
            "0.9137",
        ],
        "INT8": [
            "64.24 MB",
            "6.62 ms",
            "60.56 req/s",
            "90.48%",
            "0.9087",
        ],
    }
)

st.dataframe(
    tradeoff_df,
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "Benchmark results are from the frozen Docker serving benchmark: "
    "3 rounds × 200 measured requests per model, with 20 warm-up "
    "requests excluded from each round."
)

# ---------------------------------------------------------
# Benchmark Lab
# ---------------------------------------------------------

st.divider()

st.header("🏃 Benchmark Lab")

st.write(
    "Run a live sequential benchmark against the Dockerized "
    "FP32 or INT8 model server."
)

col1, col2 = st.columns(2)

with col1:
    benchmark_model = st.selectbox(
    "Model",
    ["FP32", "INT8", "Both"],
    key="benchmark_model",
)

with col2:
    benchmark_requests = st.selectbox(
        "Measured Requests",
        [25, 50, 100, 200],
        index=1,
        key="benchmark_requests",
    )

if benchmark_model == "FP32":
    benchmark_urls = {"FP32": FP32_API_URL}

elif benchmark_model == "INT8":
    benchmark_urls = {"INT8": INT8_API_URL}

else:
    benchmark_urls = {
        "FP32": FP32_API_URL,
        "INT8": INT8_API_URL,
    }

if st.button(
    "▶ Run Live Benchmark",
    type="primary",
    use_container_width=True,
):

    benchmark_results = {}

    for model_name, api_url in benchmark_urls.items():

        health = check_health(api_url)

        if not health:
            st.error(
                f"{model_name} server is not available."
            )
            continue

        with st.spinner(
            f"Benchmarking {model_name}..."
        ):

            try:

                benchmark_results[model_name] = run_benchmark(
                    api_url,
                    benchmark_requests,
                )

            except requests.RequestException as error:

                st.error(
                    f"{model_name} benchmark failed: {error}"
                )

    if benchmark_results:

        st.session_state.benchmark_results = {
            "requests": benchmark_requests,
            "results": benchmark_results,
        }

# ---------------------------------------------------------
# Display benchmark results
# ---------------------------------------------------------

if st.session_state.benchmark_results is not None:

    benchmark_data = st.session_state.benchmark_results
    results = benchmark_data["results"]

    st.subheader("Latest Live Benchmark")

    # -----------------------------------------------------
    # Single-model result
    # -----------------------------------------------------

    if len(results) == 1:

        model_name = list(results.keys())[0]
        result = results[model_name]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Mean",
                f"{result['mean']:.2f} ms",
            )

        with col2:
            st.metric(
                "P50",
                f"{result['p50']:.2f} ms",
            )

        with col3:
            st.metric(
                "P95",
                f"{result['p95']:.2f} ms",
            )

        with col4:
            st.metric(
                "P99",
                f"{result['p99']:.2f} ms",
            )

        with col5:
            st.metric(
                "Throughput",
                f"{result['throughput']:.2f} req/s",
            )

    # -----------------------------------------------------
    # FP32 vs INT8 comparison
    # -----------------------------------------------------

    else:

        fp32 = results["FP32"]
        int8 = results["INT8"]

        st.subheader("⚡ FP32 vs INT8")

        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Mean Latency",
                    "P50 Latency",
                    "P95 Latency",
                    "P99 Latency",
                    "Throughput",
                ],
                "FP32": [
                    f"{fp32['mean']:.2f} ms",
                    f"{fp32['p50']:.2f} ms",
                    f"{fp32['p95']:.2f} ms",
                    f"{fp32['p99']:.2f} ms",
                    f"{fp32['throughput']:.2f} req/s",
                ],
                "INT8": [
                    f"{int8['mean']:.2f} ms",
                    f"{int8['p50']:.2f} ms",
                    f"{int8['p95']:.2f} ms",
                    f"{int8['p99']:.2f} ms",
                    f"{int8['throughput']:.2f} req/s",
                ],
            }
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        # Optimization impact
        mean_change = (
            (fp32["mean"] - int8["mean"])
            / fp32["mean"]
            * 100
        )

        p50_change = (
            (fp32["p50"] - int8["p50"])
            / fp32["p50"]
            * 100
        )

        p95_change = (
            (fp32["p95"] - int8["p95"])
            / fp32["p95"]
            * 100
        )

        throughput_change = (
            (int8["throughput"] - fp32["throughput"])
            / fp32["throughput"]
            * 100
        )

        st.subheader("Optimization Impact")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean Latency",
                f"{mean_change:.1f}%",
                delta="lower with INT8",
            )

        with col2:
            st.metric(
                "P50 Latency",
                f"{p50_change:.1f}%",
                delta="lower with INT8",
            )

        with col3:
            st.metric(
                "P95 Latency",
                f"{p95_change:.1f}%",
                delta="lower with INT8",
            )

        with col4:
            st.metric(
                "Throughput",
                f"{throughput_change:.1f}%",
                delta="higher with INT8",
            )

        # Latency distributions
        st.subheader("Latency Distribution")

        latency_df = pd.DataFrame(
            {
                "FP32": pd.Series(fp32["latencies"]),
                "INT8": pd.Series(int8["latencies"]),
            }
        )

        st.line_chart(latency_df)

    st.caption(
        f"{benchmark_data['requests']} measured requests per model. "
        "10 warm-up requests were excluded from measurements."
    )
# ---------------------------------------------------------
# Request history
# ---------------------------------------------------------

st.header("📊 Live Request History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    display_df = history_df.copy()

    display_df["timestamp"] = (
        display_df["timestamp"]
        .dt.strftime("%H:%M:%S")
    )

    display_df["confidence"] = (
        display_df["confidence"] * 100
    ).round(2)

    display_df["model_latency_ms"] = (
        display_df["model_latency_ms"]
        .round(2)
    )

    display_df["client_latency_ms"] = (
        display_df["client_latency_ms"]
        .round(2)
    )

    display_df = display_df[
        [
            "timestamp",
            "model",
            "prediction",
            "confidence",
            "model_latency_ms",
            "client_latency_ms",
            "text",
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


    # Summary

    st.subheader("Session Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Requests",
            len(history_df),
        )

    with col2:

        st.metric(
            "Average Model Latency",
            f"{history_df['model_latency_ms'].mean():.2f} ms",
        )

    with col3:

        st.metric(
            "Average Client Latency",
            f"{history_df['client_latency_ms'].mean():.2f} ms",
        )

else:

    st.info(
        "No inference requests yet. Enter text above and run the model."
    )


st.divider()


# ---------------------------------------------------------
# Architecture
# ---------------------------------------------------------

st.header("🏗️ Current Serving Architecture")

st.code(
    """
User
 │
 ▼
Streamlit Dashboard
 │
 │ HTTP POST /predict
 ▼
FastAPI
 │
 ▼
ONNX Runtime
 │
 ▼
DistilBERT
 │
 ├── FP32 ONNX
 │
 └── INT8 ONNX
        │
        ▼
     Prediction
""",
    language="text",
)


st.caption(
    "Inference is performed locally through the Dockerized FastAPI server."
)