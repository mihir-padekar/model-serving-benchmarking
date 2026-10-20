"""
Model Serving Optimization Lab — Streamlit Dashboard
=====================================================
A production-ready, clean, and interactive dashboard for evaluating
and benchmarking DistilBERT ONNX model serving (FP32 vs INT8).

Deployment Modes Supported:
  1. Microservices Mode: Connects to live FastAPI model servers via HTTP.
  2. Cloud Standalone Mode: Falls back to cached in-process ONNX Runtime
     sessions when deployed on Streamlit Community Cloud without external APIs.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Optimization Lab | ONNX Serving Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 12px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(60, 130, 246, 0.4);
        box-shadow: 0 4px 12px rgba(60, 130, 246, 0.08);
        transition: all 0.2s ease-in-out;
    }
    
    /* Status Badge */
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 16px;
        background-color: rgba(34, 197, 94, 0.12);
        color: #16a34a;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .status-badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 16px;
        background-color: rgba(239, 68, 68, 0.12);
        color: #dc2626;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Result card container */
    .card-box {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background: rgba(128, 128, 128, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

# Helper to read from Streamlit secrets (cloud) or OS environment (local/docker)
def get_config_url(key, default):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration & Live Healthchecks
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ **Model Serving Lab**")
st.sidebar.caption("High-performance ONNX Runtime inference & quantization benchmarking.")
st.sidebar.divider()

# Server URL Inputs with environment variable / secrets defaults
with st.sidebar.expander("⚙️ Server Endpoints", expanded=False):
    fp32_url_input = st.text_input(
        "FP32 Server URL",
        value=get_config_url("FP32_API_URL", "http://localhost:8000"),
        help="FastAPI endpoint serving FP32 model (or remote cloud endpoint)",
    )
    int8_url_input = st.text_input(
        "INT8 Server URL",
        value=get_config_url("INT8_API_URL", "http://localhost:8001"),
        help="FastAPI endpoint serving INT8 quantized model (or remote cloud endpoint)",
    )

FP32_API_URL = fp32_url_input.rstrip("/")
INT8_API_URL = int8_url_input.rstrip("/")


# -----------------------------------------------------------------------------
# 4. In-Process ONNX Fallback (For standalone Streamlit Cloud deployments)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embedded_model(variant="int8"):
    """
    Cached in-process ONNX Runtime session and tokenizer.
    Enables zero-dependency standalone execution on Streamlit Community Cloud.
    """
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer_path = "models/tokenizer"
        model_path = f"models/{variant}/model.onnx"

        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            return None, None

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return tok, sess
    except Exception:
        return None, None


def run_embedded_inference(variant, text):
    """Fallback in-process inference if remote FastAPI is unreachable."""
    tok, sess = load_embedded_model(variant)
    if sess is None:
        raise RuntimeError(f"Embedded model 'models/{variant}/model.onnx' not available on disk.")

    enc = tok(text, return_tensors="np", truncation=True, padding=True)
    inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }

    start = time.perf_counter()
    outputs = sess.run(None, inputs)
    inf_time_ms = (time.perf_counter() - start) * 1000

    logits = outputs[0][0]
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred_class = int(np.argmax(probs))
    labels = {0: "NEGATIVE", 1: "POSITIVE"}

    return {
        "model": variant,
        "label": labels[pred_class],
        "confidence": float(probs[pred_class]),
        "latency_ms": round(inf_time_ms, 2),
        "client_latency_ms": round(inf_time_ms, 2),
        "is_embedded": True,
    }


# Helper: Check API Health & Response Time (with fallback to embedded)
def check_health(api_url, variant="int8"):
    """Check health and ping latency of a model server, or fallback to embedded ONNX."""
    start = time.perf_counter()
    try:
        response = requests.get(f"{api_url}/health", timeout=1.5)
        latency = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            return {"online": True, "mode": "api", "data": data, "ping_ms": round(latency, 1)}
    except requests.RequestException:
        pass

    # Check if local model file exists for in-process fallback
    model_path = f"models/{variant}/model.onnx"
    if os.path.exists(model_path):
        return {"online": True, "mode": "embedded", "data": {"model_loaded": variant}, "ping_ms": 0.0}

    return {"online": False, "mode": "offline", "data": None, "ping_ms": None}


def predict(api_url, text, variant="int8"):
    """Send prediction request to FastAPI, or automatically fallback to embedded engine."""
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"text": text},
            timeout=10,
        )
        if response.status_code == 200:
            client_latency = (time.perf_counter() - start) * 1000
            data = response.json()
            data["client_latency_ms"] = round(client_latency, 2)
            data["is_embedded"] = False
            return data
    except requests.RequestException:
        pass

    # Fallback to local embedded ONNX Runtime
    return run_embedded_inference(variant, text)


def run_benchmark(api_url, requests_count, variant="int8", progress_bar=None):
    """Execute live sequential inference benchmark."""
    sample_texts = [
        "This product is absolutely amazing, exceeded every expectation!",
        "Poor quality, broke on day one and customer service was unhelpful.",
        "A well-built product that delivers reasonable performance for its price.",
        "Terrible experience, would not recommend this to anyone.",
        "Extremely satisfied with the purchase and fast delivery.",
    ]

    # Warmup runs (5 requests)
    for i in range(5):
        try:
            predict(api_url, sample_texts[i % len(sample_texts)], variant=variant)
        except Exception:
            pass

    latencies = []
    model_latencies = []
    benchmark_start = time.perf_counter()

    for i in range(requests_count):
        text = sample_texts[i % len(sample_texts)]
        res = predict(api_url, text, variant=variant)
        latencies.append(res["client_latency_ms"])
        model_latencies.append(res.get("latency_ms", res["client_latency_ms"]))
        if progress_bar:
            progress_bar.progress((i + 1) / requests_count)

    total_time = time.perf_counter() - benchmark_start

    return {
        "count": requests_count,
        "mean": float(np.mean(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "throughput": round(requests_count / total_time, 2),
        "latencies": latencies,
        "model_latencies": model_latencies,
    }


# Live Status Indicators in Sidebar
fp32_health = check_health(FP32_API_URL, variant="fp32")
int8_health = check_health(INT8_API_URL, variant="int8")

st.sidebar.markdown("#### **Fleet Health**")
s_col1, s_col2 = st.sidebar.columns(2)
with s_col1:
    if fp32_health["online"]:
        if fp32_health["mode"] == "api":
            st.markdown(
                f'<div class="status-badge-online">● FP32 API<br><small>{fp32_health["ping_ms"]}ms ping</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge-online">● FP32 ONNX<br><small>in-process engine</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="status-badge-offline">● FP32 Offline</div>',
            unsafe_allow_html=True,
        )

with s_col2:
    if int8_health["online"]:
        if int8_health["mode"] == "api":
            st.markdown(
                f'<div class="status-badge-online">● INT8 API<br><small>{int8_health["ping_ms"]}ms ping</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge-online">● INT8 ONNX<br><small>in-process engine</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="status-badge-offline">● INT8 Offline</div>',
            unsafe_allow_html=True,
        )

# Notice if in-process fallback is active
if fp32_health["mode"] == "embedded" or int8_health["mode"] == "embedded":
    st.sidebar.info(
        "💡 **In-Process ONNX Active:** Running directly inside Streamlit without separate FastAPI servers."
    )

st.sidebar.divider()

# Sample Presets for Testing
st.sidebar.markdown("#### **Quick Presets**")
preset_options = {
    "Positive Review": "The build quality is incredible, fast, reliable, and exceeded expectations!",
    "Negative Review": "Utterly disappointed. Faulty product, terrible customer support, returned immediately.",
    "Subtle / Mixed": "It functions as described, but battery life is slightly below average.",
    "Short Edge Case": "Not bad at all.",
}
selected_preset = st.sidebar.selectbox("Load sample text", list(preset_options.keys()))
default_input_text = preset_options[selected_preset]

st.sidebar.divider()
st.sidebar.markdown("#### **Observability Links**")
st.sidebar.markdown(f"- 📈 [FP32 Prometheus Metrics]({FP32_API_URL}/metrics)")
st.sidebar.markdown(f"- 📈 [INT8 Prometheus Metrics]({INT8_API_URL}/metrics)")
st.sidebar.markdown("- 🔍 [Prometheus UI](http://localhost:9090)")

if st.sidebar.button("🧹 Clear Request History", use_container_width=True):
    st.session_state.history = []
    st.rerun()

# -----------------------------------------------------------------------------
# 5. Main App Layout & Tabs
# -----------------------------------------------------------------------------
st.title("⚡ Model Serving & Optimization Lab")
st.markdown(
    "Benchmark, inspect, and compare **FP32 vs. INT8 quantized DistilBERT** "
    "served via **ONNX Runtime, FastAPI, and Docker**."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧪 Live Inference",
        "🏃 Benchmark Suite",
        "🔬 Optimization Report",
        "📊 Telemetry & History",
        "🏗️ Architecture",
    ]
)

# =============================================================================
# TAB 1: Live Inference & Side-by-Side Comparison
# =============================================================================
with tab1:
    st.subheader("Interactive Model Inference")
    st.caption("Submit text to evaluate real-time classification, confidence score, and latency.")

    input_text = st.text_area(
        "Input text for sentiment classification",
        value=default_input_text,
        height=100,
        placeholder="Type or paste any customer review or sentence...",
    )

    col_btn, col_mode = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run Inference", type="primary", use_container_width=True)
    with col_mode:
        eval_mode = st.radio(
            "Evaluation Target",
            ["Side-by-Side (FP32 vs INT8)", "FP32 Only", "INT8 Only"],
            horizontal=True,
        )

    if run_btn:
        if not input_text.strip():
            st.warning("Please enter valid text before running inference.")
        else:
            with st.spinner("Executing ONNX inference..."):
                fp32_res = None
                int8_res = None

                if eval_mode in ["Side-by-Side (FP32 vs INT8)", "FP32 Only"]:
                    if fp32_health["online"]:
                        try:
                            fp32_res = predict(FP32_API_URL, input_text, variant="fp32")
                            st.session_state.history.append({
                                "timestamp": pd.Timestamp.now(),
                                "model": "FP32",
                                "text": input_text,
                                "prediction": fp32_res["label"],
                                "confidence": fp32_res["confidence"],
                                "model_latency_ms": fp32_res["latency_ms"],
                                "client_latency_ms": fp32_res["client_latency_ms"],
                            })
                        except Exception as e:
                            st.error(f"FP32 request failed: {e}")
                    else:
                        st.error("FP32 model is unavailable.")

                if eval_mode in ["Side-by-Side (FP32 vs INT8)", "INT8 Only"]:
                    if int8_health["online"]:
                        try:
                            int8_res = predict(INT8_API_URL, input_text, variant="int8")
                            st.session_state.history.append({
                                "timestamp": pd.Timestamp.now(),
                                "model": "INT8",
                                "text": input_text,
                                "prediction": int8_res["label"],
                                "confidence": int8_res["confidence"],
                                "model_latency_ms": int8_res["latency_ms"],
                                "client_latency_ms": int8_res["client_latency_ms"],
                            })
                        except Exception as e:
                            st.error(f"INT8 request failed: {e}")
                    else:
                        st.error("INT8 model is unavailable.")

            # Display Results Side-by-Side
            if fp32_res and int8_res:
                st.markdown("---")
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("#### 🔵 **FP32 Model (Baseline)**")
                    badge_color = "🟢" if fp32_res["label"] == "POSITIVE" else "🔴"
                    st.markdown(f"### {badge_color} `{fp32_res['label']}`")
                    st.progress(float(fp32_res["confidence"]), text=f"Confidence: {fp32_res['confidence'] * 100:.2f}%")
                    m1, m2 = st.columns(2)
                    m1.metric("Engine Latency", f"{fp32_res['latency_ms']:.2f} ms")
                    m2.metric("Total Latency", f"{fp32_res['client_latency_ms']:.2f} ms")

                with c2:
                    st.markdown("#### ⚡ **INT8 Model (Quantized)**")
                    badge_color = "🟢" if int8_res["label"] == "POSITIVE" else "🔴"
                    st.markdown(f"### {badge_color} `{int8_res['label']}`")
                    st.progress(float(int8_res["confidence"]), text=f"Confidence: {int8_res['confidence'] * 100:.2f}%")
                    m1, m2 = st.columns(2)
                    speedup_delta = f"{((fp32_res['latency_ms'] - int8_res['latency_ms']) / fp32_res['latency_ms']) * 100:+.1f}%"
                    m1.metric("Engine Latency", f"{int8_res['latency_ms']:.2f} ms", delta=speedup_delta)
                    m2.metric("Total Latency", f"{int8_res['client_latency_ms']:.2f} ms")

                # Comparison Summary Banner
                pred_match = fp32_res["label"] == int8_res["label"]
                conf_diff = abs(fp32_res["confidence"] - int8_res["confidence"]) * 100
                st.markdown("---")
                sc1, sc2, sc3 = st.columns(3)
                if pred_match:
                    sc1.success("✅ Prediction Parity: 100% Agreement")
                else:
                    sc1.warning("⚠️ Prediction Mismatch")
                sc2.info(f"📊 Confidence Delta: **{conf_diff:.3f}%**")
                latency_diff = fp32_res["latency_ms"] - int8_res["latency_ms"]
                if latency_diff > 0:
                    sc3.success(f"⚡ INT8 Speedup: **{latency_diff:.2f} ms faster**")
                else:
                    sc3.info(f"⚡ Engine Latency Delta: **{abs(latency_diff):.2f} ms**")

            elif fp32_res:
                st.markdown("---")
                st.markdown("#### 🔵 **FP32 Model Result**")
                st.markdown(f"### {'🟢' if fp32_res['label'] == 'POSITIVE' else '🔴'} `{fp32_res['label']}`")
                st.progress(float(fp32_res["confidence"]), text=f"Confidence: {fp32_res['confidence'] * 100:.2f}%")
                m1, m2 = st.columns(2)
                m1.metric("Model Latency", f"{fp32_res['latency_ms']:.2f} ms")
                m2.metric("Total Latency", f"{fp32_res['client_latency_ms']:.2f} ms")

            elif int8_res:
                st.markdown("---")
                st.markdown("#### ⚡ **INT8 Quantized Result**")
                st.markdown(f"### {'🟢' if int8_res['label'] == 'POSITIVE' else '🔴'} `{int8_res['label']}`")
                st.progress(float(int8_res["confidence"]), text=f"Confidence: {int8_res['confidence'] * 100:.2f}%")
                m1, m2 = st.columns(2)
                m1.metric("Model Latency", f"{int8_res['latency_ms']:.2f} ms")
                m2.metric("Total Latency", f"{int8_res['client_latency_ms']:.2f} ms")


# =============================================================================
# TAB 2: Benchmark Suite
# =============================================================================
with tab2:
    st.subheader("Live Performance & Latency Benchmark")
    st.caption("Execute concurrent or sequential stress tests across models to measure latency percentiles and throughput.")

    b_col1, b_col2, b_col3 = st.columns([2, 2, 2])
    with b_col1:
        target_model = st.selectbox("Target Service", ["Both (FP32 vs INT8)", "FP32 Only", "INT8 Only"])
    with b_col2:
        req_count = st.select_slider("Measured Requests", options=[25, 50, 100, 200], value=50)
    with b_col3:
        st.write("")
        st.write("")
        run_bench_btn = st.button("🔥 Start Benchmark", type="primary", use_container_width=True)

    if run_bench_btn:
        bench_targets = {}
        if target_model in ["Both (FP32 vs INT8)", "FP32 Only"]:
            if fp32_health["online"]:
                bench_targets["FP32"] = (FP32_API_URL, "fp32")
            else:
                st.error("FP32 model is unavailable. Cannot run benchmark.")
        if target_model in ["Both (FP32 vs INT8)", "INT8 Only"]:
            if int8_health["online"]:
                bench_targets["INT8"] = (INT8_API_URL, "int8")
            else:
                st.error("INT8 model is unavailable. Cannot run benchmark.")

        if bench_targets:
            p_bar = st.progress(0, text="Initializing benchmark runs...")
            results_payload = {}
            for name, (url, variant) in bench_targets.items():
                p_bar.text(f"Running benchmark on {name} ({req_count} requests)...")
                results_payload[name] = run_benchmark(url, req_count, variant=variant, progress_bar=p_bar)
            p_bar.empty()
            st.session_state.benchmark_results = {"requests": req_count, "results": results_payload}
            st.toast("Benchmark completed successfully!", icon="✅")

    # Render Benchmark Results
    if st.session_state.benchmark_results:
        b_data = st.session_state.benchmark_results["results"]
        st.markdown("---")

        if "FP32" in b_data and "INT8" in b_data:
            fp32_b = b_data["FP32"]
            int8_b = b_data["INT8"]

            # Key Metric Comparison Cards
            m1, m2, m3, m4 = st.columns(4)
            p50_speedup = ((fp32_b["p50"] - int8_b["p50"]) / fp32_b["p50"]) * 100
            p95_speedup = ((fp32_b["p95"] - int8_b["p95"]) / fp32_b["p95"]) * 100
            t_gain = ((int8_b["throughput"] - fp32_b["throughput"]) / fp32_b["throughput"]) * 100

            m1.metric("FP32 P50", f"{fp32_b['p50']:.2f} ms")
            m2.metric("INT8 P50", f"{int8_b['p50']:.2f} ms", delta=f"{p50_speedup:+.1f}%")
            m3.metric("INT8 P95", f"{int8_b['p95']:.2f} ms", delta=f"{p95_speedup:+.1f}%")
            m4.metric("Throughput Gain", f"{int8_b['throughput']} req/s", delta=f"{t_gain:+.1f}%")

            # Plotly Percentile Bar Chart
            st.markdown("#### **Latency Percentile Comparison (Lower is Better)**")
            metrics_labels = ["Mean Latency", "P50 Latency", "P95 Latency", "P99 Latency"]
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        name="FP32 (Baseline)",
                        x=metrics_labels,
                        y=[fp32_b["mean"], fp32_b["p50"], fp32_b["p95"], fp32_b["p99"]],
                        marker_color="#3b82f6",
                    ),
                    go.Bar(
                        name="INT8 (Quantized)",
                        x=metrics_labels,
                        y=[int8_b["mean"], int8_b["p50"], int8_b["p95"], int8_b["p99"]],
                        marker_color="#10b981",
                    ),
                ]
            )
            fig_bar.update_layout(
                barmode="group",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="Milliseconds (ms)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Request Sequence Latency Line Chart
            st.markdown("#### **Per-Request Latency Profile**")
            df_seq = pd.DataFrame({
                "Request #": list(range(1, len(fp32_b["latencies"]) + 1)),
                "FP32 Latency (ms)": fp32_b["latencies"],
                "INT8 Latency (ms)": int8_b["latencies"],
            })
            fig_line = px.line(
                df_seq,
                x="Request #",
                y=["FP32 Latency (ms)", "INT8 Latency (ms)"],
                color_discrete_map={"FP32 Latency (ms)": "#3b82f6", "INT8 Latency (ms)": "#10b981"},
            )
            fig_line.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Latency (ms)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        else:
            single_name = list(b_data.keys())[0]
            single_b = b_data[single_name]
            st.markdown(f"#### **{single_name} Benchmark Results**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Latency", f"{single_b['mean']:.2f} ms")
            c2.metric("P50 Latency", f"{single_b['p50']:.2f} ms")
            c3.metric("P95 Latency", f"{single_b['p95']:.2f} ms")
            c4.metric("Throughput", f"{single_b['throughput']} req/s")


# =============================================================================
# TAB 3: Model Optimization Report
# =============================================================================
with tab3:
    st.subheader("DistilBERT SST-2 Optimization Pipeline")
    st.caption("Comprehensive analysis of storage footprint, inference throughput, and accuracy trade-offs.")

    # High-level Metrics Cards
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Storage Reduction", "74.86%", delta="-191.3 MB")
    r2.metric("Throughput Boost", "60.56 req/s", delta="+54.9%")
    r3.metric("SST-2 Accuracy Retention", "99.4%", delta="-0.58% absolute")
    r4.metric("F1 Retention", "99.45%", delta="-0.005 F1")

    st.markdown("---")
    c_chart, c_pipe = st.columns([1, 1])

    with c_chart:
        st.markdown("#### **Model Size Comparison (MB)**")
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["INT8 Model Size", "Storage Saved vs FP32"],
                    values=[64.24, 191.29],
                    hole=0.55,
                    marker=dict(colors=["#10b981", "#cbd5e1"]),
                    textinfo="label+value",
                )
            ]
        )
        fig_donut.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
            annotations=[dict(text="74.9%<br>Saved", x=0.5, y=0.5, font_size=18, showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_pipe:
        st.markdown("#### **Quantization & Validation Workflow**")
        pipeline_steps = [
            ("1. HuggingFace Export", "Load DistilBERT SST-2 PyTorch weights"),
            ("2. ONNX Graph Export", "Generate dynamic-axes FP32 graph (255.5 MB)"),
            ("3. Dynamic INT8 Quantization", "Quantize weights to QInt8 using ONNX Runtime"),
            ("4. Validation & Parity Test", "Evaluate SST-2 validation dataset for accuracy parity"),
            ("5. Production Serving Fleet", "Dockerized FastAPI serving with async threadpool"),
        ]
        for step, desc in pipeline_steps:
            st.markdown(f"**{step}** — *{desc}*")

    st.markdown("---")
    st.markdown("#### **Validation Benchmark Matrix (SST-2 Dataset)**")
    tradeoff_df = pd.DataFrame(
        {
            "Metric": [
                "Model File Size",
                "Mean Engine Latency (Batch 1)",
                "API Server Throughput",
                "Validation Accuracy (SST-2)",
                "Validation F1 Score",
            ],
            "FP32 Baseline": ["255.53 MB", "15.32 ms", "39.09 req/s", "91.06%", "0.9137"],
            "INT8 Quantized": ["64.24 MB", "6.62 ms", "60.56 req/s", "90.48%", "0.9087"],
            "Net Impact": ["-74.86% disk space", "2.3x faster inference", "+54.9% throughput", "-0.58% accuracy", "-0.005 F1 delta"],
        }
    )
    st.dataframe(tradeoff_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: Live Telemetry & Request History
# =============================================================================
with tab4:
    st.subheader("Live Request History & Telemetry")
    st.caption("Inspect live session requests, latency logs, and download telemetry data.")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        # Telemetry Summary Cards
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Requests", len(hist_df))
        h2.metric("Avg Engine Latency", f"{hist_df['model_latency_ms'].mean():.2f} ms")
        h3.metric("Avg Roundtrip Latency", f"{hist_df['client_latency_ms'].mean():.2f} ms")
        pos_ratio = (hist_df["prediction"] == "POSITIVE").mean() * 100
        h4.metric("Positive Sentiment", f"{pos_ratio:.1f}%")

        st.markdown("---")

        display_df = hist_df.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")
        display_df["confidence"] = (display_df["confidence"] * 100).round(2).astype(str) + "%"

        st.dataframe(
            display_df[["timestamp", "model", "prediction", "confidence", "model_latency_ms", "client_latency_ms", "text"]],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download History CSV",
            data=csv_data,
            file_name=f"inference_history_{int(time.time())}.csv",
            mime="text/csv",
        )
    else:
        st.info("No inference requests captured in this session yet. Run tests from the **Live Inference** tab.")


# =============================================================================
# TAB 5: System Architecture & Deployment
# =============================================================================
with tab5:
    st.subheader("Serving & Deployment Architecture")
    st.caption("Multi-container orchestration setup with FastAPI, ONNX Runtime, and Prometheus.")

    st.code(
        """
                                  ┌───────────────────────────┐
                                  │    Streamlit Dashboard    │
                                  │        (Port 8501)        │
                                  └─────────────┬─────────────┘
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          │                                           │
                 HTTP POST│/predict                                   │HTTP POST /predict
                          ▼                                           ▼
             ┌─────────────────────────┐                 ┌─────────────────────────┐
             │    FastAPI FP32 Server  │                 │    FastAPI INT8 Server  │
             │       (Port 8000)       │                 │       (Port 8001)       │
             ├─────────────────────────┤                 ├─────────────────────────┤
             │ • asyncio.to_thread()   │                 │ • asyncio.to_thread()   │
             │ • ONNX FP32 (255 MB)    │                 │ • ONNX INT8 (64 MB)     │
             │ • Prometheus /metrics   │                 │ • Prometheus /metrics   │
             └────────────┬────────────┘                 └────────────┬────────────┘
                          │                                           │
                          └─────────────────────┬─────────────────────┘
                                                │ Scrapes /metrics every 5s
                                                ▼
                                  ┌───────────────────────────┐
                                  │     Prometheus Engine     │
                                  │        (Port 9090)        │
                                  └───────────────────────────┘
        """,
        language="text",
    )

    st.markdown("#### **Docker Compose Deployment**")
    st.markdown("To start the entire production fleet in a single command:")
    st.code("docker compose up --build", language="bash")

    st.markdown("#### **API Reference**")
    api_docs = pd.DataFrame(
        {
            "Method": ["GET", "GET", "POST", "GET"],
            "Endpoint": ["/", "/health", "/predict", "/metrics"],
            "Description": [
                "Service status & active model variant",
                "Liveness and readiness probe for load balancers",
                "Non-blocking sentiment prediction with confidence & engine latency",
                "Prometheus scrape endpoint exposing P95/P99 latency histograms and counters",
            ],
        }
    )
    st.dataframe(api_docs, use_container_width=True, hide_index=True)
