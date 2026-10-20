"""
Model Serving Benchmark — Local Fleet Launcher
===============================================
Launches the full local development stack in a single command:
  1. FP32 Model Server  -> http://localhost:8000
  2. INT8 Model Server  -> http://localhost:8001
  3. Streamlit UI       -> http://localhost:8501

Press Ctrl+C to stop all services cleanly.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding across Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent

# Locate the Python executable in the venv if available, else use current sys.executable
VENV_PYTHON = ROOT_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

processes = []


def stop_all(sig=None, frame=None):
    """Gracefully terminate all child processes."""
    print("\nStopping all services...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
    print("All services stopped.")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    print("=" * 65)
    print("Starting Model Serving Benchmark Local Fleet")
    print(f"Python Runtime: {PYTHON_EXE}")
    print("=" * 65)

    # 1. Start FP32 Model Server on Port 8000
    env_fp32 = os.environ.copy()
    env_fp32["MODEL_VARIANT"] = "fp32"
    env_fp32["PYTHONIOENCODING"] = "utf-8"
    print("\n[1/3] Starting FP32 Model Server on http://localhost:8000 ...")
    p_fp32 = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env_fp32,
        cwd=str(ROOT_DIR),
    )
    processes.append(p_fp32)

    # 2. Start INT8 Model Server on Port 8001
    env_int8 = os.environ.copy()
    env_int8["MODEL_VARIANT"] = "int8"
    env_int8["PYTHONIOENCODING"] = "utf-8"
    print("[2/3] Starting INT8 Model Server on http://localhost:8001 ...")
    p_int8 = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"],
        env=env_int8,
        cwd=str(ROOT_DIR),
    )
    processes.append(p_int8)

    # Give servers a short moment to load models into memory
    print("\nWarming up model servers (loading ONNX graphs)...")
    time.sleep(3)

    # 3. Start Streamlit Dashboard on Port 8501
    env_ui = os.environ.copy()
    env_ui["FP32_API_URL"] = "http://localhost:8000"
    env_ui["INT8_API_URL"] = "http://localhost:8001"
    env_ui["PYTHONIOENCODING"] = "utf-8"
    print("[3/3] Starting Streamlit Dashboard on http://localhost:8501 ...\n")
    p_ui = subprocess.Popen(
        [
            PYTHON_EXE,
            "-m",
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ],
        env=env_ui,
        cwd=str(ROOT_DIR),
    )
    processes.append(p_ui)

    print("=" * 65)
    print("All services running!")
    print("   -> Dashboard:   http://localhost:8501")
    print("   -> FP32 API:    http://localhost:8000 (Docs: /docs, Metrics: /metrics)")
    print("   -> INT8 API:    http://localhost:8001 (Docs: /docs, Metrics: /metrics)")
    print("   Press Ctrl+C at any time to shut down the fleet.")
    print("=" * 65 + "\n")

    try:
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"Warning: A service exited unexpectedly (PID {p.pid}). Shutting down.")
                    stop_all()
            time.sleep(1)
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()

