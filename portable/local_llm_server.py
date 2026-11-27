"""Start the bundled loopback language model when an offline launcher needs it."""

from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import time

from inference.local_llm import LOCAL_LLM_MODEL, LocalReportExplainer


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return True


def _start_server(root: Path, port: int) -> subprocess.Popen | None:
    runtime = root / "inference_runtime" / "local_llm"
    executable = runtime / "llama.cpp" / "llama-server.exe"
    model = runtime / "qwen2.5-1.5b-ollama.gguf"
    if not executable.is_file() or not model.is_file():
        print("[本地大模型] 找不到随项目提供的推理程序或权重，病例报告可继续使用。", flush=True)
        return None
    command = [
        str(executable), "--model", str(model), "--alias", LOCAL_LLM_MODEL,
        "--host", "127.0.0.1", "--port", str(port), "--offline", "--no-ui",
        "--ctx-size", "2048", "--parallel", "1", "--n-gpu-layers", "0",
        "--threads", str(min(6, os.cpu_count() or 1)),
    ]
    try:
        return subprocess.Popen(
            command, cwd=executable.parent, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        print(f"[本地大模型] 无法启动推理程序（{type(error).__name__}），病例报告可继续使用。", flush=True)
        return None


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextmanager
def local_llm_server(root: Path, port: int):
    """Reuse a healthy model or own the subprocess started for this app."""
    explainer = LocalReportExplainer(port=port)
    child = None
    try:
        if _port_in_use(port):
            if explainer.is_available():
                print(f"[本地大模型] 已连接 127.0.0.1:{port} 的 {LOCAL_LLM_MODEL}。", flush=True)
            else:
                print(f"[本地大模型] 端口 {port} 上的服务尚未提供 {LOCAL_LLM_MODEL}，病例报告可继续使用。",
                      flush=True)
        else:
            child = _start_server(root, port)
            if child is not None:
                deadline = time.monotonic() + 20
                while child.poll() is None and time.monotonic() < deadline:
                    if explainer.is_available():
                        print(f"[本地大模型] {LOCAL_LLM_MODEL} 已在 127.0.0.1:{port} 就绪。", flush=True)
                        break
                    time.sleep(0.2)
                else:
                    print("[本地大模型] 推理服务未能及时就绪，病例报告可继续使用。", flush=True)
        yield explainer
    finally:
        if child is not None:
            _stop_server(child)
