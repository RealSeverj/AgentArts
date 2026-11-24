"""One-click launcher for the Retinal Screening & Hospital Closed-Loop System.

Usage:
    python run.py           # Start API + Hospital Portal + SSH public tunnel (default: https://severj.top)
    python run.py local     # Start local API + Hospital Portal only (no SSH tunnel)
    python run.py test      # Run full unit test suite
    python run.py export    # Refresh AgentArts single-agent JSONL and dual-app Dify 0.6.0 YAML bundles
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
DEFAULT_PUBLIC_ORIGIN = "https://severj.top"
DEFAULT_SSH_HOST = "severj-cn"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_REMOTE_PORT = 18000


def _find_py310_python() -> Path | None:
    """Locate conda py310 interpreter when launched from another Python environment."""
    candidates = []
    env_override = os.environ.get("RETINA_PY310_PYTHON")
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend([
        Path.home() / "anaconda3" / "envs" / "py310" / "python.exe",
        Path.home() / "miniconda3" / "envs" / "py310" / "python.exe",
        Path.home() / "anaconda3" / "envs" / "py310" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "py310" / "bin" / "python",
    ])
    for item in candidates:
        if item.is_file():
            return item.resolve()
    return None


def _ensure_py310() -> None:
    if os.environ.get("RETINA_SKIP_PY310_REEXEC") == "1":
        return
    curr = Path(sys.executable).resolve()
    if "py310" in curr.as_posix().lower():
        return
    target = _find_py310_python()
    if target is None or target == curr:
        return
    env = dict(os.environ, RETINA_SKIP_PY310_REEXEC="1")
    code = subprocess.call([str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(code)


class TunnelSupervisor:
    """Keep an SSH reverse tunnel alive in a daemon thread and close cleanly on exit."""

    def __init__(self, *, ssh_host: str, remote_port: int, local_host: str, local_port: int):
        self.ssh_host = ssh_host
        self.remote_port = remote_port
        self.local_host = local_host
        self.local_port = local_port
        self._stop = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil.which("ssh"):
            raise RuntimeError("未检测到系统 ssh 命令，无法建立公网反向隧道；可改用 python run.py local。")
        self._thread = threading.Thread(target=self._loop, name="ssh-tunnel", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cmd = [
            "ssh",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
            "-N",
            "-R", f"127.0.0.1:{self.remote_port}:{self.local_host}:{self.local_port}",
            self.ssh_host,
        ]
        while not self._stop.is_set():
            try:
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._proc = subprocess.Popen(cmd)
                proc = self._proc
                proc.wait()
            except OSError as exc:
                print(f"[隧道] 启动失败：{exc}", flush=True)
            finally:
                with self._lock:
                    self._proc = None
            if not self._stop.is_set():
                print("[隧道] 连接已断开，3 秒后自动重连...", flush=True)
                self._stop.wait(3.0)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def cmd_export(origin: str) -> None:
    from portable.run import new_machine_key

    api_key = new_machine_key()
    out_root = ROOT / "agent" / "agent_ready_severj"
    dify_dir = out_root / "dify_workflows"
    out_root.mkdir(parents=True, exist_ok=True)

    # Refresh sanitized repo copy + local key-embedded copy
    subprocess.run(
        [sys.executable, str(ROOT / "agent" / "export_dify_workflows.py"), "--origin", origin],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "agent" / "export_dify_workflows.py"),
            "--origin", origin,
            "--output", str(dify_dir),
            "--api-key", api_key,
        ],
        check=True,
    )
    print(f"[导出] 已生成 Dify 0.6.0 双应用工作流至：{dify_dir}")


def cmd_test() -> int:
    res = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=str(ROOT))
    return res.returncode


def cmd_serve(*, with_tunnel: bool, public_origin: str, ssh_host: str, host: str, port: int, remote_port: int, device: str) -> None:
    sys.path.insert(0, str(ROOT))
    from portable.run import new_machine_key, serve

    new_machine_key()
    dify_dir = ROOT / "agent" / "agent_ready_severj" / "dify_workflows"
    if not dify_dir.is_dir():
        cmd_export(public_origin)

    tunnel: TunnelSupervisor | None = None
    if with_tunnel:
        tunnel = TunnelSupervisor(ssh_host=ssh_host, remote_port=remote_port, local_host=host, local_port=port)
        tunnel.start()

    print("=" * 64)
    print("视健康·眼底筛查与医患随访多智能体系统启动中")
    print(f"  本地检查接口 : http://{host}:{port}/readyz")
    print(f"  本地演示门户 : http://{host}:{port}/v1/demo")
    print(f"  本地拍片室   : http://{host}:{port}/v1/imaging_room")
    if with_tunnel:
        print(f"  公网演示门户 : {public_origin}/v1/demo")
        print(f"  公网拍片室   : {public_origin}/v1/imaging_room")
        print(f"  SSH 反向隧道 : {ssh_host} (127.0.0.1:{remote_port} -> {host}:{port})")
    print("  停止服务     : 按 Ctrl+C 即可安全退出")
    print("=" * 64, flush=True)

    serve_args = argparse.Namespace(
        host=host,
        port=port,
        device=device,
        public_origin=public_origin if with_tunnel else None,
    )
    try:
        serve(serve_args)
    finally:
        if tunnel is not None:
            print("\n[停止] 正在关闭 SSH 反向隧道...", flush=True)
            tunnel.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视健康·眼底筛查多智能体系统一键启动脚本")
    sub = parser.add_subparsers(dest="command")

    for name in ("up", "local"):
        p = sub.add_parser(name, help="启动服务" + ("与公网 SSH 隧道" if name == "up" else "（仅本机模式）"))
        p.add_argument("--origin", default=DEFAULT_PUBLIC_ORIGIN)
        p.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
        p.add_argument("--host", default=DEFAULT_HOST)
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    exp = sub.add_parser("export", help="刷新导出 AgentArts 工作流包")
    exp.add_argument("--origin", default=DEFAULT_PUBLIC_ORIGIN)

    sub.add_parser("test", help="运行全部单元测试")
    return parser


def main() -> int:
    _ensure_py310()
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command or "up"

    if cmd == "test":
        return cmd_test()
    if cmd == "export":
        cmd_export(getattr(args, "origin", DEFAULT_PUBLIC_ORIGIN))
        return 0
    if cmd in ("up", "local"):
        cmd_serve(
            with_tunnel=(cmd == "up"),
            public_origin=getattr(args, "origin", DEFAULT_PUBLIC_ORIGIN),
            ssh_host=getattr(args, "ssh_host", DEFAULT_SSH_HOST),
            host=getattr(args, "host", DEFAULT_HOST),
            port=getattr(args, "port", DEFAULT_PORT),
            remote_port=getattr(args, "remote_port", DEFAULT_REMOTE_PORT),
            device=getattr(args, "device", "auto"),
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
