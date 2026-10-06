"""生命周期工具：dcm_launch / dcm_restart_gui / dcm_reload_tools。"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Any, Dict, Optional

from .. import config
from ..bridge_client import BridgeClient
from ..context import get_ctx
from ._util import get_client


def _free_port(start: int) -> int:
    port = start
    while port < start + 100:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            port += 1
        finally:
            try:
                s.close()
            except Exception:
                pass
    return start


def _gui_env() -> dict:
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _launch_gui() -> Dict[str, Any]:
    ctx = get_ctx()
    port = _free_port(config.default_port())
    python = config.gui_python()
    script = config.main_script()
    config.ensure_dirs()

    cmd = [python, script, "--mcp", "--mcp-port", str(port)]
    logf = open(config.gui_log_path(), "ab", buffering=0)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=config.repo_root(),
            env=_gui_env(),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except Exception as e:
        return {"ok": False, "error": f"启动 GUI 进程失败: {e}"}

    with open(config.gui_log_path(), "a", encoding="utf-8") as f:
        f.write(f"[mcp] spawned pid={proc.pid} port={port}\n")
        f.flush()

    ctx.gui_process = proc
    ctx.gui_pid = proc.pid
    ctx.port = port

    client = BridgeClient(port=port)
    ctx.client = client

    deadline = time.time() + 60.0
    ready = False
    while time.time() < deadline:
        if client.connected or client.connect(timeout=2.0):
            evt = client.wait_event("gui_ready", timeout=8.0)
            if evt is not None or client.connected:
                ready = True
                break
        time.sleep(0.5)

    if not ready:
        poll = proc.poll()
        return {"ok": False, "error": "GUI 桥在 60s 内未就绪（见 mcp_records/gui.log）",
                "pid": proc.pid, "port": port, "proc_alive": poll is None, "rc": poll}

    return {"ok": True, "pid": proc.pid, "port": port}


def register(mcp) -> None:
    @mcp.tool()
    def dcm_launch() -> dict:
        """启动 DICOM Analysis Tool GUI 进程并等待 TCP 桥就绪。

        返回 {ok, pid, port}；已运行返回 already_connected/already_running。
        启动后即可通过 dcm_query_state / dcm_wait_event 读取 GUI 上的操作。
        """
        ctx = get_ctx()
        client = get_client(auto_connect=True)
        if client is not None and client.connected:
            return {"ok": True, "already_connected": True, "port": ctx.port, "pid": ctx.gui_pid}
        proc = getattr(ctx, "gui_process", None)
        if proc is not None and proc.poll() is None:
            return {"ok": True, "already_running": True, "pid": proc.pid, "port": ctx.port}
        result = _launch_gui()
        if result.get("ok") and ctx.recorder is not None:
            ctx.recorder.record_event("gui_ready", {"pid": result.get("pid"), "port": result.get("port")})
        return result

    @mcp.tool()
    def dcm_restart_gui() -> dict:
        """重启 GUI 进程（main/ui/bridge 改动后调用生效）。

        先安全关闭旧进程再启动新进程，返回 {ok, pid, port, old_pid, reconnected}。
        """
        ctx = get_ctx()
        old_pid = ctx.gui_pid
        proc = getattr(ctx, "gui_process", None)
        if proc is not None and proc.poll() is None:
            client = get_client(auto_connect=False)
            if client is not None and client.connected:
                try:
                    client.call("shutdown", {}, timeout=5.0)
                except Exception:
                    pass
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        old_client = ctx.client
        if old_client is not None:
            old_client.close()
        ctx.client = None
        ctx.gui_process = None
        time.sleep(1.0)

        result = _launch_gui()
        result["old_pid"] = old_pid
        result["reconnected"] = bool(result.get("ok"))
        return result

    @mcp.tool()
    def dcm_reload_tools() -> dict:
        """热重载 MCP 工具层（mcp_dicom_tool/tools/*.py 改动后调用）。返回重载清单。"""
        from . import reload_tools
        from ..context import get_ctx as _g
        ctx = _g()
        registered = reload_tools(ctx.mcp)
        return {"ok": True, "reloaded": registered}
