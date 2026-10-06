"""工具层共享工具函数。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..bridge_client import BridgeClient
from ..context import get_ctx


def get_client(auto_connect: bool = True) -> Optional[BridgeClient]:
    ctx = get_ctx()
    client = ctx.client
    if client is None:
        from .. import config
        client = BridgeClient(port=ctx.port or config.default_port())
        ctx.client = client

        def _on_event(evt):
            if ctx.recorder is not None:
                try:
                    ctx.recorder.record_event(evt.get("type"), evt.get("data") or {})
                except Exception:
                    pass

        client.set_event_handler(_on_event)
    if auto_connect and not client.connected:
        client.connect(timeout=2.0)
    return client


def call(op: str, args: Optional[Dict[str, Any]] = None, timeout: float = 30.0,
         tool_name: str = "") -> Tuple[bool, Any]:
    client = get_client(auto_connect=True)
    if client is None:
        return False, {"error": "client init failed"}
    ok, data = client.call(op, args, timeout=timeout)
    ctx = get_ctx()
    if ctx.recorder is not None:
        ctx.recorder.record_tool_call(tool_name or f"bridge:{op}", args or {}, ok, data)
    if not ok:
        msg = data.get("error") if isinstance(data, dict) else str(data)
        ctx.last_error = msg
    return ok, data


def require_bridge(tool_name: str) -> Tuple[bool, Any]:
    client = get_client(auto_connect=True)
    if client is None or not client.connected:
        return False, {"error": "GUI bridge 未连接（先 dcm_launch 启动主程序）"}
    return True, None
