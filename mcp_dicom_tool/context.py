"""MCP server 全局上下文（进程级单例）。"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .bridge_client import BridgeClient
from .recorder import Recorder


class Context:
    def __init__(self) -> None:
        self.recorder: Optional[Recorder] = None
        self.client: Optional[BridgeClient] = None
        self.gui_process: Any = None
        self.gui_pid: Optional[int] = None
        self.port: Optional[int] = None
        self.mcp: Any = None
        self.last_error: Optional[str] = None


_ctx: Optional[Context] = None


def get_ctx() -> Context:
    global _ctx
    if _ctx is None:
        _ctx = Context()
    return _ctx
