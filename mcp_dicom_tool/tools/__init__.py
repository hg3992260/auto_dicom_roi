"""工具注册与热重载。"""
from __future__ import annotations

import importlib
import pkgutil
from typing import List


def _tool_modules():
    return [m.name for m in pkgutil.iter_modules(__path__)
            if not m.name.startswith("_")]


def register_all(mcp) -> List[str]:
    from . import lifecycle, inspect, capture, recording
    lifecycle.register(mcp)
    inspect.register(mcp)
    capture.register(mcp)
    recording.register(mcp)
    return _tool_modules()


def reload_tools(mcp) -> List[str]:
    mods = _tool_modules()
    for name in mods:
        importlib.reload(importlib.import_module(f"{__name__}.{name}"))
    return register_all(mcp)
