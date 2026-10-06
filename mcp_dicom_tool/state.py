"""事件类型常量 + GuiState 快照辅助。

桥协议: TCP line-delimited JSON
  请求 {"id","op","args"}  响应 {"id","ok","data"}  事件 {"type","data","ts"}
"""
from __future__ import annotations

# GUI 主动推送的事件类型
EVENT_TYPES = [
    "gui_ready",
    "gui_exit",
    "file_loaded",
    "scan_done",
    "model_loaded",
    "sam_point_added",
    "sam_mask_updated",
    "sam_mask_saved",
    "manual_roi",
    "roi_detect_done",
    "error",
    "state",
    "user_action",
]


def is_valid_event(etype: str) -> bool:
    return etype in EVENT_TYPES
