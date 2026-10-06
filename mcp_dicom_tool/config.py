"""路径 / 端口 / 解释器配置。

环境变量（可选覆盖）:
  DICOM_TOOL_PYTHON   GUI 解释器（必须含 PySide6/pydicom/opencv）
  DICOM_TOOL_REPO     仓库根目录
  DICOM_TOOL_MCP_PORT 桥端口（默认 7810，被占时自动向后探测 7810-7910）
"""
from __future__ import annotations

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))


def _default_python() -> str:
    if os.environ.get("DICOM_TOOL_PYTHON"):
        return os.environ["DICOM_TOOL_PYTHON"]
    return sys.executable


def repo_root() -> str:
    return os.environ.get("DICOM_TOOL_REPO") or REPO_ROOT


def main_script() -> str:
    return os.path.join(repo_root(), "main.py")


def gui_python() -> str:
    return _default_python()


def default_port() -> int:
    try:
        return int(os.environ.get("DICOM_TOOL_MCP_PORT", "7810"))
    except ValueError:
        return 7810


def record_dir() -> str:
    return os.path.join(repo_root(), "mcp_records")


def masks_dir() -> str:
    return os.path.join(record_dir(), "masks")


def shots_dir() -> str:
    return os.path.join(record_dir(), "shots")


def gui_log_path() -> str:
    return os.path.join(record_dir(), "gui.log")


def db_path() -> str:
    return os.path.join(record_dir(), "dicom_tool.db")


def events_jsonl() -> str:
    return os.path.join(record_dir(), "events.jsonl")


def latest_state_json() -> str:
    return os.path.join(record_dir(), "latest_state.json")


def latest_mask_json() -> str:
    return os.path.join(record_dir(), "latest_mask.json")


def ensure_dirs() -> None:
    os.makedirs(record_dir(), exist_ok=True)
    os.makedirs(masks_dir(), exist_ok=True)
    os.makedirs(shots_dir(), exist_ok=True)
