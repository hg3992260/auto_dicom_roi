"""DICOM Analysis Tool MCP server 入口（FastMCP, stdio JSON-RPC）。

运行方式:
  python mcp_dicom_tool\\server.py
  python -m mcp_dicom_tool.server
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_dicom_tool import config  # noqa: E402
from mcp_dicom_tool.context import get_ctx  # noqa: E402
from mcp_dicom_tool.recorder import Recorder  # noqa: E402


def build_server() -> FastMCP:
    mcp = FastMCP(
        "dicom_tool",
        instructions=(
            "DICOM Analysis Tool MCP 服务：读取 GUI 上的一切用户操作记录。\n"
            "先用 dcm_launch 启动主程序 GUI，再在 GUI 上做 SAM 交互/ROI 保存；\n"
            "随后用 dcm_query_events/dcm_list_masks/dcm_get_mask/dcm_wait_event 读取记录，\n"
            "dcm_screenshot 截图确认，dcm_get_rois 读取 ROI 汇总库用于统计计算。"
        ),
    )

    config.ensure_dirs()
    ctx = get_ctx()
    ctx.recorder = Recorder()
    ctx.mcp = mcp

    from mcp_dicom_tool.tools import register_all
    register_all(mcp)

    return mcp


def main() -> int:
    mcp = build_server()
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
