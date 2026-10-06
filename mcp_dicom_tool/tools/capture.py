"""截图工具：dcm_screenshot。"""
from __future__ import annotations

from typing import Optional

from .. import config
from ._util import call, require_bridge

def register(mcp) -> None:
    @mcp.tool()
    def dcm_screenshot() -> dict:
        """截取 GUI 渲染区域（当前查看器画面），返回 png_base64 与统计。

        stats 含 mean/bright_pct/unique_colors 用于诊断显示异常。
        """
        ok, data = call("screenshot", {}, timeout=30.0, tool_name="dcm_screenshot")
        if not ok:
            return data
        return data

    @mcp.tool()
    def dcm_screenshot_save() -> dict:
        """截取 GUI 渲染区域并保存到 mcp_records/shots/，返回文件路径。"""
        from .. import config
        ok, data = call("screenshot", {"out_dir": config.shots_dir()},
                        timeout=30.0, tool_name="dcm_screenshot_save")
        if not ok:
            return data
        return data
