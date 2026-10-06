"""录制工具：等待事件 / 手动记录 mask / 导出记录。"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .. import config
from ..context import get_ctx
from ._util import call


def register(mcp) -> None:
    @mcp.tool()
    def dcm_wait_event(etype: str, timeout: float = 120.0, contains: Optional[str] = None) -> dict:
        """阻塞等待 GUI 推送的指定事件。

        etype: file_loaded/scan_done/model_loaded/sam_point_added/
        sam_mask_updated/sam_mask_saved/manual_roi/roi_detect_done/gui_ready/gui_exit/error。
        contains 为事件 data 的 JSON 子串过滤（如某个文件名）。返回匹配事件或 null。
        """
        from ._util import get_client
        client = get_client(auto_connect=True)
        if client is None or not client.connected:
            return {"ok": False, "error": "GUI bridge 未连接（先 dcm_launch）"}
        evt = client.wait_event(etype, timeout=timeout, contains=contains)
        return {"ok": evt is not None, "event": evt}

    @mcp.tool()
    def dcm_record_current_mask(score: float = 0.0, note: str = "") -> dict:
        """把 GUI 当前显示的 mask 记录为一次 sam_mask_updated 事件（落盘 npy/png + 统计）。

        用于在用户手动确认前主动留痕当前 SAM 结果，供后续计算。
        """
        ok, data = call("record_sam_mask", {"score": score, "note": note}, timeout=30.0,
                        tool_name="dcm_record_current_mask")
        if not ok:
            return data
        return data

    @mcp.tool()
    def dcm_export(out_path: Optional[str] = None) -> dict:
        """导出全部记录（事件/工具调用/错误）到 JSONL 文件。返回导出路径。"""
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        events = ctx.recorder.query_events(limit=100000)
        tools = ctx.recorder.query_tool_calls(limit=100000)
        errors = ctx.recorder.query_errors(limit=100000)
        out_path = out_path or os.path.join(config.record_dir(), "export.jsonl")
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps({"kind": "event", **e}, ensure_ascii=False) + "\n")
            for t in tools:
                f.write(json.dumps({"kind": "tool_call", **t}, ensure_ascii=False) + "\n")
            for er in errors:
                f.write(json.dumps({"kind": "error", **er}, ensure_ascii=False) + "\n")
        ctx.recorder.record_tool_call("dcm_export", {}, True,
                                      {"path": out_path, "events": len(events),
                                       "tools": len(tools), "errors": len(errors)})
        return {"ok": True, "path": out_path, "events": len(events),
                "tools": len(tools), "errors": len(errors)}

    @mcp.tool()
    def dcm_stats_summary() -> dict:
        """汇总当前记录概况：事件总数、各类事件计数、mask 数量、ROI 数量。"""
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        events = ctx.recorder.query_events(limit=100000)
        type_counts: Dict[str, int] = {}
        mask_count = 0
        for e in events:
            t = e.get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
            if t == "sam_mask_updated":
                mask_count += 1
        roi_count = 0
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            import dicom_summary_db
            roi_count = len(dicom_summary_db.get_all_rois())
        except Exception:
            pass
        return {"ok": True, "total_events": len(events), "type_counts": type_counts,
                "mask_count": mask_count, "roi_count": roi_count}
