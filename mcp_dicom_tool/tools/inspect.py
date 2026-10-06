"""读取工具：状态 / 事件 / masks / ROI / DICOM 扫描。"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

from ..context import get_ctx
from ._util import call, require_bridge


def _json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return s


def register(mcp) -> None:
    @mcp.tool()
    def dcm_query_state() -> dict:
        """读取 GUI 当前状态（实时，需 GUI 已启动）。

        返回 dicom_loaded/dicom_path/sam_mode/window_center/window_width/
        points/point_labels/mask_present/model_loaded/current_mask_id 等。
        """
        ok, data = call("query_state", {}, timeout=10.0, tool_name="dcm_query_state")
        if not ok:
            return data
        return data

    @mcp.tool()
    def dcm_query_events(etype: Optional[str] = None, limit: int = 200) -> dict:
        """读取 GUI 操作记录（SQLite 事件表）。

        etype 可过滤：file_loaded/scan_done/model_loaded/sam_point_added/
        sam_mask_updated/sam_mask_saved/manual_roi/roi_detect_done/gui_ready/gui_exit/error。
        返回 [{t, type, data}, ...]，倒序。
        """
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        rows = ctx.recorder.query_events(etype=etype, limit=limit)
        ctx.recorder.record_tool_call("dcm_query_events",
                                      {"etype": etype, "limit": limit}, True, {"count": len(rows)})
        return {"ok": True, "count": len(rows), "events": rows}

    @mcp.tool()
    def dcm_query_errors(limit: int = 50) -> dict:
        """读取错误记录（SQLite errors 表）。返回 [{t, level, source, msg}, ...] 倒序。"""
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        rows = ctx.recorder.query_errors(limit=limit)
        return {"ok": True, "count": len(rows), "errors": rows}

    @mcp.tool()
    def dcm_query_tool_calls(tool: Optional[str] = None, limit: int = 100) -> dict:
        """读取 MCP 工具调用记录（SQLite tool_calls 表）。"""
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        rows = ctx.recorder.query_tool_calls(tool=tool, limit=limit)
        return {"ok": True, "count": len(rows), "calls": rows}

    @mcp.tool()
    def dcm_list_masks(limit: int = 50) -> dict:
        """列出已记录的 SAM masks（从 sam_mask_updated 事件抽取）。

        每项含 mask_id/mask_path/mask_png/file/score/source/stats。
        """
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        events = ctx.recorder.query_events(etype="sam_mask_updated", limit=limit * 3)
        masks = []
        for e in events:
            d = e.get("data", {})
            if d.get("mask_id"):
                masks.append({
                    "mask_id": d.get("mask_id"),
                    "mask_path": d.get("mask_path"),
                    "mask_png": d.get("mask_png"),
                    "file": d.get("file"),
                    "basename": d.get("basename"),
                    "score": d.get("score"),
                    "source": d.get("source"),
                    "note": d.get("note"),
                    "stats": d.get("stats"),
                    "t": e.get("t"),
                })
            if len(masks) >= limit:
                break
        ctx.recorder.record_tool_call("dcm_list_masks", {"limit": limit}, True,
                                      {"count": len(masks)})
        return {"ok": True, "count": len(masks), "masks": masks}

    @mcp.tool()
    def dcm_get_mask(mask_id: str) -> dict:
        """读取指定 mask：加载 npy 数组并返回形状/像素数/与保存的统计。

        用于后续计算（如复算 HU 统计、与 DICOM 对齐）。
        """
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        events = ctx.recorder.query_events(etype="sam_mask_updated", limit=500)
        target = None
        for e in events:
            d = e.get("data", {})
            if d.get("mask_id") == mask_id:
                target = d
                break
        if target is None:
            return {"ok": False, "error": f"mask_id 未找到: {mask_id}"}
        npy_path = target.get("mask_path", "")
        info = {"mask_id": mask_id, "file": target.get("file"), "score": target.get("score"),
                "source": target.get("source"), "stats": target.get("stats"),
                "mask_png": target.get("mask_png"), "mask_path": npy_path}
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            info["shape"] = list(arr.shape)
            info["pixel_count"] = int(arr.sum())
            info["dtype"] = str(arr.dtype)
        return {"ok": True, "mask": info}

    @mcp.tool()
    def dcm_get_latest_mask() -> dict:
        """读取最近一次记录的 SAM mask 完整信息。"""
        ctx = get_ctx()
        if ctx.recorder is None:
            return {"error": "recorder not initialized"}
        events = ctx.recorder.query_events(etype="sam_mask_updated", limit=50)
        for e in events:
            d = e.get("data", {})
            if d.get("mask_id"):
                return _get_mask_helper(d["mask_id"])
        return {"ok": False, "error": "暂无记录的 mask"}

    @mcp.tool()
    def dcm_get_rois(limit: int = 200) -> dict:
        """读取 ROI 汇总数据库（roi_summary.db）中所有已保存 ROI 记录。

        含患者/方法/面积/均值HU/标准差/扫描信息/描述等，供统计计算。
        """
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            import dicom_summary_db
            rois = dicom_summary_db.get_all_rois()
            rois = rois[:limit]
            return {"ok": True, "count": len(rois), "rois": rois}
        except Exception as e:
            return {"ok": False, "error": f"读取 ROI 数据库失败: {e}"}

    @mcp.tool()
    def dcm_scan_case(path: str, max_depth: int = 4, max_files: int = 50000) -> dict:
        """只读扫描 DICOM 病例目录，按 SeriesInstanceUID 分组返回序列信息。

        返回每个序列 series_description/modality/file_count/实例范围等，
        以及 patient_count/series_count/file_count 汇总。
        """
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from dicom_cluster import DicomCluster
            cluster = DicomCluster()
            patients = cluster.scan_folder(path, progress_callback=None)
            summary = cluster.get_summary(patients)
            out_series = []
            seen_series = set()
            for p in patients.values():
                for st in p.studies.values():
                    for se in st.series.values():
                        key = se.uid or id(se)
                        if key in seen_series:
                            continue
                        seen_series.add(key)
                        files = se.files or []
                        instances = sorted(
                            [f.instance_number for f in files if f.instance_number],
                            key=lambda x: int(x) if x.isdigit() else 0)
                        out_series.append({
                            "series_description": se.description,
                            "modality": se.modality,
                            "file_count": len(files),
                            "instance_range": [instances[0], instances[-1]] if instances else [],
                        })
            return {"ok": True,
                    "patient_count": summary.get("patient_count"),
                    "series_count": summary.get("series_count"),
                    "file_count": summary.get("file_count"),
                    "modalities": summary.get("modalities"),
                    "series": out_series}
        except Exception as e:
            return {"ok": False, "error": f"扫描失败: {e}"}


def _get_mask_helper(mask_id: str) -> dict:
    from ..context import get_ctx
    ctx = get_ctx()
    if ctx.recorder is None:
        return {"error": "recorder not initialized"}
    events = ctx.recorder.query_events(etype="sam_mask_updated", limit=500)
    for e in events:
        d = e.get("data", {})
        if d.get("mask_id") == mask_id:
            npy_path = d.get("mask_path", "")
            info = {"mask_id": mask_id, "file": d.get("file"), "score": d.get("score"),
                    "source": d.get("source"), "stats": d.get("stats"),
                    "mask_png": d.get("mask_png"), "mask_path": npy_path}
            if os.path.exists(npy_path):
                arr = np.load(npy_path)
                info["shape"] = list(arr.shape)
                info["pixel_count"] = int(arr.sum())
                info["dtype"] = str(arr.dtype)
            return {"ok": True, "mask": info}
    return {"ok": False, "error": f"mask_id 未找到: {mask_id}"}
