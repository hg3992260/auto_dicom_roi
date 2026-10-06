"""运行在 GUI 进程内：TCP server + 主线程调度 + 事件记录。

把 MCP server 发来的 op 请求投递到 Qt 主线程执行（QueuedConnection），
并把 GUI 主动事件（file_loaded/scan_done/sam_mask_updated/roi_saved/...）
推回客户端并写入 recorder（SQLite+JSONL）。

协议: TCP line-delimited JSON（见 bridge_client.py）。
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage, QPixmap

from . import config
from .recorder import Recorder

_mask_seq = {"n": 0}
_mask_seq_lock = threading.Lock()


def _next_mask_id() -> str:
    with _mask_seq_lock:
        _mask_seq["n"] += 1
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{_mask_seq['n']:04d}"


# ---------------------------------------------------------------------------
# 状态采集 / 截图 / mask 保存（在主线程内执行，直接访问 MainWindow）
# ---------------------------------------------------------------------------

def collect_state(win) -> Dict[str, Any]:
    viewer = getattr(win, "viewer", None)
    sam_panel = getattr(win, "sam_panel", None)
    state = {
        "dicom_loaded": bool(getattr(viewer, "_raw_pixels", None) is not None),
        "dicom_path": getattr(viewer, "_current_path", "") or "",
        "sam_mode": bool(getattr(viewer, "sam_mode", False)),
        "window_center": float(getattr(viewer, "raw_window_center", 0.0)),
        "window_width": float(getattr(viewer, "raw_window_width", 0.0)),
        "points": list(getattr(viewer, "_points", []) or []),
        "point_labels": list(getattr(viewer, "_point_labels", []) or []),
        "mask_present": bool(getattr(viewer, "_mask", None) is not None),
        "model_loaded": bool(sam_panel is not None and getattr(sam_panel.engine, "predictor", None) is not None),
        "model_status": sam_panel.engine.get_status() if sam_panel is not None else "",
        "current_mask_id": getattr(viewer, "_last_mask_id", None),
    }
    if state["dicom_loaded"]:
        raw = viewer._raw_pixels
        state["image_shape"] = list(raw.shape[:2])
    return state


def compute_mask_stats_from_viewer(viewer, mask: np.ndarray) -> Dict[str, Any]:
    raw = getattr(viewer, "_raw_pixels", None)
    if raw is None or mask is None:
        return {"error": "no image/mask"}
    if mask.shape[:2] != raw.shape[:2]:
        mask = np.asarray(mask, dtype=np.uint8)
        import cv2
        mask = cv2.resize(mask, (raw.shape[1], raw.shape[0])).astype(bool)
    hu = raw[mask]
    if hu.size == 0:
        return {"error": "empty selection"}
    sx, sy = viewer._get_pixel_spacing()
    return {
        "pixel_count": int(hu.size),
        "area_mm2": round(float(hu.size * sx * sy), 3),
        "mean_hu": round(float(np.mean(hu)), 3),
        "std_hu": round(float(np.std(hu)), 3),
        "min_hu": round(float(np.min(hu)), 3),
        "max_hu": round(float(np.max(hu)), 3),
        "median_hu": round(float(np.median(hu)), 3),
        "pixel_spacing": [float(sx), float(sy)],
    }


def save_mask_to_disk(mask: np.ndarray, file_path: str) -> Dict[str, Any]:
    mask_id = _next_mask_id()
    config.ensure_dirs()
    npy_path = os.path.join(config.masks_dir(), f"{mask_id}.npy")
    png_path = os.path.join(config.masks_dir(), f"{mask_id}.png")
    np.save(npy_path, np.asarray(mask, dtype=bool))
    from PIL import Image
    img = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255))
    img.save(png_path)
    return {"mask_id": mask_id, "mask_path": npy_path, "mask_png": png_path}


def record_sam_mask(win, mask: np.ndarray, file_path: str,
                    score: float = 0.0, source: str = "sam_interactive",
                    note: str = "") -> Dict[str, Any]:
    viewer = getattr(win, "viewer", None)
    saved = save_mask_to_disk(mask, file_path)
    stats = compute_mask_stats_from_viewer(viewer, mask)
    data = {
        **saved,
        "file": file_path,
        "basename": os.path.basename(file_path) if file_path else "",
        "score": float(score),
        "source": source,
        "stats": stats,
        "points": list(getattr(viewer, "_points", []) or []),
        "point_labels": list(getattr(viewer, "_point_labels", []) or []),
        "note": note,
        "ts": time.time(),
    }
    if viewer is not None:
        viewer._last_mask_id = saved["mask_id"]
    return data


def take_screenshot(win, out_dir: Optional[str] = None) -> Dict[str, Any]:
    viewer = getattr(win, "viewer", None)
    if viewer is None:
        return {"ok": False, "error": "no viewer"}
    label = viewer._label
    pm = label.grab()
    qimg = pm.toImage()
    buf = io.BytesIO()
    from PySide6.QtCore import QBuffer
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    qimg.save(buffer, "PNG")
    png_bytes = bytes(buffer.data())
    buffer.close()
    out_dir = out_dir or ""
    path = ""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"shot_{int(time.time()*1000)}.png")
        with open(path, "wb") as f:
            f.write(png_bytes)
    img = np.frombuffer(png_bytes, dtype=np.uint8)
    import cv2
    bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    stats = {}
    if bgr is not None:
        stats = {
            "mean": round(float(bgr.mean()), 2),
            "bright_pct": round(float((bgr.max(axis=2) > 50).mean() * 100.0), 2),
            "unique_colors": int(len(np.unique(bgr.reshape(-1, 3), axis=0))),
        }
    return {
        "ok": True,
        "path": path,
        "bytes": len(png_bytes),
        "png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# 跨线程调度器（主线程执行 op）
# ---------------------------------------------------------------------------

class _Dispatcher(QtCore.QObject):
    _cmd = QtCore.Signal(object)

    def __init__(self, win, bridge) -> None:
        super().__init__()
        self.win = win
        self.bridge = bridge
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._plock = threading.Lock()
        self._cmd.connect(self._on_cmd, QtCore.Qt.ConnectionType.QueuedConnection)

    def submit(self, req_id: str, op: str, args: Dict[str, Any], timeout: float) -> Tuple[bool, Any]:
        evt = threading.Event()
        holder: Dict[str, Any] = {}
        with self._plock:
            self._pending[req_id] = {"evt": evt, "holder": holder}
        self._cmd.emit((req_id, op, args))
        if not evt.wait(timeout):
            with self._plock:
                self._pending.pop(req_id, None)
            return False, {"error": f"op '{op}' 主线程执行超时 ({timeout}s)"}
        return holder.get("ok", False), holder.get("data", {})

    @QtCore.Slot(object)
    def _on_cmd(self, payload) -> None:
        req_id, op, args = payload
        ok = False
        data: Any = {}
        try:
            data = self._execute(op, args)
            ok = True
        except Exception as e:
            traceback.print_exc()
            data = {"error": f"{type(e).__name__}: {e}"}
            ok = False
            try:
                self.bridge.record_event("error", {"error": data["error"], "op": op})
            except Exception:
                pass
        with self._plock:
            item = self._pending.pop(req_id, None)
        if item is not None:
            item["holder"]["ok"] = ok
            item["holder"]["data"] = data
            item["evt"].set()

    def _execute(self, op: str, args: Dict[str, Any]) -> Any:
        win = self.win
        if op == "query_state":
            return {"ok": True, "state": collect_state(win)}
        if op == "screenshot":
            return take_screenshot(win, out_dir=args.get("out_dir"))
        if op == "get_current_mask":
            return self._get_current_mask(args)
        if op == "record_sam_mask":
            return self._record_sam_mask(args)
        if op == "shutdown":
            return self._shutdown()
        if op == "set_status":
            return {"ok": True}
        raise ValueError(f"unknown op: {op}")

    def _get_current_mask(self, args: Dict[str, Any]) -> Dict[str, Any]:
        win = self.win
        viewer = getattr(win, "viewer", None)
        mask = getattr(viewer, "_mask", None)
        if mask is None:
            return {"ok": False, "error": "no current mask"}
        data = record_sam_mask(win, mask, getattr(viewer, "_current_path", "") or "",
                               score=float(args.get("score", 0.0)),
                               source="sam_current",
                               note=args.get("note", ""))
        self.bridge.record_event("sam_mask_updated", data)
        return {"ok": True, **data}

    def _record_sam_mask(self, args: Dict[str, Any]) -> Dict[str, Any]:
        win = self.win
        viewer = getattr(win, "viewer", None)
        mask = getattr(viewer, "_mask", None)
        if mask is None:
            return {"ok": False, "error": "no current mask"}
        data = record_sam_mask(win, mask, getattr(viewer, "_current_path", "") or "",
                               score=float(args.get("score", 0.0)),
                               source=args.get("source", "sam_interactive"),
                               note=args.get("note", ""))
        self.bridge.record_event("sam_mask_updated", data)
        return {"ok": True, **data}

    def _shutdown(self) -> Dict[str, Any]:
        QtCore.QTimer.singleShot(0, self._do_shutdown)
        return {"ok": True, "shutting_down": True}

    def _do_shutdown(self) -> None:
        win = self.win
        try:
            win.close()
        except Exception:
            pass
        QtWidgets.QApplication.quit()


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

class GuiBridgeServer:
    def __init__(self, win, port: int = 7810, host: str = "127.0.0.1",
                 recorder: Optional[Recorder] = None) -> None:
        self.win = win
        self.port = port
        self.host = host
        self.recorder = recorder or Recorder()
        self.dispatcher = _Dispatcher(win, self)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._closing = threading.Event()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.record_event("gui_ready", {"port": self.port})

    def record_event(self, etype: str, data: Dict[str, Any]) -> None:
        evt = {"type": etype, "data": data, "ts": time.time()}
        try:
            self.recorder.record_event(etype, data)
        except Exception:
            pass
        with self._clients_lock:
            clients = list(self._clients)
        for c in clients:
            self._send(c, evt)

    def _serve(self) -> None:
        while self._running.is_set():
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._sock.bind((self.host, self.port))
                self._sock.listen(4)
                self._sock.settimeout(1.0)
            except OSError:
                self.port += 1
                continue
            while self._running.is_set():
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with self._clients_lock:
                    self._clients.append(conn)
                threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _client_loop(self, conn: socket.socket) -> None:
        buf = b""
        while self._running.is_set() and not self._closing.is_set():
            try:
                chunk = conn.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "id" in msg and "op" in msg:
                    self._handle_request(conn, msg)
        with self._clients_lock:
            if conn in self._clients:
                self._clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    def _handle_request(self, conn: socket.socket, msg: Dict[str, Any]) -> None:
        req_id = msg["id"]
        op = msg.get("op", "")
        args = msg.get("args", {}) or {}
        timeout = float(msg.get("timeout", 120.0))
        ok, data = self.dispatcher.submit(req_id, op, args, timeout)
        resp = {"id": req_id, "ok": ok, "data": data}
        self._send(conn, resp)

    def _send(self, conn: socket.socket, obj: Dict[str, Any]) -> None:
        try:
            conn.sendall((json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        except OSError:
            pass

    def push_event(self, etype: str, data: Dict[str, Any]) -> None:
        self.record_event(etype, data)

    def stop(self) -> None:
        self._closing.set()
        self._running.clear()
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass


def start_bridge(win, port: int, recorder: Optional[Recorder] = None) -> GuiBridgeServer:
    bridge = GuiBridgeServer(win, port=port, recorder=recorder)
    win._mcp_bridge = bridge
    bridge.start()
    return bridge


# ---------------------------------------------------------------------------
# GUI 端安全埋点钩子（UI 代码调用，bridge 不存在时静默跳过）
# ---------------------------------------------------------------------------

def get_bridge(win) -> Optional[GuiBridgeServer]:
    return getattr(win, "_mcp_bridge", None)


def emit_event(win, etype: str, data: Dict[str, Any]) -> None:
    """GUI 用户操作埋点：bridge 存在则记录+推送，否则静默忽略。"""
    try:
        b = getattr(win, "_mcp_bridge", None)
        if b is not None:
            b.record_event(etype, data)
    except Exception:
        pass


def emit_sam_mask(win, mask: np.ndarray, file_path: str,
                  score: float = 0.0, source: str = "sam_interactive",
                  note: str = "") -> Optional[str]:
    """GUI 端 SAM mask 埋点：保存 mask 落盘 + 计算统计 + 记录 sam_mask_updated 事件。"""
    try:
        b = getattr(win, "_mcp_bridge", None)
        if b is None:
            return None
        data = record_sam_mask(win, mask, file_path, score=score, source=source, note=note)
        b.record_event("sam_mask_updated", data)
        return data.get("mask_id")
    except Exception:
        return None
