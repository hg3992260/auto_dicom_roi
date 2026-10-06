"""DICOM Analysis Tool MCP 服务：GUI 事件记录 + 桥接读取。

- 在 GUI 进程内嵌 recorder（SQLite+JSONL 双写）与 TCP bridge，
  用户每次 SAM 交互/ROI 保存/文件加载等操作都会被持久化记录；
- MCP server（stdio/FastMCP）读取记录、查询状态、截图、导出，
  供后续计算与复盘。
"""
from __future__ import annotations

__version__ = "1.0.0"
