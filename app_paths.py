from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


APP_NAME = "DICOM_Analysis_Tool"


def get_runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_user_data_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    target = base / APP_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_output_dir() -> Path:
    target = get_user_data_dir() / "output"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_db_path() -> Path:
    return get_user_data_dir() / "roi_summary.db"


def iter_model_search_dirs():
    seen = set()
    roots = [get_runtime_root(), get_user_data_dir()]
    roots.extend(get_runtime_root().parents)
    roots.append(Path.cwd().resolve())
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved / "models"
