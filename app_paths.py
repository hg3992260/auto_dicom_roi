from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


APP_NAME = "DICOM_Analysis_Tool"
SAM_CHECKPOINT_NAMES = (
    "sam_vit_b_01ec64.pth",
    "sam_vit_l_0b3195.pth",
    "sam_vit_h_4b8939.pth",
)


def get_runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_bundle_root() -> Path | None:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return None
    try:
        return Path(bundle_root).resolve()
    except Exception:
        return Path(bundle_root)


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
    roots = []
    bundle_root = get_bundle_root()
    if bundle_root is not None:
        roots.append(bundle_root)
    roots.extend([get_runtime_root(), get_user_data_dir(), Path.cwd().resolve()])
    roots.extend(get_runtime_root().parents)
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved / "models"


def iter_checkpoint_candidates():
    seen = set()

    def _yield(path: Path):
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        yield resolved

    roots = []
    bundle_root = get_bundle_root()
    if bundle_root is not None:
        roots.append(bundle_root)
    runtime_root = get_runtime_root()
    roots.extend([runtime_root, get_user_data_dir(), Path.cwd().resolve()])
    roots.extend(runtime_root.parents)

    for model_dir in iter_model_search_dirs():
        for name in SAM_CHECKPOINT_NAMES:
            if (model_dir / name).exists():
                yield from _yield(model_dir / name)

    for root in roots:
        for name in SAM_CHECKPOINT_NAMES:
            candidate = root / name
            if candidate.exists():
                yield from _yield(candidate)


def describe_model_search_paths() -> list[str]:
    paths = []
    for model_dir in iter_model_search_dirs():
        paths.append(str(model_dir))
    return paths
