# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DICOM Analysis Tool"""

import importlib
import os
import platform
import sys
import traceback
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()


def require_module(module_name):
    """Fail fast during build instead of shipping a broken package."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        tb = traceback.format_exc(limit=12)
        raise SystemExit(
            f"[pyinstaller.spec] Missing or broken dependency: {module_name}\n"
            f"reason: {detail or repr(exc)}\n"
            f"traceback:\n{tb}"
        ) from exc


def warn_module(module_name):
    """Warn but don't fail if an optional module is missing."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        print(f"[pyinstaller.spec] WARNING: optional module {module_name} not found: {exc}")
        return None


def safe_collect_submodules(module_name):
    try:
        return collect_submodules(module_name)
    except Exception as exc:
        print(f"[pyinstaller.spec] WARNING: cannot collect submodules for {module_name}: {exc}")
        return []


def extend_unique(target, items):
    seen = set(target)
    for item in items:
        if item not in seen:
            target.append(item)
            seen.add(item)


def extend_pairs_unique(target, items):
    seen = set(target)
    for item in items:
        if item not in seen:
            target.append(item)
            seen.add(item)


def merge_package_all(package_name, datas_target, binaries_target, hidden_target):
    datas_pkg, binaries_pkg, hidden_pkg = collect_all(package_name)
    extend_pairs_unique(datas_target, datas_pkg)
    extend_pairs_unique(binaries_target, binaries_pkg)
    extend_unique(hidden_target, hidden_pkg)


def merge_package_files(package_name, datas_target, binaries_target):
    extend_pairs_unique(datas_target, collect_data_files(package_name))
    try:
        extend_pairs_unique(binaries_target, collect_dynamic_libs(package_name))
    except Exception:
        pass


IS_MAC = platform.system() == "Darwin"

require_module("torch")
warn_module("torchvision")
require_module("segment_anything")
require_module("onnxruntime")
require_module("rapidocr_onnxruntime")
require_module("openpyxl")
require_module("et_xmlfile")
require_module("pydicom")
require_module("skimage")
require_module("imageio")
require_module("tifffile")
require_module("lazy_loader")
require_module("yaml")
require_module("tqdm")
require_module("shapely")
require_module("pyclipper")

added_files = [
    ('logo/Gemini_Generated_Image_egithcegithcegit.png', 'logo'),
]

# PyCt6 theme JSON and assets
import PyCt6 as _pyct6
_pyct6_dir = Path(_pyct6.__file__).parent
for _subdir, _target in [
    ('widgets/themes', 'PyCt6/widgets/themes'),
    ('widgets/images', 'PyCt6/widgets/images'),
    ('windows/images', 'PyCt6/windows/images'),
]:
    _d = _pyct6_dir / _subdir
    if _d.exists():
        for _f in _d.iterdir():
            added_files.append((str(_f), _target))

# RapidOCR ONNX model files
for p in sys.path:
    candidate = Path(p) / 'rapidocr_onnxruntime' / 'models'
    if candidate.exists():
        for onnx_file in candidate.glob('*.onnx'):
            added_files.append((str(onnx_file), 'rapidocr_onnxruntime/models'))
        break

# Hidden imports
hidden_imports = [
    'pydicom', 'pydicom.encaps', 'pydicom.valuerep',
    'numpy', 'numpy._core', 'numpy.core',
    'cv2', 'PIL', 'PIL.Image', 'skimage',
    'rapidocr_onnxruntime',
    'rapidocr_onnxruntime.ch_ppocr_det',
    'rapidocr_onnxruntime.ch_ppocr_det.text_detect',
    'rapidocr_onnxruntime.ch_ppocr_rec',
    'rapidocr_onnxruntime.ch_ppocr_rec.text_recognize',
    'rapidocr_onnxruntime.ch_ppocr_cls',
    'rapidocr_onnxruntime.ch_ppocr_cls.text_cls',
    'rapidocr_onnxruntime.utils',
    'rapidocr_onnxruntime.main',
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.providers',
    'et_xmlfile',
    'imageio',
    'tifffile',
    'lazy_loader',
    'yaml',
    'tqdm',
    'shapely',
    'pyclipper',
    'openpyxl', 'scipy', 'scipy.ndimage',
    'segment_anything',
    'segment_anything.build_sam',
    'segment_anything.predictor',
    'segment_anything.automatic_mask_generator',
    'segment_anything.modeling',
    'segment_anything.modeling.sam',
    'segment_anything.modeling.image_encoder',
    'segment_anything.modeling.mask_decoder',
    'segment_anything.modeling.prompt_encoder',
    'segment_anything.modeling.transformer',
    'segment_anything.modeling.common',
    'segment_anything.utils',
    'segment_anything.utils.amg',
    'segment_anything.utils.transforms',
    'segment_anything.utils.onnx',
    'torchvision',
    'torchvision.io',
    'torchvision.ops',
    'torchvision.transforms',
    'torchvision.transforms.functional',
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.utils',
    'torch.serialization', 'torch.multiprocessing',
]
# Collect package submodules aggressively for bundled runtime imports.
extend_unique(hidden_imports, safe_collect_submodules('torch'))
extend_unique(hidden_imports, safe_collect_submodules('torchvision'))
extend_unique(hidden_imports, safe_collect_submodules('segment_anything'))
extend_unique(hidden_imports, safe_collect_submodules('onnxruntime'))
extend_unique(hidden_imports, safe_collect_submodules('rapidocr_onnxruntime'))

datas = [(str(Path(src)), dst) for src, dst in added_files]
datas += collect_data_files('PySide6')
datas += collect_data_files('rapidocr_onnxruntime')
datas += collect_data_files('onnxruntime')
datas += collect_data_files('segment_anything')
datas += collect_data_files('torch')
try:
    datas += collect_data_files('torchvision')
except Exception:
    pass

# torch: collect all binaries and data files aggressively
binaries = collect_dynamic_libs('torch')
try:
    binaries += collect_dynamic_libs('torchvision')
except Exception:
    pass
binaries += collect_dynamic_libs('onnxruntime')

for package_name in [
    'openpyxl',
    'pydicom',
    'skimage',
]:
    merge_package_all(package_name, datas, binaries, hidden_imports)

for package_name in [
    'et_xmlfile',
    'imageio',
    'tifffile',
    'lazy_loader',
    'yaml',
    'tqdm',
    'shapely',
    'pyclipper',
]:
    merge_package_files(package_name, datas, binaries)

# Explicitly add torch native libraries into torch/lib.
# PyTorch resolves several binaries relative to the package directory,
# so flattening them into "." can cause runtime import failures.
_torch_check = require_module("torch")
_torch_dir = Path(_torch_check.__file__).parent
_torch_lib = _torch_dir / 'lib'
if _torch_lib.exists():
    for pattern in ('*.dll', '*.so', '*.dylib', '*.pyd'):
        for native_lib in _torch_lib.glob(pattern):
            binaries.append((str(native_lib), 'torch/lib'))

a = Analysis(
    ['main.py'],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pandas', 'jupyter'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DICOM_Analysis_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX frequently breaks torch/onnxruntime native libraries.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo/Gemini_Generated_Image_egithcegithcegit.png' if Path('logo/Gemini_Generated_Image_egithcegithcegit.png').exists() else None,
)

# On macOS, wrap EXE in a .app bundle
if IS_MAC:
    app = BUNDLE(
        exe,
        name='DICOM_Analysis_Tool.app',
        icon='logo/Gemini_Generated_Image_egithcegithcegit.png' if Path('logo/Gemini_Generated_Image_egithcegithcegit.png').exists() else None,
        bundle_identifier='com.dicom.analysis.tool',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSMinimumSystemVersion': '11.0',
        },
    )
