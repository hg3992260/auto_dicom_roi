# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DICOM Analysis Tool"""

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path('.').absolute()

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
rapidocr_dir = None
for p in sys.path:
    candidate = Path(p) / 'rapidocr_onnxruntime' / 'models'
    if candidate.exists():
        rapidocr_dir = candidate
        break
if rapidocr_dir:
    for onnx_file in rapidocr_dir.glob('*.onnx'):
        added_files.append((str(onnx_file), 'rapidocr_onnxruntime/models'))

# Hidden imports for lazy-loaded modules
hidden_imports = [
    'pydicom', 'numpy', 'cv2', 'PIL', 'skimage',
    'rapidocr_onnxruntime',
    'rapidocr_onnxruntime.ch_ppocr_det',
    'rapidocr_onnxruntime.ch_ppocr_rec',
    'rapidocr_onnxruntime.ch_ppocr_cls',
    'rapidocr_onnxruntime.utils',
    'onnxruntime', 'openpyxl', 'scipy',
    'segment_anything',
]

datas = [(str(Path(src)), dst) for src, dst in added_files]
datas += collect_data_files('PySide6')

a = Analysis(
    ['main.py'],
    pathex=[str(ROOT)],
    binaries=[],
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
    upx=True,
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
