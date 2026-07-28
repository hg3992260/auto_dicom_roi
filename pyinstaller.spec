# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DICOM Analysis Tool"""

import os, sys
from pathlib import Path

ROOT = Path('.').absolute()

# RapidOCR ONNX model files
rapidocr_dir = None
for p in sys.path:
    candidate = Path(p) / 'rapidocr_onnxruntime' / 'models'
    if candidate.exists():
        rapidocr_dir = candidate
        break

added_files = [
    ('logo/Gemini_Generated_Image_egithcegithcegit.png', 'logo'),
]

if rapidocr_dir:
    for onnx_file in rapidocr_dir.glob('*.onnx'):
        added_files.append((str(onnx_file), 'rapidocr_onnxruntime/models'))

# Hidden imports for lazy-loaded modules
hidden_imports = [
    'pydicom', 'numpy', 'cv2', 'opencv-python',
    'PIL', 'skimage',
    'rapidocr_onnxruntime',
    'rapidocr_onnxruntime.ch_ppocr_det',
    'rapidocr_onnxruntime.ch_ppocr_rec',
    'rapidocr_onnxruntime.ch_ppocr_cls',
    'rapidocr_onnxruntime.utils',
    'onnxruntime',
    'PySide6',
    'PyCt6',
    'openpyxl',
    'scipy',
    'segment_anything',
]

datas = [(str(Path(src)), dst) for src, dst in added_files]

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
