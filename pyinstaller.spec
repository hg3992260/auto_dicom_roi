# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DICOM Analysis Tool"""

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

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
    'openpyxl', 'scipy', 'scipy.ndimage',
    'segment_anything',
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
]

datas = [(str(Path(src)), dst) for src, dst in added_files]
datas += collect_data_files('PySide6')
datas += collect_data_files('rapidocr_onnxruntime')
datas += collect_data_files('onnxruntime')

binaries = collect_dynamic_libs('onnxruntime')

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
