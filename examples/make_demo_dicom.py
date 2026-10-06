# -*- coding: utf-8 -*-
"""
生成**合成演示 DICOM 数据集**（无任何真实患者信息）
==================================================
用途：README 截图、功能试用、CI 冒烟测试。
- 假患者：DEMO^ANON / DEMO001
- 合成图像：噪声背景 + 组织椭圆 + 细亮"神经"曲线
- 含 6000 组 Overlay Plane（LSB 位序），使 overlay ROI 检测可用

用法: python make_demo_dicom.py <输出目录> [层数]
"""
from __future__ import annotations
import os, sys, datetime
import numpy as np
import cv2

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage


def bits_lsb(mask: np.ndarray) -> bytes:
    """按 LSB-first 位序打包二值掩膜（DICOM Overlay 标准）。"""
    flat = mask.astype(np.uint8).ravel()
    return np.packbits(flat, bitorder='little').tobytes()


def make_slice(idx: int, size: int = 256, seed: int = 20260916) -> tuple[np.ndarray, np.ndarray]:
    """返回 (灰度图 uint16, overlay 掩膜 bool)"""
    rng = np.random.default_rng(seed + idx * 97)
    img = rng.normal(60, 14, (size, size))

    # 组织椭圆
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = ((yy - cy) / (size * 0.42)) ** 2 + ((xx - cx) / (size * 0.30)) ** 2
    img[r <= 1.0] += 150
    img[(r <= 0.30)] += 90          # 高信号核心

    # 细亮"神经"曲线（合成走行）
    t = np.linspace(-1.0, 1.0, 90)
    px = (cx + size * 0.16 * np.sin(2.2 * t + idx * 0.25) + size * 0.05 * idx).astype(int)
    py = (cy + size * 0.30 * t).astype(int)
    ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px, py = px[ok], py[ok]
    pts = np.stack([px, py], 1).reshape(-1, 1, 2)
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.polylines(img8, [pts], False, 235, 1)
    img = img.astype(np.float32)
    img[py, px] = 235

    # Overlay：细曲线 + 一个圆形 ROI（圆形度足够，便于 overlay 检测出轮廓）
    ovl = np.zeros((size, size), np.uint8)
    cv2.polylines(ovl, [pts], False, 1, 1)
    blob_c = (int(cx + size * 0.20 + idx * 3), int(cy - size * 0.05))
    cv2.circle(ovl, blob_c, 7, 1, -1)
    img8[ovl.astype(bool)] = np.maximum(img8[ovl.astype(bool)], 225)
    img = img8.astype(np.float32)

    return np.clip(img, 0, 4095).astype(np.uint16), ovl.astype(bool)


def make_one(idx: int, n: int, series_uid: str, study_uid: str) -> FileDataset:
    px, ovl = make_slice(idx)
    size = px.shape[0]

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()

    ds = FileDataset('', {}, file_meta=fm, preamble=b'\0' * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID

    ds.PatientName = 'DEMO^ANON'
    ds.PatientID = 'DEMO001'
    ds.PatientBirthDate = '19700101'
    ds.PatientSex = 'O'
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.StudyID = 'DEMOSTUDY'
    ds.AccessionNumber = 'DEMO0001'
    ds.StudyDate = ds.SeriesDate = ds.ContentDate = '20260101'
    ds.StudyTime = ds.SeriesTime = ds.ContentTime = '120000'

    ds.Modality = 'MR'
    ds.Manufacturer = 'DEMO'
    ds.ManufacturerModelName = 'SyntheticPhantom'
    ds.InstitutionName = 'DEMO INSTITUTION'
    ds.StationName = 'DEMO-STATION'
    ds.SeriesDescription = 'Demo 3D-STIR-VISTA (synthetic)'
    ds.ProtocolName = 'DEMO_FACIAL_NERVE'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M']

    ds.Rows, ds.Columns = size, size
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 2.0
    ds.SpacingBetweenSlices = 2.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(idx) * 2.0]
    ds.SliceLocation = float(idx) * 2.0
    ds.InstanceNumber = idx + 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 200
    ds.WindowWidth = 400
    ds.PixelData = px.tobytes()

    # ---- 6000 组 Overlay Plane（LSB 位序）----
    ds.add_new(0x60000010, 'US', size)              # OverlayRows
    ds.add_new(0x60000011, 'US', size)              # OverlayColumns
    ds.add_new(0x60000015, 'IS', 1)                 # NumberOfFramesInOverlay
    ds.add_new(0x60000022, 'LO', 'DEMO_ANNOTATION')
    ds.add_new(0x60000040, 'CS', 'G')               # OverlayType = Graphics
    ds.add_new(0x60000045, 'LO', 'DEMO region')
    ds.add_new(0x60000050, 'SS', [1, 1])            # OverlayOrigin
    ds.add_new(0x60000100, 'US', 1)                 # OverlayBitsAllocated
    ds.add_new(0x60000102, 'US', 0)                 # OverlayBitPosition
    ds.add_new(0x60003000, 'OW', bits_lsb(ovl))     # OverlayData
    return ds


def main():
    out = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_dicom'))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    folder = os.path.join(out, 'DEMO001')
    os.makedirs(folder, exist_ok=True)
    series_uid, study_uid = generate_uid(), generate_uid()
    for i in range(n):
        ds = make_one(i, n, series_uid, study_uid)
        p = os.path.join(folder, f'{i+1:05d}.dcm')
        ds.save_as(p, write_like_original=False)
    total = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
    print(f'已生成 {n} 个合成 DICOM -> {folder}  ({total/1024:.0f} KB)')


if __name__ == '__main__':
    main()
