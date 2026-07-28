import os
import json
import time
from typing import Dict, List, Optional, Any, Callable

import numpy as np
import cv2
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from skimage import morphology, measure, segmentation as skseg
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class RoiEngine:
    METHODS = ["overlay", "ocr"]

    def __init__(self):
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required")

    def _load_dcm_image(self, file_path: str) -> Optional[np.ndarray]:
        try:
            ds = pydicom.dcmread(file_path, force=True)
        except Exception:
            return None
        try:
            pixels = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixels = pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            if pixels.ndim == 3:
                pixels = pixels[:, :, 0]
            mn, mx = pixels.min(), pixels.max()
            if mx > mn:
                pixels = ((pixels - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                pixels = np.zeros_like(pixels, dtype=np.uint8)
            return pixels
        except Exception:
            return None

    def _get_kernel(self, size: int, shape: str = "ellipse") -> np.ndarray:
        size = max(3, size | 1)
        if shape == "ellipse":
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _preprocess_image(self, image: np.ndarray, params: Dict) -> np.ndarray:
        blur_size = params.get("blur", 3)
        if blur_size > 0:
            ksize = blur_size | 1
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image

    def detect(self, file_path: str, method: str = "overlay",
               params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {self.METHODS}")

        image = self._load_dcm_image(file_path)
        if image is None:
            return [{"error": "Failed to load DICOM image", "file": file_path}]

        dicom_params = self._extract_dicom_params(file_path)

        params = params or {}
        min_area = params.get("min_area", 50)
        max_area = params.get("max_area", None)
        kernel_size = params.get("kernel_size", 3)
        min_circularity = params.get("min_circularity", 0.1)
        max_aspect_ratio = params.get("max_aspect_ratio", 5.0)

        processed = self._preprocess_image(image, params)

        roi_regions = []

        if method == "overlay":
            ds = pydicom.dcmread(file_path, force=True)
            roi_regions = self._extract_roi_from_overlay(
                ds, image.shape[:2], min_area, max_area or 999999,
                dicom_params)
        elif method == "otsu":
            roi_regions = self._detect_otsu(
                processed, min_area, max_area, kernel_size,
                min_circularity, max_aspect_ratio, dicom_params)
        elif method == "adaptive":
            roi_regions = self._detect_adaptive(
                processed, min_area, max_area, kernel_size,
                params.get("block_size", 11), params.get("c", 2),
                min_circularity, max_aspect_ratio, dicom_params)
        elif method == "watershed":
            roi_regions = self._detect_watershed(
                processed, min_area, max_area, kernel_size,
                min_circularity, max_aspect_ratio, dicom_params)
        elif method == "edge":
            roi_regions = self._detect_edge(
                processed, min_area, max_area, kernel_size,
                min_circularity, max_aspect_ratio, dicom_params)
        elif method == "ocr":
            ds = pydicom.dcmread(file_path, force=True)
            roi_regions = self._extract_roi_from_overlay(
                ds, image.shape[:2], min_area, max_area or 999999,
                dicom_params)

        for roi in roi_regions:
            if 'source' not in roi:
                roi['source'] = roi.get('source', 'image_processing')

        return roi_regions

    def _extract_dicom_params(self, file_path: str) -> Dict:
        try:
            ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            info = {}
            mod = str(getattr(ds, 'Modality', '')).strip()
            info['modality'] = mod
            ps = getattr(ds, 'PixelSpacing', None)
            if ps is not None and len(ps) >= 2:
                info['pixel_spacing_x'] = float(ps[0])
                info['pixel_spacing_y'] = float(ps[1])
                info['pixel_area_mm2'] = float(ps[0]) * float(ps[1])
            else:
                info['pixel_spacing_x'] = 1.0
                info['pixel_spacing_y'] = 1.0
                info['pixel_area_mm2'] = 1.0
            return info
        except Exception:
            return {'modality': '', 'pixel_spacing_x': 1.0,
                    'pixel_spacing_y': 1.0, 'pixel_area_mm2': 1.0}

    def _compute_min_area_mm2(self, modality: str, pixel_size_mm: float) -> float:
        if modality == 'CT':
            if pixel_size_mm < 0.3:
                return 3.0
            else:
                return 30.0
        elif modality == 'MR':
            return 70.0
        else:
            return 0.0

    def _extract_contours(self, binary: np.ndarray, image: np.ndarray,
                          min_area: int, max_area: Optional[int],
                          min_circularity: float = 0.1,
                          max_aspect_ratio: float = 5.0,
                          dicom_params: Optional[Dict] = None,
                          skip_clinical: bool = False) -> List[Dict]:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dp = dicom_params or {}
        modality = dp.get('modality', '')
        px_area_mm2 = dp.get('pixel_area_mm2', 1.0)
        min_area_mm2 = self._compute_min_area_mm2(modality, px_area_mm2 ** 0.5)
        min_pixels = 30

        rois = []
        for ci, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area or (max_area and area > max_area):
                continue
            if area < min_pixels:
                continue
            physical_area = area * px_area_mm2
            if not skip_clinical and min_area_mm2 > 0 and physical_area < min_area_mm2:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else float('inf')
            if aspect_ratio > max_aspect_ratio or aspect_ratio < 1.0 / max_aspect_ratio:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < min_circularity:
                continue

            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            roi_pixels = image[mask > 0]
            epsilon = 0.001 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            stats = {}
            if len(roi_pixels) > 0:
                stats = {
                    "mean": float(np.mean(roi_pixels)),
                    "std": float(np.std(roi_pixels)),
                    "min": float(np.min(roi_pixels)),
                    "max": float(np.max(roi_pixels)),
                    "pixel_count": int(len(roi_pixels)),
                }
            rois.append({
                "roi_id": f"roi_{ci}",
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area": float(area),
                "area_mm2": float(physical_area),
                "perimeter": float(perimeter),
                "circularity": float(circularity),
                "contour": approx.reshape(-1, 2).tolist(),
                "centroid": {"x": int(x + w // 2), "y": int(y + h // 2)},
                "statistics": stats,
            })
        return rois

    def _detect_otsu(self, image: np.ndarray, min_area: int,
                     max_area: Optional[int], kernel_size: int = 3,
                     min_circularity: float = 0.1,
                     max_aspect_ratio: float = 5.0,
                     dicom_params: Optional[Dict] = None) -> List[Dict]:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = self._get_kernel(kernel_size)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return self._extract_contours(
            binary, image, min_area, max_area, min_circularity, max_aspect_ratio,
            dicom_params)

    def _detect_adaptive(self, image: np.ndarray, min_area: int,
                         max_area: Optional[int], kernel_size: int = 3,
                         block_size: int = 11, c_val: int = 2,
                         min_circularity: float = 0.1,
                         max_aspect_ratio: float = 5.0,
                         dicom_params: Optional[Dict] = None) -> List[Dict]:
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size | 1, c_val)
        kernel = self._get_kernel(kernel_size)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return self._extract_contours(
            binary, image, min_area, max_area, min_circularity, max_aspect_ratio,
            dicom_params)

    def _detect_watershed(self, image: np.ndarray, min_area: int,
                          max_area: Optional[int], kernel_size: int = 3,
                          min_circularity: float = 0.1,
                          max_aspect_ratio: float = 5.0,
                          dicom_params: Optional[Dict] = None) -> List[Dict]:
        _, thresh = cv2.threshold(image, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = self._get_kernel(kernel_size)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.watershed(color_img, markers)
        binary = np.zeros(image.shape, dtype=np.uint8)
        for label_id in range(2, markers.max() + 1):
            binary[markers == label_id] = 255
        return self._extract_contours(
            binary, image, min_area, max_area, min_circularity, max_aspect_ratio,
            dicom_params)

    def _detect_edge(self, image: np.ndarray, min_area: int,
                     max_area: Optional[int], kernel_size: int = 3,
                     min_circularity: float = 0.1,
                     max_aspect_ratio: float = 5.0,
                     dicom_params: Optional[Dict] = None) -> List[Dict]:
        edges = cv2.Canny(image, 50, 150)
        kernel = self._get_kernel(kernel_size)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return self._extract_contours(
            edges, image, min_area, max_area, min_circularity, max_aspect_ratio,
            dicom_params)

    def _extract_roi_from_overlay(self, dicom_data, image_shape,
                                   min_area: int, max_area: int,
                                   dicom_params: Optional[Dict] = None) -> List[Dict]:
        roi_regions = []

        def _process_overlay_binary(overlay_binary, image, mina, maxa):
            """Run connected component filtering then contour extraction"""
            overlay_binary = overlay_binary.astype(np.uint8) * 255
            # Connected component analysis to split lines from solid regions
            num_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
                overlay_binary, connectivity=8)
            clean = np.zeros_like(overlay_binary)
            for i in range(1, num_labels):
                area = stats_cc[i, cv2.CC_STAT_AREA]
                if area < mina:
                    continue
                # Extract component mask and compute shape metrics
                comp_mask = (labels == i).astype(np.uint8) * 255
                comp_cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not comp_cnts:
                    continue
                cnt = max(comp_cnts, key=cv2.contourArea)
                peri = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
                # Filter out line-like components (very low circularity + small area = thin strokes)
                if circ < 0.05:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                ar = w / h if h > 0 else float('inf')
                if ar > 15 and area < 100:
                    continue
                clean[labels == i] = 255
            if clean.sum() == 0:
                return []
            return self._extract_contours(
                clean, image, mina, maxa, 0.1, 5.0,
                dicom_params, skip_clinical=True)

        # Method 1: pydicom built-in overlay_array
        for group in range(0x6000, 0x601F, 2):
            try:
                if hasattr(dicom_data, 'overlay_array'):
                    overlays = dicom_data.overlay_array(group)
                    if overlays is not None and overlays.size > 0:
                        image = self._load_dcm_image_via_ds(dicom_data)
                        if image is not None:
                            new_rois = _process_overlay_binary(
                                overlays > 0, image, min_area, max_area)
                            for r in new_rois:
                                r["source"] = "roi_overlay"
                            roi_regions.extend(new_rois)
            except Exception:
                pass

        # Method 2: manual parsing (6000 groups)
        if not roi_regions:
            for group in range(0x6000, 0x601F, 2):
                data_tag = (group, 0x3000)
                rows_tag = (group, 0x0010)
                cols_tag = (group, 0x0011)
                if data_tag not in dicom_data:
                    continue
                try:
                    rows = int(dicom_data[rows_tag].value) if rows_tag in dicom_data else image_shape[0]
                    cols = int(dicom_data[cols_tag].value) if cols_tag in dicom_data else image_shape[1]
                    raw = dicom_data[data_tag].value
                    if not isinstance(raw, bytes):
                        continue
                    total_bits = rows * cols
                    required_bytes = (total_bits + 7) // 8
                    if len(raw) < required_bytes:
                        raw = raw + b'\x00' * (required_bytes - len(raw))
                    elif len(raw) > required_bytes:
                        raw = raw[:required_bytes]
                    overlay_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
                    overlay_bits = overlay_bits[:total_bits]
                    overlay_binary = (overlay_bits.reshape(rows, cols) > 0)
                    image = self._load_dcm_image_via_ds(dicom_data)
                    if image is not None:
                        new_rois = _process_overlay_binary(
                            overlay_binary, image, min_area, max_area)
                        for r in new_rois:
                            r["source"] = "roi_overlay"
                        roi_regions.extend(new_rois)
                except Exception:
                    continue

        # Method 3: RT Structure Set
        if not roi_regions:
            try:
                mod = getattr(dicom_data, 'Modality', '')
                if mod == 'RTSTRUCT' and hasattr(dicom_data, 'ROIContourSequence'):
                    for roi_contour in dicom_data.ROIContourSequence:
                        if hasattr(roi_contour, 'ContourSequence'):
                            for contour in roi_contour.ContourSequence:
                                if hasattr(contour, 'ContourData'):
                                    contour_data = contour.ContourData
                                    points = np.array(contour_data).reshape(-1, 3)[:, :2]
                                    cv_contour = points.astype(np.int32).reshape(-1, 1, 2)
                                    area = cv2.contourArea(cv_contour)
                                    if min_area <= area <= max_area:
                                        x, y, w, h = cv2.boundingRect(cv_contour)
                                        roi_regions.append({
                                            "roi_id": f"rt_roi_{len(roi_regions)+1}",
                                            "bbox": {"x": int(x), "y": int(y),
                                                     "width": int(w), "height": int(h)},
                                            "area": float(area),
                                            "contour": cv_contour.reshape(-1, 2).tolist(),
                                            "centroid": {"x": int(x+w//2), "y": int(y+h//2)},
                                            "source": "roi_overlay",
                                        })
            except Exception:
                pass

        return roi_regions

    def _load_dcm_image_via_ds(self, ds) -> Optional[np.ndarray]:
        try:
            pixels = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixels = pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            if pixels.ndim == 3:
                pixels = pixels[:, :, 0]
            mn, mx = pixels.min(), pixels.max()
            if mx > mn:
                pixels = ((pixels - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                pixels = np.zeros_like(pixels, dtype=np.uint8)
            return pixels
        except Exception:
            return None

    def extract_overlay_mask(self, file_path: str) -> Optional[np.ndarray]:
        try:
            ds = pydicom.dcmread(file_path, force=True)
        except Exception:
            return None
        all_overlays = None
        h, w = 0, 0
        try:
            pixels = ds.pixel_array
            h, w = pixels.shape[:2]
        except Exception:
            pass

        # Method 1: pydicom built-in
        for group in range(0x6000, 0x601F, 2):
            try:
                if hasattr(ds, 'overlay_array'):
                    overlays = ds.overlay_array(group)
                    if overlays is not None and overlays.size > 0:
                        if all_overlays is None:
                            all_overlays = overlays.astype(bool)
                        else:
                            all_overlays |= overlays.astype(bool)
            except Exception:
                pass

        # Method 2: manual parsing
        if all_overlays is None:
            for group in range(0x6000, 0x601F, 2):
                data_tag = (group, 0x3000)
                rows_tag = (group, 0x0010)
                cols_tag = (group, 0x0011)
                if data_tag not in ds:
                    continue
                try:
                    rows = int(ds[rows_tag].value) if rows_tag in ds else h
                    cols = int(ds[cols_tag].value) if cols_tag in ds else w
                    raw = ds[data_tag].value
                    if not isinstance(raw, bytes):
                        continue
                    total_bits = rows * cols
                    required_bytes = (total_bits + 7) // 8
                    if len(raw) < required_bytes:
                        raw = raw + b'\x00' * (required_bytes - len(raw))
                    elif len(raw) > required_bytes:
                        raw = raw[:required_bytes]
                    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
                    overlay = bits[:total_bits].reshape(rows, cols).astype(bool)
                    if all_overlays is None:
                        all_overlays = overlay
                    else:
                        all_overlays |= overlay
                except Exception:
                    continue

        return all_overlays

    def render_overlay_for_ocr(self, file_path: str) -> Optional[np.ndarray]:
        mask = self.extract_overlay_mask(file_path)
        if mask is None or mask.sum() == 0:
            return None
        canvas = np.zeros(mask.shape, dtype=np.uint8)
        canvas[mask] = 255
        return canvas
