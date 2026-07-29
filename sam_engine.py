import os
import sys
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple

import numpy as np
import cv2

from app_paths import iter_model_search_dirs

_torch_import_error = None
_torch_diag = ""
_sam_import_error = None
try:
    import torch
    HAS_TORCH = True
except ImportError as e:
    torch = None
    HAS_TORCH = False
    _torch_import_error = str(e)
    # Try to get more diagnostic info
    try:
        import traceback
        _torch_diag = traceback.format_exc()[-300:]
    except Exception:
        pass

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError as e:
    HAS_SAM = False
    _sam_import_error = str(e)

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class SamEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.available = HAS_TORCH and HAS_SAM and HAS_PYDICOM
        self.device = "cpu"
        if HAS_TORCH and torch.cuda.is_available():
            self.device = "cuda"
        self.model_type = "vit_b"
        self.checkpoint_path = ""
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        self._find_checkpoint()

    def _find_checkpoint(self):
        search_paths = []
        for model_dir in iter_model_search_dirs():
            search_paths.extend([
                os.path.join(model_dir, "sam_vit_b_01ec64.pth"),
                os.path.join(model_dir, "sam_vit_h_4b8939.pth"),
                os.path.join(model_dir, "sam_vit_l_0b3195.pth"),
            ])
        for p in search_paths:
            if os.path.exists(p):
                self.checkpoint_path = p
                if "vit_h" in p:
                    self.model_type = "vit_h"
                elif "vit_l" in p:
                    self.model_type = "vit_l"
                else:
                    self.model_type = "vit_b"
                return

    def set_checkpoint(self, path: str):
        self.checkpoint_path = path
        if "vit_h" in path:
            self.model_type = "vit_h"
        elif "vit_l" in path:
            self.model_type = "vit_l"
        else:
            self.model_type = "vit_b"
        self.sam = None
        self.predictor = None
        self.mask_generator = None

    def is_ready(self) -> bool:
        return self.available and bool(self.checkpoint_path)

    def get_status(self) -> str:
        if not HAS_PYDICOM:
            return "pydicom not installed"
        if not HAS_TORCH:
            detail = _torch_diag or _torch_import_error or '未安装'
            return f"PyTorch不可用: {detail[:120]}"
        if not HAS_SAM:
            detail = _sam_import_error or "unknown import error"
            return f"segment-anything不可用: {detail[:120]}"
        if not self.checkpoint_path:
            return "SAM checkpoint not found"
        if self.sam is not None:
            return f"SAM {self.model_type} loaded on {self.device}"
        return "Ready"

    def load_model(self, status_callback: Optional[Callable[[str], None]] = None) -> bool:
        if not self.available:
            if status_callback:
                status_callback("SAM unavailable: missing torch/SAM/pkg")
            return False
        if self.sam is not None:
            if status_callback:
                status_callback("SAM already loaded")
            return True
        if not self.checkpoint_path:
            if status_callback:
                status_callback("SAM checkpoint not found — 请在界面上选择 .pth 模型文件")
            return False
        if status_callback:
            status_callback("Loading SAM model...")
        try:
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam, points_per_side=32,
                pred_iou_thresh=0.86, stability_score_thresh=0.92,
                crop_n_layers=1, crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            if status_callback:
                status_callback(f"SAM {self.model_type} loaded on {self.device}")
            return True
        except Exception as e:
            err_msg = f"SAM load failed: {e} | path={self.checkpoint_path} | device={self.device}"
            if status_callback:
                status_callback(err_msg)
            if self.device == "cuda":
                try:
                    if status_callback:
                        status_callback("Retrying on CPU...")
                    self.device = "cpu"
                    self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                    self.sam.to(device="cpu")
                    self.predictor = SamPredictor(self.sam)
                    self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)
                    if status_callback:
                        status_callback("SAM loaded on CPU")
                    return True
                except Exception:
                    pass
            self.sam = None
            return False

    def set_image(self, image: np.ndarray):
        if self.predictor is None:
            self.load_model()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            pass
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        self._current_image = image

    def predict_mask(self, points: np.ndarray, labels: np.ndarray) -> Tuple:
        if self.predictor is None:
            raise RuntimeError("SAM not loaded. Call load_model() first.")
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        return masks, scores, logits

    def auto_segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.mask_generator is None:
            self.load_model()
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_outputs = self.mask_generator.generate(image_rgb)
        results = []
        for i, m in enumerate(mask_outputs):
            binary_mask = m["segmentation"]
            bbox = m["bbox"]
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            for ci, cnt in enumerate(contours):
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                roi_pixels = image[binary_mask] if image.ndim == 2 else None
                stats = {}
                if roi_pixels is not None and len(roi_pixels) > 0:
                    stats = {
                        "mean": float(np.mean(roi_pixels)),
                        "std": float(np.std(roi_pixels)),
                        "min": float(np.min(roi_pixels)),
                        "max": float(np.max(roi_pixels)),
                        "pixel_count": int(len(roi_pixels)),
                    }
                results.append({
                    "roi_id": f"sam_{i}_{ci}",
                    "method": "sam_automatic",
                    "source": "magic_seg_sam",
                    "area": float(m["area"]),
                    "bbox": {"x": int(bbox[0]), "y": int(bbox[1]),
                             "width": int(bbox[2]), "height": int(bbox[3])},
                    "contour": approx.reshape(-1, 2).tolist() if len(approx) > 2 else [],
                    "iou_score": float(m.get("predicted_iou", 0)),
                    "stability_score": float(m.get("stability_score", 0)),
                    "statistics": stats,
                })
        return results

    def process_file(self, file_path: str) -> Dict[str, Any]:
        if not HAS_PYDICOM:
            return {"success": False, "error": "pydicom not installed", "file": file_path}
        try:
            ds = pydicom.dcmread(file_path, force=True)
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
        except Exception as e:
            return {"success": False, "error": str(e), "file": file_path}
        try:
            self.load_model()
            rois = self.auto_segment(pixels)
            return {"success": True, "file": file_path, "roi_count": len(rois), "rois": rois}
        except Exception as e:
            return {"success": False, "error": str(e), "file": file_path}
