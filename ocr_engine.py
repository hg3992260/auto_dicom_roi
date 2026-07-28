import re
import numpy as np
from typing import Dict, List, Optional
from rapidocr_onnxruntime import RapidOCR


class OcrEngine:
    def __init__(self):
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            has_dml = 'DmlExecutionProvider' in providers
        except Exception:
            has_dml = False
        self._engine = RapidOCR(
            box_thresh=0.3,
            unclip_ratio=1.8,
            det_use_dml=has_dml,
            cls_use_dml=has_dml,
            rec_use_dml=has_dml,
        )

    def extract(self, image: np.ndarray) -> List[Dict]:
        gray = image if image.ndim == 2 else image[:, :, 0] if image.shape[2] == 3 else image
        h, w = gray.shape[:2]
        if w < 500 or h < 500:
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        result, _ = self._engine(gray)
        if result is None:
            return []
        blocks = []
        for item in result:
            box, text, conf = item
            text = text.strip()
            if not text:
                continue
            x_min = min(p[0] for p in box)
            y_min = min(p[1] for p in box)
            x_max = max(p[0] for p in box)
            y_max = max(p[1] for p in box)
            scale_x = w / gray.shape[1]
            scale_y = h / gray.shape[0]
            blocks.append({
                "text": text,
                "bbox": (int(x_min * scale_x), int(y_min * scale_y),
                         int((x_max - x_min) * scale_x), int((y_max - y_min) * scale_y)),
                "confidence": float(conf),
                "floats": extract_floats(text),
            })
        return blocks

    def chunk(self, blocks: List[Dict]) -> List[Dict]:
        if not blocks:
            return []
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
        groups = []
        current = [sorted_blocks[0]]
        for b in sorted_blocks[1:]:
            prev_y_bottom = current[-1]["bbox"][1] + current[-1]["bbox"][3]
            gap = b["bbox"][1] - prev_y_bottom
            if gap < 30:
                current.append(b)
            else:
                groups.append(current)
                current = [b]
        if current:
            groups.append(current)

        records = []
        for g in groups:
            all_text = " ".join(b["text"] for b in g)
            all_floats = []
            for b in g:
                all_floats.extend(b.get("floats", []))
            y_min = min(b["bbox"][1] for b in g)
            y_max = max(b["bbox"][1] + b["bbox"][3] for b in g)

            label = ""
            min_v = max_v = mean_v = std_v = area_v = None
            for b in g:
                t = b["text"]
                if re.search(r'(?i)ROI?\d', t):
                    label = t
                elif 'min' in t.lower() or 'max' in t.lower() or re.search(r'\u6700\u5c0f|\u6700\u5927', t):
                    f = b.get("floats", [])
                    if len(f) >= 2:
                        min_v, max_v = f[0], f[1]
                elif 'mean' in t.lower() or 'std' in t.lower() or re.search(r'\u5e73\u5747|\u6807\u51c6', t):
                    f = b.get("floats", [])
                    if len(f) >= 2:
                        mean_v, std_v = f[0], f[1]
                elif 'area' in t.lower() or re.search(r'\u9762\u79ef|mm', t):
                    f = b.get("floats", [])
                    if f:
                        area_v = f[0]

            records.append({
                "label": label,
                "text": all_text[:200],
                "y_range": (y_min, y_max),
                "all_floats": all_floats[:10],
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "std": std_v,
                "area_mm2": area_v,
            })
        return records


def extract_floats(ocr_text: str) -> List[float]:
    text = ocr_text.replace("。", ".").replace("，", ".").replace(",", ".")
    pattern = r'[-+]?\d+(?:\.\d+)?'
    matches = re.findall(pattern, text)
    try:
        return [float(m) for m in matches]
    except (ValueError, TypeError):
        return []


import cv2
