import os
import json
import time
from typing import Dict, List, Any

import numpy as np
import cv2


def render_overlay(image: np.ndarray, rois: List[Dict],
                   alpha: float = 0.4) -> np.ndarray:
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        overlay = image.copy()
    else:
        overlay = cv2.cvtColor(image[:, :, :3],
                                cv2.COLOR_RGB2BGR if image.shape[2] == 4
                                else image)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
    ]

    mask_overlay = np.zeros_like(overlay, dtype=np.uint8)

    for ri, roi in enumerate(rois):
        contour = roi.get("contour", roi.get("contour_points", []))
        bbox = roi.get("bbox", {})
        color = colors[ri % len(colors)]

        if contour and isinstance(contour, list) and len(contour) >= 3:
            pts = np.array(contour, dtype=np.int32)
            cv2.fillPoly(mask_overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 2)
        elif bbox:
            x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), \
                         bbox.get("width", 0), bbox.get("height", 0)
            if w > 0 and h > 0:
                cv2.rectangle(mask_overlay, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        label = roi.get("roi_id", f"roi_{ri}")
        text_x = bbox.get("x", 10) if bbox else 10
        text_y = bbox.get("y", 10) - 5 if bbox else 15 + ri * 20
        if text_y < 15:
            text_y = 15
        cv2.putText(overlay, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    result = cv2.addWeighted(overlay, 1.0, mask_overlay, alpha, 0)
    return result


def save_results(file_path: str, rois: List[Dict], output_dir: str,
                 method: str = "roi", image: np.ndarray = None) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    basename = os.path.splitext(os.path.basename(file_path))[0]

    json_path = os.path.join(output_dir,
                              f"{basename}_{method}_{ts}.json")
    clean_rois = []
    for r in rois:
        clean = {k: v for k, v in r.items()
                 if k not in ("_file", "_size_kb", "_error")}
        clean_rois.append(clean)

    result = {
        "file": file_path,
        "method": method,
        "timestamp": ts,
        "roi_count": len(clean_rois),
        "rois": clean_rois,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    png_path = ""
    if image is not None and clean_rois:
        overlay = render_overlay(image, clean_rois)
        png_path = os.path.join(output_dir,
                                 f"{basename}_{method}_{ts}.png")
        cv2.imwrite(png_path, overlay)

    return {"json": json_path, "png": png_path}
