import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from PyCt6 import CApplication, set_appearance_mode

set_appearance_mode("light")

from ui.main_window import MainWindow


def _run_selftest() -> int:
    """Frozen-app self-test, triggered by DICOM_SELFTEST=1.
    Verifies every key module imports inside the bundled environment and that
    a SAM checkpoint (path given via DICOM_SELFTEST_MODEL) loads and runs.
    Writes results to the path in DICOM_SELFTEST_OUT (default: selftest.json)."""
    import json
    import time

    report = {"ok": True, "imports": {}, "sam": None, "elapsed_s": 0.0}
    t0 = time.time()
    modules = [
        "torch", "numpy", "cv2", "pydicom", "PIL", "skimage", "openpyxl",
        "onnxruntime", "rapidocr_onnxruntime", "segment_anything", "shapely",
        "PySide6", "PyCt6", "imageio", "tifffile", "scipy", "yaml", "tqdm",
    ]
    for m in modules:
        try:
            __import__(m)
            report["imports"][m] = "ok"
        except Exception as e:  # noqa: BLE001
            report["imports"][m] = f"FAIL: {type(e).__name__}: {e}"
            report["ok"] = False

    model_path = os.environ.get("DICOM_SELFTEST_MODEL", "").strip()
    if model_path and os.path.isfile(model_path):
        try:
            from sam_engine import SamEngine
            import numpy as np
            engine = SamEngine()
            if engine.device == "cpu":
                import torch
                if torch.backends.mps.is_available():
                    engine.device = "mps"
            engine.set_checkpoint(model_path)
            ok = engine.load_model()
            if ok:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                img[:, :] = 20
                img[64:192, 64:192] = 200
                engine.set_image(img)
                masks, scores, _ = engine.predict_mask(
                    np.array([[128, 128]], dtype=np.float32), np.array([1]))
                report["sam"] = {
                    "loaded": True, "model_type": engine.model_type,
                    "device": engine.device, "masks_shape": list(masks.shape),
                    "best_iou": float(scores.max()),
                }
            else:
                report["sam"] = {"loaded": False, "status": engine.get_status()}
                report["ok"] = False
        except Exception as e:  # noqa: BLE001
            report["sam"] = {"loaded": False, "error": f"{type(e).__name__}: {e}"}
            report["ok"] = False
    else:
        report["sam"] = {"loaded": None, "note": "no model path provided"}

    report["elapsed_s"] = round(time.time() - t0, 2)
    out = os.environ.get("DICOM_SELFTEST_OUT", os.path.join(PROJECT_DIR, "selftest.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return 0 if report["ok"] else 1


def main():
    if os.environ.get("DICOM_SELFTEST", "0") == "1":
        sys.exit(_run_selftest())
    app = CApplication()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
