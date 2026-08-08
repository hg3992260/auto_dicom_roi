#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动测试: 确保 SAM 模块可以加载 models/ 下三个精度的模型权重
  - sam_vit_b_01ec64.pth  (ViT-B STANDARD)
  - sam_vit_l_0b3195.pth  (ViT-L STANDARD+)
  - sam_vit_h_4b8939.pth  (ViT-H HI-PRECISION)

同时验证: 程序只确认模型权重是否可用, 不限制从什么路径读取模型
(任意路径下的 .pth 均可通过 set_checkpoint + load_model 加载并推理).

用法: conda activate dicom && python test_sam_loading.py
"""
import os
import sys
import gc
import time
import shutil
import tempfile
import traceback

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

from app_paths import SAM_CHECKPOINT_NAMES
from sam_engine import SamEngine, HAS_TORCH, HAS_SAM, HAS_PYDICOM

MODELS_DIR = os.path.join(PROJECT_DIR, "models")

PASS = []
FAIL = []


def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    (PASS if ok else FAIL).append((name, detail))
    return ok


def make_test_image(size=256):
    """合成测试图: 128x128 的高亮方块位于暗背景中央, 用于点提示推理."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)
    img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = (200, 200, 200)
    return img


def pick_device(engine):
    """优先使用 MPS (Apple Silicon), 否则 CPU. 不改变程序默认行为, 仅加速测试."""
    if HAS_TORCH:
        import torch
        if torch.backends.mps.is_available():
            engine.device = "mps"
        elif torch.cuda.is_available():
            engine.device = "cuda"


def verify_model_usable(engine, label, path):
    """加载权重并做一次真实推理, 确认权重可用而非仅文件存在."""
    t0 = time.time()
    engine.set_checkpoint(path)
    mtype = engine.model_type
    ok = engine.load_model()
    if not ok:
        return report(label, False, f"load_model() returned False | status={engine.get_status()}")
    if engine.sam is None or engine.predictor is None:
        return report(label, False, "sam/predictor 未创建")
    # 真实推理: 单点提示
    try:
        img = make_test_image()
        engine.set_image(img)
        pts = np.array([[img.shape[1] // 2, img.shape[0] // 2]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int64)
        masks, scores, logits = engine.predict_mask(pts, lbl)
        # multimask_output=True -> masks shape (num_masks, H, W); num_masks >= 1
        assert masks.ndim == 3 and masks.shape[0] >= 1, f"unexpected mask shape {masks.shape}"
        assert scores is not None and len(scores) >= 1, "no scores"
        seg = masks[0]
        area = int(seg.sum())
        dt = time.time() - t0
        return report(label, True,
                      f"model_type={mtype} | device={engine.device} | masks={masks.shape} | "
                      f"best_iou={float(scores.max()):.3f} | seg_area={area}px | {dt:.1f}s")
    except Exception as e:
        return report(label, False, f"inference failed: {type(e).__name__}: {e}")


def test_three_precisions():
    print("\n=== 1) 三个精度模型从 models/ 加载 ===")
    expected = {
        "sam_vit_b_01ec64.pth": "vit_b",
        "sam_vit_l_0b3195.pth": "vit_l",
        "sam_vit_h_4b8939.pth": "vit_h",
    }
    all_ok = True
    for name in SAM_CHECKPOINT_NAMES:
        path = os.path.join(MODELS_DIR, name)
        if not os.path.isfile(path):
            all_ok &= report(name, False, f"file missing: {path}")
            continue
        engine = SamEngine()
        pick_device(engine)
        ok = verify_model_usable(engine, name, path)
        all_ok &= ok
        if ok:
            # 校验 model_type 推断
            expect_type = expected[name]
            det = engine.model_type
            all_ok &= report(f"{name} -> model_type", det == expect_type,
                             f"detected={det} expected={expect_type}")
        del engine
        gc.collect()
    return all_ok


def test_path_independence():
    print("\n=== 2) 不限制读取路径: 从任意目录加载模型 ===")
    src = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
    tmpdir = tempfile.mkdtemp(prefix="sam_alt_path_")
    try:
        dst = os.path.join(tmpdir, "my_custom_weights.pth")
        shutil.copy2(src, dst)
        engine = SamEngine()
        pick_device(engine)
        ok = verify_model_usable(engine, f"加载自外部路径 {os.path.basename(dst)}", dst)
        return ok
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_gui_smoke():
    print("\n=== 3) main.py 启动冒烟测试 ===")
    import subprocess
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = (
        "import sys; from PySide6.QtCore import QTimer; "
        "import main as m; "
        "app = m.CApplication(); w = m.MainWindow(); w.show(); "
        "QTimer.singleShot(1500, app.quit); "
        "sys.exit(app.exec())"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            return report("main.py 启动并正常退出", True, "GUI 构造 + show + 事件循环 OK")
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-8:]
        return report("main.py 启动并正常退出", False, f"rc={r.returncode} | " + " | ".join(tail))
    except Exception as e:
        return report("main.py 启动并正常退出", False, f"{type(e).__name__}: {e}")


def main():
    print(f"python: {sys.executable}")
    print(f"torch={HAS_TORCH} sam={HAS_SAM} pydicom={HAS_PYDICOM}")
    t_start = time.time()

    r1 = test_three_precisions()
    r2 = test_path_independence()
    r3 = test_gui_smoke()

    print("\n" + "=" * 60)
    print(f"结果: PASS={len(PASS)}  FAIL={len(FAIL)}  总耗时 {time.time() - t_start:.0f}s")
    if FAIL:
        print("失败项:")
        for name, detail in FAIL:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print("全部通过 ✔")
    sys.exit(0)


if __name__ == "__main__":
    main()
