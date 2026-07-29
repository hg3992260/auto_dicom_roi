"""Interactive 4-panel SAM viewer — manual seed points → HTML report."""
import os, sys, base64, json, time
import numpy as np
import cv2
import pydicom
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QListWidget, QSplitter,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QMouseEvent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_engine import RoiEngine
from sam_engine import SamEngine

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "面神经已处理图像", "面神经已处理图像", "miao")
BASE_DIR = os.path.abspath(BASE_DIR) if os.path.exists(BASE_DIR) else ""

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SAM_RESULTS_DIR = os.path.join(OUTPUT_DIR, "sam_masks")
os.makedirs(SAM_RESULTS_DIR, exist_ok=True)

engine = RoiEngine()
sam = SamEngine()

FG_COLOR = QColor(0, 255, 0)
BG_COLOR = QColor(255, 80, 80)
MASK_COLOR = np.array([255, 0, 127], dtype=np.uint8)
C_DOT_RADIUS = 5


def norm_u8(pixels):
    pixels = pixels.astype(np.float32)
    if pixels.ndim == 3:
        pixels = pixels[:, :, 0]
    mn, mx = pixels.min(), pixels.max()
    return ((pixels - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(pixels, dtype=np.uint8)


def np_to_qpixmap(arr):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    else:
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


def img_to_b64(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode()


class InteractiveSamView(QLabel):
    """Clickable SAM view — left=foreground(green), right=background(red)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("border: 2px solid #7C3AED; background: #222;")
        self.setText("点击添加种子点\n左键=前景 右键=背景")
        self.setFont(QFont("Microsoft YaHei", 10))
        self.setStyleSheet("border:2px solid #7C3AED; background:#1a1a2e; color:#aaa;")
        self.points = []
        self.point_labels = []
        self.base_pixmap = None
        self.sam_mask = None

    def set_base_image(self, pixmap):
        self.base_pixmap = pixmap
        self.points.clear()
        self.point_labels.clear()
        self.sam_mask = None
        self._redraw()

    def set_sam_mask(self, mask):
        self.sam_mask = mask
        self._redraw()

    def mousePressEvent(self, ev: QMouseEvent):
        if self.base_pixmap is None:
            return
        label_size = self.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return
        pm_size = self.base_pixmap.size()
        scale = min(label_size.width() / pm_size.width(), label_size.height() / pm_size.height())
        offset_x = (label_size.width() - pm_size.width() * scale) / 2
        offset_y = (label_size.height() - pm_size.height() * scale) / 2
        img_x = (ev.position().x() - offset_x) / scale
        img_y = (ev.position().y() - offset_y) / scale
        if img_x < 0 or img_x >= pm_size.width() or img_y < 0 or img_y >= pm_size.height():
            return
        if ev.button() == Qt.LeftButton:
            self.points.append((int(img_x), int(img_y)))
            self.point_labels.append(1)
        elif ev.button() == Qt.RightButton:
            self.points.append((int(img_x), int(img_y)))
            self.point_labels.append(0)
        self._redraw()

    def _redraw(self):
        if self.base_pixmap is None:
            return
        pm = QPixmap(self.base_pixmap)
        label_size = self.size()
        if label_size.width() <= 0:
            self.setPixmap(pm)
            return
        pm = pm.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter = QPainter(pm)
        if self.sam_mask is not None:
            alpha_mask = cv2.resize(
                self.sam_mask.astype(np.uint8),
                (pm.width(), pm.height()),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            overlay = np.zeros((pm.height(), pm.width(), 4), dtype=np.uint8)
            overlay[alpha_mask] = [255, 0, 127, 96]
            overlay_img = QImage(
                overlay.data,
                pm.width(),
                pm.height(),
                pm.width() * 4,
                QImage.Format_RGBA8888,
            ).copy()
            painter.drawImage(0, 0, overlay_img)

        for (px, py), lbl in zip(self.points, self.point_labels):
            sx = int(px * pm.width() / self.base_pixmap.width())
            sy = int(py * pm.height() / self.base_pixmap.height())
            color = FG_COLOR if lbl == 1 else BG_COLOR
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(QPoint(sx, sy), C_DOT_RADIUS, C_DOT_RADIUS)
        painter.end()
        self.setPixmap(pm)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._redraw()


class SamAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Interactive Annotation Tool")
        self.setMinimumSize(1200, 800)

        self.files = []
        self.current_idx = -1
        self.results = {}

        self._build_ui()
        self._init_sam()
        self._load_files()

    def _init_sam(self):
        if not sam.available:
            QMessageBox.critical(self, "Error", "SAM不可用: 缺少torch/segment-anything")
            return
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        default_ckpt = os.path.join(models_dir, "sam_vit_b_01ec64.pth")
        if not os.path.exists(default_ckpt):
            default_ckpt = os.path.join(models_dir, "sam_vit_l_0b3195.pth")
        if not os.path.exists(default_ckpt):
            default_ckpt = os.path.join(models_dir, "sam_vit_h_4b8939.pth")
        if os.path.exists(default_ckpt):
            sam.checkpoint_path = default_ckpt
            sam.model_type = "vit_b" if "vit_b" in default_ckpt else ("vit_l" if "vit_l" in default_ckpt else "vit_h")
            ok = sam.load_model()
            if ok:
                print(f"SAM loaded: {sam.model_type} on {sam.device.upper()}")
                self.status_label.setText(f"SAM {sam.model_type.upper()} ready ({sam.device.upper()})")
            else:
                QMessageBox.warning(self, "Warning", f"SAM加载失败: {default_ckpt}")
        else:
            QMessageBox.warning(self, "Warning", "models目录下未找到SAM权重文件")

    def _load_files(self):
        if not os.path.exists(BASE_DIR):
            self._browse_folder()
            return
        for patient in sorted(os.listdir(BASE_DIR)):
            pp = os.path.join(BASE_DIR, patient, 'DICOM')
            if not os.path.exists(pp):
                continue
            for root, dirs, fnames in os.walk(pp):
                for fn in sorted(fnames):
                    fp = os.path.join(root, fn)
                    if os.path.getsize(fp) < 100 or fn.endswith(('.png', '.jpg', '.log', '.txt')):
                        continue
                    mask = engine.extract_overlay_mask(fp)
                    if mask is not None and mask.sum() > 0:
                        self.files.append((fp, fn, patient))
        self.file_list.addItems([f"{p}/{f}" for _, f, p in self.files])
        self.progress.setMaximum(len(self.files))
        self._update_status()

    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        hsplit = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout(main)
        main_layout.addWidget(hsplit)

        # --- Left panel: File list + controls ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setFixedWidth(300)

        left_layout.addWidget(QLabel("Overlay文件列表:"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentRowChanged.connect(self._on_file_select)
        left_layout.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        btn_prev = QPushButton("◀ 上一个")
        btn_prev.clicked.connect(lambda: self._navigate(-1))
        btn_next = QPushButton("下一个 ▶")
        btn_next.clicked.connect(lambda: self._navigate(1))
        btn_row.addWidget(btn_prev)
        btn_row.addWidget(btn_next)
        left_layout.addLayout(btn_row)

        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)

        self.status_label = QLabel("就绪")
        left_layout.addWidget(self.status_label)

        btn_load = QPushButton("浏览文件夹")
        btn_load.clicked.connect(self._browse_folder)
        left_layout.addWidget(btn_load)

        btn_report = QPushButton("📄 生成HTML报告")
        btn_report.setStyleSheet("background: #059669; color: #fff; padding: 8px; font-weight: bold;")
        btn_report.clicked.connect(self._generate_report)
        left_layout.addWidget(btn_report)

        hsplit.addWidget(left)

        # --- Right panel: 2x2 grid ---
        right = QWidget()
        grid = QGridLayout(right)
        grid.setSpacing(6)

        self.view_orig = self._make_view("原始图像", "#2563EB")
        self.view_ovl = self._make_view("Overlay Mask", "#059669")
        self.view_comp = self._make_view("叠加显示", "#D97706")
        self.sam_view = InteractiveSamView()

        grid.addWidget(self.view_orig, 0, 0)
        grid.addWidget(self.view_ovl, 0, 1)
        grid.addWidget(self.view_comp, 1, 0)
        grid.addWidget(self.sam_view, 1, 1)

        hsplit.addWidget(right)

        # --- Bottom buttons ---
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 4, 0, 0)

        btn_run = QPushButton("▶ Run SAM")
        btn_run.setStyleSheet("background: #7C3AED; color: #fff; padding: 8px 20px; font-weight: bold; font-size: 13px;")
        btn_run.clicked.connect(self._run_sam)

        btn_undo = QPushButton("↶ 撤销SAM")
        btn_undo.setStyleSheet("background: #D97706; color: #fff; padding: 8px 16px; font-size: 13px;")
        btn_undo.clicked.connect(self._undo_sam)

        btn_clear = QPushButton("✕ 重来")
        btn_clear.setStyleSheet("background: #64748B; color: #fff; padding: 8px 16px; font-size: 13px;")
        btn_clear.clicked.connect(self._clear_seeds)

        btn_save = QPushButton("✅ 确认并下一张")
        btn_save.setStyleSheet("background: #059669; color: #fff; padding: 8px 20px; font-weight: bold; font-size: 13px;")
        btn_save.clicked.connect(self._save_current)

        btn_skip = QPushButton("⏩ 跳过")
        btn_skip.setStyleSheet("background: #94A3B8; color: #fff; padding: 8px 16px; font-size: 13px;")
        btn_skip.clicked.connect(self._skip_current)

        bottom_layout.addWidget(btn_run)
        bottom_layout.addWidget(btn_undo)
        bottom_layout.addWidget(btn_clear)
        bottom_layout.addWidget(btn_save)
        bottom_layout.addWidget(btn_skip)
        bottom_layout.addStretch()

        main_layout.addWidget(bottom)

    def _make_view(self, title, border_color):
        gb = QGroupBox(title)
        gb.setStyleSheet(f"QGroupBox {{ border: 2px solid {border_color}; font-weight: bold; padding: 4px; }}")
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(2, 2, 2, 2)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(300, 280)
        label.setStyleSheet("background: #1a1a2e; color: #888;")
        label.setText("等待加载...")
        layout.addWidget(label)
        return gb

    def _on_file_select(self, idx):
        if idx < 0 or idx >= len(self.files):
            return
        self.current_idx = idx
        self._load_current_file()

    def _navigate(self, delta):
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.files):
            self.file_list.setCurrentRow(new_idx)

    def _load_current_file(self):
        fp, fn, patient = self.files[self.current_idx]
        try:
            ds = pydicom.dcmread(fp, force=True)
        except Exception:
            return

        mask = engine.extract_overlay_mask(fp)
        if mask is None or mask.sum() == 0:
            return

        try:
            px = ds.pixel_array
            if px.ndim == 3:
                px = px[:, :, 0]
            orig = norm_u8(px)
        except Exception:
            return

        h, w = mask.shape
        if orig.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (orig.shape[1], orig.shape[0])).astype(bool)

        self._current_orig = orig
        self._current_mask = mask
        self._current_file = fp

        ovl = np.zeros(orig.shape, dtype=np.uint8)
        ovl[mask] = 255

        comp = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        comp[mask] = [0, 255, 0]

        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        self._current_rgb = orig_rgb

        self.view_orig.findChild(QLabel).setPixmap(np_to_qpixmap(orig))
        self.view_ovl.findChild(QLabel).setPixmap(np_to_qpixmap(ovl))
        self.view_comp.findChild(QLabel).setPixmap(np_to_qpixmap(comp))
        self.sam_view.set_base_image(np_to_qpixmap(orig_rgb))
        self.sam_view.set_sam_mask(None)

        self._update_status()

    def _update_status(self):
        done = len(self.results)
        total = len(self.files)
        self.progress.setValue(done)
        self.status_label.setText(f"已标注: {done}/{total} | 当前: 第{self.current_idx+1}个")

    def _run_sam(self):
        if sam.predictor is None:
            QMessageBox.warning(self, "Warning", "SAM模型未加载")
            return
        if not self.sam_view.points:
            QMessageBox.warning(self, "Warning", "请先在SAM视图点击种子点(左键=前景)")
            return
        if not hasattr(self, '_current_rgb') or self._current_rgb is None:
            QMessageBox.warning(self, "Warning", "请先从左侧列表选择一个文件")
            return
        self.status_label.setText("SAM预测中...")
        QApplication.processEvents()
        try:
            sam.set_image(self._current_rgb)
            points = np.array(self.sam_view.points, dtype=np.float32)
            labels = np.array(self.sam_view.point_labels, dtype=np.int32)
            masks, scores, _ = sam.predict_mask(points, labels)
            best_idx = scores.argmax()
            self.sam_view.set_sam_mask(masks[best_idx])
            self.status_label.setText(
                f"SAM完成 | score:{scores[best_idx]:.3f} | {int(masks[best_idx].sum())}px")
        except Exception as e:
            QMessageBox.critical(self, "SAM错误", f"预测失败:\n{e}")

    def _clear_seeds(self):
        self.sam_view.points.clear()
        self.sam_view.point_labels.clear()
        self.sam_view.set_sam_mask(None)
        self.sam_view._redraw()
        self.status_label.setText("种子点已清除")

    def _undo_sam(self):
        self.sam_view.set_sam_mask(None)
        self.sam_view._redraw()
        self.status_label.setText("SAM结果已撤销 — 可调整种子点后重新Run")

    def _save_current(self):
        fp = self._current_file
        if self.sam_view.sam_mask is None:
            QMessageBox.warning(self, "Warning", "请先Run SAM得到结果")
            return
        mask = self.sam_view.sam_mask
        highlight = self._current_rgb.copy()
        highlight[mask] = MASK_COLOR
        highlight = cv2.addWeighted(self._current_rgb, 1.0, highlight, 0.5, 0)
        for (px, py), lbl in zip(self.sam_view.points, self.sam_view.point_labels):
            cv2.circle(highlight, (px, py), 4, (0, 255, 0) if lbl == 1 else (0, 0, 255), -1)

        self.results[fp] = {
            'mask': mask.copy(),
            'points': self.sam_view.points.copy(),
            'labels': self.sam_view.point_labels.copy(),
            'highlight_bgr': cv2.cvtColor(highlight, cv2.COLOR_RGB2BGR),
            'n_foreground': sum(1 for l in self.sam_view.point_labels if l == 1),
            'n_background': sum(1 for l in self.sam_view.point_labels if l == 0),
            'mask_px': int(mask.sum()),
        }
        basename = os.path.splitext(os.path.basename(fp))[0]
        np.save(os.path.join(SAM_RESULTS_DIR, f"{basename}_mask.npy"), mask)
        meta = {
            'file': fp,
            'points': self.sam_view.points.copy(),
            'labels': self.sam_view.point_labels.copy(),
            'mask_px': int(mask.sum()),
        }
        with open(os.path.join(SAM_RESULTS_DIR, f"{basename}_meta.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)
        self._update_status()
        self._navigate(1)

    def _skip_current(self):
        fp = self._current_file
        self.results[fp] = None
        self._update_status()
        self._navigate(1)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if not folder:
            return
        global BASE_DIR
        BASE_DIR = folder
        self.files.clear()
        self.file_list.clear()
        self.results.clear()
        self._load_files()

    def _generate_report(self):
        if not self.results:
            QMessageBox.information(self, "Info", "没有保存任何结果")
            return
        t0 = time.time()
        blocks = []
        stats_rows = []
        all_areas = []
        n_done = 0

        for fp, data in self.results.items():
            if data is None:
                blocks.append(f'<div class="file-block"><div class="fh"><b>{os.path.basename(fp)}</b><span>SAM: 跳过</span></div></div>')
                continue
            n_done += 1
            fn = os.path.basename(fp)
            patient = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            try:
                ds = pydicom.dcmread(fp, force=True)
                px = ds.pixel_array
                if px.ndim == 3:
                    px = px[:, :, 0]
                orig = norm_u8(px)
            except Exception:
                blocks.append(f'<div class="file-block"><b>{fn}</b> 读取失败</div>')
                continue

            ovl_mask = engine.extract_overlay_mask(fp)
            h, w = ovl_mask.shape
            if orig.shape[:2] != (h, w):
                ovl_mask = cv2.resize(ovl_mask.astype(np.uint8), (orig.shape[1], orig.shape[0])).astype(bool)

            ovl = np.zeros(orig.shape, dtype=np.uint8)
            ovl[ovl_mask] = 255
            comp = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            comp[ovl_mask] = [0, 255, 0]

            sam_highlight = data.get('highlight_bgr')
            b64_sam = img_to_b64(sam_highlight) if sam_highlight is not None else ''
            sam_html = f'<div class="sam"><span>SAM标注</span><img src="data:image/png;base64,{b64_sam}"/></div>' if b64_sam else ''

            blocks.append(f'''
            <div class="file-block">
                <div class="fh"><b>{fn}</b><span>{orig.shape[1]}×{orig.shape[0]}</span>
                <span>Overlay: {int(ovl_mask.sum())}px · SAM: {data['mask_px']}px | +{data['n_foreground']} -{data['n_background']}</span></div>
                <div class="img-row">
                    <div><span>原始</span><img src="data:image/png;base64,{img_to_b64(orig)}"/></div>
                    <div><span>Overlay</span><img src="data:image/png;base64,{img_to_b64(ovl)}"/></div>
                    <div><span>叠加</span><img src="data:image/png;base64,{img_to_b64(comp)}"/></div>
                    {sam_html}
                </div>
            </div>''')

            sam_mask = data['mask']
            mask_px = data['mask_px']
            if mask_px <= 0:
                continue

            px_vals = px[sam_mask]
            spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            area_mm2 = float(mask_px * spacing[0] * spacing[1])
            all_areas.append(area_mm2)
            sam_mean = float(np.mean(px_vals))
            sam_std = float(np.std(px_vals))

            peri = float(cv2.arcLength(cv2.findContours(
                sam_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True))
            circ = 4 * np.pi * mask_px / (peri * peri) if peri > 0 else 0

            stats_rows.append(
                f'<tr><td>{fn}</td><td>{patient}</td>'
                f'<td>{mask_px}</td><td>{area_mm2:.2f}</td>'
                f'<td>{sam_mean:.1f}</td><td>{sam_std:.1f}</td>'
                f'<td>{float(np.min(px_vals)):.1f}/{float(np.max(px_vals)):.1f}</td>'
                f'<td>{float(np.median(px_vals)):.1f}</td><td>{circ:.3f}</td></tr>')

        stats_table = ''
        if stats_rows:
            total_area = sum(all_areas)
            avg_area = total_area / len(all_areas) if all_areas else 0
            stats_table = f'''
            <h2>SAM分割统计参数 ({n_done}个标注文件)</h2>
            <table class="stats-tbl"><thead><tr>
            <th>文件</th><th>患者</th><th>像素数</th><th>面积(mm²)</th>
            <th>均值HU</th><th>标准差</th><th>Min/Max HU</th><th>中位数</th><th>圆形度</th>
            </tr></thead><tbody>{''.join(stats_rows)}</tbody></table>
            <div class="stat-summary">
            总标注: {n_done} | 跳过: {len(self.results) - n_done} |
            总像素: {sum(d['mask_px'] for d in self.results.values() if d)} |
            总面积: {total_area:.2f} mm² | 平均面积: {avg_area:.2f} mm²
            </div>'''

        html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>SAM手工标注报告</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",sans-serif;background:#f5f7fa;color:#1e293b;padding:16px}}
h1{{text-align:center;color:#0b1e35;margin-bottom:4px}}
h2{{background:#7C3AED;color:#fff;padding:6px 14px;font-size:15px;border-radius:6px 6px 0 0}}
.summary{{text-align:center;color:#64748b;font-size:13px;margin-bottom:20px}}
.stat-summary{{text-align:center;color:#059669;font-weight:bold;font-size:13px;padding:8px;margin:10px 0}}
.file-block{{background:#fff;border:1px solid #e2e8f0;padding:10px;margin-bottom:16px;border-radius:6px}}
.fh{{display:flex;gap:12px;align-items:center;font-size:12px;margin-bottom:8px;flex-wrap:wrap}}
.fh b{{color:#2563eb}}
.fh span{{color:#64748b;font-size:11px}}
.img-row{{display:flex;gap:10px;flex-wrap:wrap}}
.img-row>div{{flex:1;min-width:200px;text-align:center}}
.img-row>div.sam{{border:2px solid #7C3AED;border-radius:6px;padding:2px;background:#F5F3FF}}
.img-row span{{display:block;font-size:10px;font-weight:600;color:#64748b;margin-bottom:3px}}
.img-row>div.sam span{{color:#7C3AED;font-size:11px}}
.img-row img{{width:100%;border:1px solid #e2e8f0;border-radius:4px}}
.stats-tbl{{width:100%;border-collapse:collapse;font-size:11px;margin-top:20px;background:#fff}}
.stats-tbl th{{background:#7C3AED;color:#fff;padding:4px 6px;text-align:left;font-size:10px}}
.stats-tbl td{{padding:3px 6px;border-bottom:1px solid #e2e8f0}}
.stats-tbl tr:nth-child(even){{background:#F5F3FF}}
</style></head><body>
<h1>SAM手工标注报告</h1>
<div class="summary">总文件: {len(self.results)} · 已标注: {n_done} · {time.strftime('%Y-%m-%d %H:%M:%S')} · {time.time()-t0:.1f}s</div>
{stats_table}
{''.join(blocks)}
</body></html>'''

        out_path = os.path.join(OUTPUT_DIR, "sam_manual_report.html")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        QMessageBox.information(self, "完成",
                                f"报告+统计已生成: {out_path}\n大小: {os.path.getsize(out_path)/1024**2:.0f} MB")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SamAnnotationTool()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
