import os
import json
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox,
    QGroupBox, QLabel,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from PyCt6 import CButton
from roi_engine import RoiEngine
from ocr_engine import OcrEngine
import dicom_summary_db

C_ACCENT = "#2563EB"
C_ACCENT_HOVER = "#1D4ED8"
C_GREEN = "#059669"
C_GREEN_HOVER = "#047857"
C_TITLE = "#0B1E35"
C_TEXT = "#1E293B"
C_SUBTEXT = "#475569"
C_CARD_BG = "#FFFFFF"
C_CARD_BORDER = "#1873FF"
C_SIDEBAR = "#F0F4F8"
C_SPLITTER = "#CBD5E1"


class RoiWorker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    done = Signal(str, str, str, object)

    def __init__(self, file_paths, method, params, output_dir):
        super().__init__()
        self.file_paths = file_paths
        self.method = method
        self.params = params
        self.output_dir = output_dir

    def run(self):
        engine = RoiEngine()
        ocr_engine = None
        all_rois = []
        total = len(self.file_paths)
        for idx, fp in enumerate(self.file_paths):
            self.progress.emit(idx + 1, total)
            self.log.emit(f"Processing: {os.path.basename(fp)}")
            try:
                rois = engine.detect(fp, method=self.method, params=self.params)
                all_rois.append({"file": fp, "rois": rois})
                if rois and not (len(rois) == 1 and "error" in rois[0]):
                    dicom_summary_db.add_merged_roi(fp, self.method, rois)
                # OCR: auto-run on overlay images
                mask = engine.extract_overlay_mask(fp)
                if mask is not None and mask.sum() > 100:
                    if ocr_engine is None:
                        ocr_engine = OcrEngine()
                    ocr_img = engine.render_overlay_for_ocr(fp)
                    if ocr_img is not None:
                        blocks = ocr_engine.extract(ocr_img)
                        if blocks:
                            dicom_summary_db.add_ocr_blocks(fp, blocks)
                            self.log.emit(f"OCR: {len(blocks)} blocks")
            except Exception as e:
                self.log.emit(f"Error: {fp} - {e}")
                all_rois.append({"file": fp, "rois": [], "error": str(e)})

        ts_dir = os.path.join(self.output_dir, f"roi_{self.method}")
        os.makedirs(ts_dir, exist_ok=True)
        json_path = os.path.join(ts_dir, "all_rois.json")
        result = {
            "method": self.method,
            "params": self.params,
            "file_count": total,
            "results": all_rois
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        total_rois = sum(len(r.get("rois", [])) for r in all_rois)
        summary = f"Done: {total} files, {total_rois} ROIs"
        self.done.emit(summary, json_path, "", None)


class RoiPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = RoiEngine()
        self.worker = None
        self._viewer = None
        self._output_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "output")
        self._setup_ui()

    def set_viewer(self, viewer):
        self._viewer = viewer

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        title = QLabel("ROI 检测配置")
        tf = QFont("Segoe UI", 12, QFont.Weight.Bold)
        title.setFont(tf)
        title.setStyleSheet(f"color: {C_TITLE};")
        layout.addWidget(title)

        gb = QGroupBox("检测参数")
        gb.setStyleSheet(f"""
            QGroupBox {{
                font-size: 12px;
                font-weight: bold;
                color: {C_SUBTEXT};
                border: 1px solid {C_SPLITTER};
                border-radius: 6px;
                margin-top: 8px;
                padding: 10px 8px 6px 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: {C_ACCENT};
            }}
        """)
        gb_layout = QVBoxLayout(gb)
        gb_layout.setSpacing(6)

        row1 = QHBoxLayout()
        lbl1 = QLabel("方法: overlay (DICOM Overlay Plane)")
        lbl1.setStyleSheet(f"color: {C_ACCENT}; font-size: 12px; font-weight: bold;")
        row1.addWidget(lbl1)
        row1.addStretch()
        gb_layout.addLayout(row1)
        gb_layout.addLayout(row1)

        row2 = QHBoxLayout()
        lbl2 = QLabel("最小区域:")
        lbl2.setFixedWidth(60)
        lbl2.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; font-weight: 600;")
        row2.addWidget(lbl2)
        self.min_area_label = QLabel("100")
        self.min_area_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 12px; font-weight: bold;")
        self.min_area_label.setFixedWidth(40)
        row2.addWidget(self.min_area_label)
        row2.addStretch()
        gb_layout.addLayout(row2)

        row3 = QHBoxLayout()
        lbl3 = QLabel("输出:")
        lbl3.setFixedWidth(60)
        lbl3.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; font-weight: 600;")
        row3.addWidget(lbl3)
        self.output_label = QLabel(self._output_dir)
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
        row3.addWidget(self.output_label)
        row3.addStretch()
        gb_layout.addLayout(row3)

        layout.addWidget(gb)

        self.run_btn = CButton(
            master=self, text="开始ROI检测",
            width=240, height=36,
            background_color=(C_ACCENT, C_ACCENT),
            hover_color=(C_ACCENT_HOVER, C_ACCENT_HOVER),
            text_color=("white", "white"),
            font_size=12,
            font_style="bold",
        )
        self.run_btn.button().clicked.connect(self._on_run)
        layout.addWidget(self.run_btn.button())

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {C_SPLITTER};
                border-radius: 4px;
                background: {C_SIDEBAR};
                height: 12px;
                text-align: center;
                font-size: 10px;
                color: {C_TITLE};
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background: {C_ACCENT};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px; font-weight: 500;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def set_files(self, files):
        self._files = files
        self.status_label.setText(f"已选 {len(files)} 个文件" if files else "就绪")

    def _on_run(self):
        self._run_overlay_display()
        self._run_batch_save()

    def _run_batch_save(self):
        file_paths = []
        owner = self.parent()
        while owner:
            if hasattr(owner, '_patients') and owner._patients:
                for p in owner._patients.values():
                    for s in p.studies.values():
                        for se in s.series.values():
                            for f in se.files:
                                file_paths.append(f.file_path)
                break
            owner = owner.parent()
        if not file_paths:
            QMessageBox.warning(self, "提示", "请先扫描DICOM文件夹")
            return
        params = {"min_area": 10}
        self.worker = RoiWorker(file_paths, "overlay", params, self._output_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _run_overlay_display(self):
        if self._viewer is None or self._viewer.current_path is None:
            QMessageBox.warning(self, "提示", "请先在病人列表中点击一个DICOM文件")
            return

        file_path = self._viewer.current_path
        all_overlays = self.engine.extract_overlay_mask(file_path)

        if all_overlays is None:
            QMessageBox.information(self, "提示", "当前DICOM文件不包含ROI叠加层")
            return

        pixel_count = int(all_overlays.sum())
        self._viewer.set_mask(all_overlays)
        self.status_label.setText(f"Overlay已显示 | {pixel_count} px")
        self.status_label.setStyleSheet(f"color: {C_GREEN}; font-weight: bold; font-size: 11px;")

    def _on_progress(self, val, total):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"{val}/{total}")

    def _on_log(self, msg):
        self.status_label.setText(msg)

    def _on_done(self, summary, json_path, png_path, _image):
        self.progress_bar.setVisible(False)
        self.run_btn.button().setEnabled(True)
        self.status_label.setText(summary)
        self.status_label.setStyleSheet(f"color: {C_GREEN}; font-weight: bold; font-size: 11px;")
        QMessageBox.information(self, "ROI检测完成",
                                 f"{summary}\n\n结果已保存至:\n{json_path}")
