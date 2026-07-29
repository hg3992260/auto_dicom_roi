import os
import json
import time
import numpy as np
import cv2
import pydicom

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox,
    QLabel, QInputDialog, QLineEdit, QFileDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from PyCt6 import CButton

from sam_engine import SamEngine
import dicom_summary_db

C_ACCENT = "#2563EB"
C_ACCENT_HOVER = "#1D4ED8"
C_GREEN = "#059669"
C_GREEN_HOVER = "#047857"
C_RED = "#DC2626"
C_TITLE = "#0B1E35"
C_TEXT = "#1E293B"
C_SUBTEXT = "#475569"
C_CARD_BG = "#FFFFFF"
C_SPLITTER = "#CBD5E1"
C_SIDEBAR = "#F0F4F8"
C_WARN = "#D97706"


class SamPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = SamEngine()
        self._viewer = None
        self._current_mask = None
        self._current_file = None
        self._file_list = []
        self._current_idx = 0
        self._output_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "output")
        self._setup_ui()

    def set_viewer(self, viewer):
        self._viewer = viewer
        viewer.manual_roi_completed.connect(self._on_manual_roi_completed)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Model path input row
        model_row = QHBoxLayout()
        model_row.setSpacing(3)
        self.model_path_btn = CButton(
            master=self,
            text="选择SAM模型文件...",
            width=160, height=24,
            font_size=9,
        )
        self.model_path_btn.button().clicked.connect(self._browse_model)
        model_row.addWidget(self.model_path_btn.button())

        self.confirm_load_btn = CButton(
            master=self, text="确认加载", width=64, height=24,
            background_color=(C_GREEN, C_GREEN),
            hover_color=(C_GREEN_HOVER, C_GREEN_HOVER),
            text_color=("white", "white"), font_size=9, font_style="bold")
        self.confirm_load_btn.button().clicked.connect(self._on_confirm_load)
        model_row.addWidget(self.confirm_load_btn.button())
        layout.addLayout(model_row)

        # Model status - high contrast
        self.model_status = QLabel(self.engine.get_status())
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 500; padding: 2px 4px;")
        layout.addWidget(self.model_status)

        # Navigation row
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self.nav_prev_btn = CButton(
            master=self, text="◀ 前一张", width=80, height=24, font_size=10)
        self.nav_prev_btn.button().clicked.connect(self._on_prev_image)
        self.nav_prev_btn.button().setEnabled(False)
        nav_row.addWidget(self.nav_prev_btn.button())
        self.nav_next_btn = CButton(
            master=self, text="后一张 ▶", width=80, height=24, font_size=10)
        self.nav_next_btn.button().clicked.connect(self._on_next_image)
        self.nav_next_btn.button().setEnabled(False)
        nav_row.addWidget(self.nav_next_btn.button())
        self.nav_label = QLabel("")
        self.nav_label.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 9px;")
        nav_row.addWidget(self.nav_label)
        nav_row.addStretch()
        layout.addLayout(nav_row)

        # Manual ROI controls
        manual_row = QHBoxLayout()
        manual_row.setSpacing(4)
        self.manual_btn = CButton(
            master=self, text="手动ROI", width=70, height=24,
            font_size=10,
            background_color=(C_WARN, C_WARN),
            hover_color=("#B45309", "#B45309"),
            text_color=("white", "white"))
        self.manual_btn.button().clicked.connect(self._on_manual_roi)
        manual_row.addWidget(self.manual_btn.button())
        from PySide6.QtWidgets import QComboBox
        self.shape_cb = QComboBox()
        self._shape_map = {"矩形": "rect", "圆形": "circle", "直线": "line", "角度": "angle", "多边形": "poly", "多点测量": "measure"}
        self.shape_cb.addItems(list(self._shape_map.keys()))
        self.shape_cb.setCurrentText("矩形")
        self.shape_cb.setStyleSheet(f"font-size: 10px; padding: 1px 4px;")
        self.shape_cb.currentTextChanged.connect(self._on_shape_changed)
        self.shape_cb.setVisible(False)
        manual_row.addWidget(self.shape_cb)
        self.manual_done_btn = CButton(
            master=self, text="完成", width=50, height=24, font_size=10)
        self.manual_done_btn.button().clicked.connect(self._on_manual_done)
        self.manual_done_btn.button().setVisible(False)
        manual_row.addWidget(self.manual_done_btn.button())
        self.manual_clear_btn = CButton(
            master=self, text="清除", width=50, height=24, font_size=10)
        self.manual_clear_btn.button().clicked.connect(self._on_manual_clear)
        self.manual_clear_btn.button().setVisible(False)
        manual_row.addWidget(self.manual_clear_btn.button())
        manual_row.addStretch()
        layout.addLayout(manual_row)

        # Action buttons
        act_row = QHBoxLayout()
        act_row.setSpacing(4)
        self.int_clear_btn = CButton(
            master=self, text="清除标记", width=80, height=24, font_size=10)
        self.int_clear_btn.button().clicked.connect(self._on_clear_points)
        act_row.addWidget(self.int_clear_btn.button())
        self.int_save_btn = CButton(
            master=self, text="保存ROI", width=80, height=24,
            background_color=(C_GREEN, C_GREEN),
            hover_color=(C_GREEN_HOVER, C_GREEN_HOVER),
            text_color=("white", "white"), font_size=10,
        )
        self.int_save_btn.button().clicked.connect(self._on_save_mask)
        self.int_save_btn.button().setEnabled(False)
        act_row.addWidget(self.int_save_btn.button())
        act_row.addStretch()
        layout.addLayout(act_row)

        # Status
        self.status_label = QLabel("就绪 — 切换至SAM分割标签后，在右侧图像上左键=前景 右键=背景")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 500;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM模型权重文件",
            os.path.expanduser("~"),
            "PyTorch模型 (*.pth);;所有文件 (*.*)")
        if path:
            self._selected_model_path = path
            self.model_path_btn.button().setText(os.path.basename(path))
            self.model_path_btn.button().setToolTip(path)

    def _on_confirm_load(self):
        path = getattr(self, '_selected_model_path', '')
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "提示", "请先选择有效的SAM模型文件")
            return
        self.confirm_load_btn.button().setEnabled(False)
        self.model_status.setText("正在加载...")
        self.model_status.setStyleSheet(f"color: {C_WARN}; font-weight: bold; font-size: 11px;")
        self.engine.set_checkpoint(path)
        def _update_status(msg):
            self.model_status.setText(msg)
            self.model_status.setStyleSheet(f"color: {C_WARN}; font-weight: bold; font-size: 11px;")
        ok = self.engine.load_model(status_callback=_update_status)
        fname = os.path.basename(path)
        mtype = self.engine.model_type.upper()
        if mtype == "VIT_H":
            label = "HI-PRECISION"
            accent = "#7C3AED"
            bg = "#F5F3FF"
        elif mtype == "VIT_L":
            label = "STANDARD+"
            accent = "#D97706"
            bg = "#FFFBEB"
        else:
            label = "STANDARD"
            accent = C_GREEN
            bg = "#ECFDF5"
        if ok:
            self.model_status.setText(f"✓ 加载完成 | {mtype} ({label}) | {self.engine.device.upper()}")
            self.model_status.setStyleSheet(
                f"color: {accent}; font-weight: bold; font-size: 12px; "
                f"background-color: {bg}; padding: 4px 8px; border-radius: 4px; "
                f"border: 2px solid {accent};")
            if self._viewer is not None and self._viewer.image_rgb is not None:
                self.engine.set_image(self._viewer.image_rgb)
                v = self._viewer
                if v._points:
                    try:
                        points_np = np.array(v._points)
                        labels_np = np.array(v._point_labels)
                        masks, scores, _ = self.engine.predict_mask(points_np, labels_np)
                        best_idx = scores.argmax()
                        self._current_mask = masks[best_idx]
                        v.set_mask(masks[best_idx])
                        self.int_save_btn.button().setEnabled(True)
                        self.status_label.setText(
                            f"模型已切换 | {mtype} | Mask自动更新 (Score: {scores[best_idx]:.3f})")
                    except Exception as e:
                        self.status_label.setText(f"模型已切换 | {mtype} | 标记已保留，请点击重新预测")
                else:
                    self.status_label.setText(f"模型已切换 | {mtype} | 就绪")
        else:
            self.model_status.setText(f"✗ 加载失败 | {fname}")
            self.model_status.setStyleSheet(
                f"color: {C_RED}; font-weight: bold; font-size: 12px; "
                f"background-color: #FEF2F2; padding: 4px 8px; border-radius: 4px; "
                f"border: 2px solid {C_RED};")
        self.confirm_load_btn.button().setEnabled(True)

    def _extract_dicom_metadata(self, file_path):
        try:
            ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            info = {}
            for tag, key in [
                ("PatientName", "患者姓名"), ("PatientID", "患者ID"),
                ("StudyDate", "检查日期"), ("Modality", "成像模式"),
                ("BodyPartExamined", "检查部位"), ("StudyDescription", "检查描述"),
                ("SeriesDescription", "系列描述"), ("Manufacturer", "制造商"),
                ("ManufacturerModelName", "设备型号"),
                ("SliceThickness", "层厚"), ("KVP", "kVp"),
                ("Rows", "行数"), ("Columns", "列数"),
            ]:
                val = getattr(ds, tag, None)
                if val not in (None, "", "None"):
                    info[key] = str(val)
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None:
                try:
                    info["像素间距"] = f"{float(ps[0]):.3f} × {float(ps[1]):.3f} mm"
                except Exception:
                    pass
            return info
        except Exception:
            return {}

    def _compute_roi_stats(self, file_path, mask):
        try:
            ds = pydicom.dcmread(file_path, force=True)
            pixels = ds.pixel_array.astype(np.float32)
            if pixels.ndim == 3:
                pixels = pixels[:, :, 0]
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            hu = pixels * slope + intercept
            if mask.shape[:2] != hu.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (hu.shape[1], hu.shape[0])).astype(bool)
            roi_pixels = hu[mask]
            if roi_pixels.size == 0:
                return {"error": "选区为空"}
            spacing = getattr(ds, 'PixelSpacing', None)
            area_mm2 = 0.0
            if spacing is not None and len(spacing) >= 2:
                area_mm2 = float(roi_pixels.size * float(spacing[0]) * float(spacing[1]))
            return {
                "像素数": int(roi_pixels.size),
                "面积(mm²)": f"{area_mm2:.2f}",
                "均值": f"{float(np.mean(roi_pixels)):.2f} HU",
                "标准差": f"{float(np.std(roi_pixels)):.2f} HU",
                "最小值": f"{float(np.min(roi_pixels)):.2f} HU",
                "最大值": f"{float(np.max(roi_pixels)):.2f} HU",
                "中位数": f"{float(np.median(roi_pixels)):.2f} HU",
            }
        except Exception as e:
            return {"error": str(e)}

    def set_files(self, files):
        self._file_list = files[:]
        self.status_label.setText(f"已选 {len(files)} 个文件" if files else "就绪")

    def set_file_context(self, file_path, file_list):
        self._file_list = list(file_list) if file_list else []
        try:
            self._current_idx = self._file_list.index(file_path)
        except ValueError:
            self._current_idx = 0
            if file_path:
                self._file_list = [file_path]
        self._current_file = file_path
        self._current_mask = None
        self._update_nav()
        if self.engine.predictor is not None and self._viewer is not None:
            rgb = self._viewer.image_rgb
            if rgb is not None:
                self.engine.set_image(rgb)

    def _update_nav(self):
        n = len(self._file_list)
        self.nav_prev_btn.button().setEnabled(n > 0 and self._current_idx > 0)
        self.nav_next_btn.button().setEnabled(n > 0 and self._current_idx < n - 1)
        self.nav_label.setText(f"{self._current_idx + 1}/{n}" if n > 0 else "")

    def _on_prev_image(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._load_image_to_viewer(self._file_list[self._current_idx])
            self._update_nav()

    def _on_next_image(self):
        if self._current_idx < len(self._file_list) - 1:
            self._current_idx += 1
            self._load_image_to_viewer(self._file_list[self._current_idx])
            self._update_nav()

    def _load_image_to_viewer(self, file_path):
        if self._viewer is not None:
            self._viewer.load_dicom(file_path)
            self._current_file = file_path
            self._current_mask = None
            if self.engine.predictor is not None and self._viewer.image_rgb is not None:
                self.engine.set_image(self._viewer.image_rgb)
            self.int_save_btn.button().setEnabled(False)
            self.status_label.setText(f"已加载: {os.path.basename(file_path)}")

    def on_viewer_point(self, x, y, label):
        if not self.engine.predictor:
            self.status_label.setText("请先选择并加载SAM模型")
            return
        try:
            v = self._viewer
            if v is None:
                return
            points_np = np.array(v._points)
            labels_np = np.array(v._point_labels)
            masks, scores, _ = self.engine.predict_mask(points_np, labels_np)
            best_idx = scores.argmax()
            self._current_mask = masks[best_idx]
            v.set_mask(masks[best_idx])
            self.int_save_btn.button().setEnabled(True)
            self.status_label.setText(f"Mask已更新 (Score: {scores[best_idx]:.3f})")
        except Exception as e:
            self.status_label.setText(f"预测失败: {e}")

    def _on_clear_points(self):
        self._current_mask = None
        self.int_save_btn.button().setEnabled(False)
        if self._viewer is not None:
            self._viewer.clear_mask()
        self.status_label.setText("标记已清除")

    # ---- Manual ROI ----
    def _on_manual_roi(self):
        if self._viewer is None or self._viewer._raw_pixels is None:
            return
        shape_label = self.shape_cb.currentText()
        shape = self._shape_map.get(shape_label, "rect")
        showing = self.shape_cb.isVisible()
        self.shape_cb.setVisible(not showing)
        self.manual_done_btn.button().setVisible(not showing)
        self.manual_clear_btn.button().setVisible(not showing)
        if showing:
            self._viewer.set_manual_draw("", False)
            self._viewer.set_sam_mode(True)
            self.manual_btn.button().setText("手动ROI")
            self.manual_btn.button().setStyleSheet(
                f"background-color: {C_WARN}; color: white; font-size: 10px;")
            self.status_label.setText("手动ROI已关闭")
        else:
            self._viewer.set_sam_mode(False)
            self._viewer.set_manual_draw(shape, True)
            self.manual_btn.button().setText("关闭手动")
            self.manual_btn.button().setStyleSheet(
                f"background-color: {C_RED}; color: white; font-size: 10px;")
            hints = {"rect": "左键点角→拖动→右键完成", "circle": "左键圆心→拖动半径→右键完成",
                     "line": "左键起点→拖动→右键完成", "angle": "3次左键: 边点1→顶点→边点2",
                     "poly": "左键加点→右键闭合", "measure": "左键加点→右键结束测量"}
            self.status_label.setText(f"手动ROI [{shape_label}]: {hints.get(shape, '')}")
            self._manual_roi_data = None

    def _on_manual_roi_completed(self, data):
        self._manual_roi_data = data
        shape = self._shape_map.get(self.shape_cb.currentText(), "")
        extra = ""
        if data.get("angle_deg"):
            extra = f" ∠{data['angle_deg']}°"
        elif data.get("length_mm"):
            extra = f" {data['length_px']}px ({data['length_mm']}mm)"
        self.status_label.setText(f"手动ROI完成 [{shape}]: {len(data['points'])} pts{extra} — 可保存")
        self.int_save_btn.button().setEnabled(True)

    def _on_manual_done(self):
        if self._viewer is None:
            return
        self._viewer.set_manual_draw("", False)
        self._on_manual_roi()

    def _on_manual_clear(self):
        self._manual_roi_data = None
        self.int_save_btn.button().setEnabled(False)
        if self._viewer is not None:
            shape = self._shape_map.get(self.shape_cb.currentText(), "rect")
            self._viewer.set_manual_draw(shape, True)
        self.status_label.setText(f"手动ROI: 已清除")

    def _on_shape_changed(self, text):
        self._manual_roi_data = None
        self.int_save_btn.button().setEnabled(False)
        if self._viewer is not None and self.shape_cb.isVisible():
            shape = self._shape_map.get(text, "rect")
            self._viewer.set_manual_draw(shape, True)
            hints = {"rect": "左键点角→拖动→右键完成", "circle": "左键圆心→拖动半径→右键完成",
                     "line": "左键起点→拖动→右键完成", "angle": "3次左键: 边点1→顶点→边点2",
                     "poly": "左键加点→右键闭合", "measure": "左键加点→右键结束测量"}
            self.status_label.setText(f"手动ROI [{text}]: {hints.get(shape, '')}")

    def _on_save_mask(self):
        is_manual = hasattr(self, '_manual_roi_data') and self._manual_roi_data is not None
        if not is_manual and (self._current_mask is None or self._current_file is None):
            return
        if is_manual and (self._viewer is None or self._current_file is None):
            return
        if is_manual:
            mask_data = self._viewer.get_draw_mask()
            if mask_data is None or mask_data.sum() == 0:
                QMessageBox.warning(self, "提示", "手动ROI区域为空")
                return
            mask_to_save = mask_data > 0
            method_label = f"manual_{self._manual_roi_data['shape']}"
        else:
            mask_to_save = self._current_mask
            method_label = "sam_interactive"

        meta = self._extract_dicom_metadata(self._current_file)
        stats = self._compute_roi_stats(self._current_file, mask_to_save)

        meta_lines = ["═════ DICOM 元数据 ═════"]
        for k, v in meta.items():
            meta_lines.append(f"  {k}: {v}")
        if not meta:
            meta_lines.append("  (无元数据)")
        stats_lines = ["───── ROI 统计 ─────"]
        if is_manual and self._manual_roi_data.get("shape") in ("line", "measure"):
            d = self._manual_roi_data
            stats_lines.append(f"  长度(px): {d.get('length_px', '-')}")
            stats_lines.append(f"  长度(mm): {d.get('length_mm', '-')}")
        elif is_manual and self._manual_roi_data.get("shape") == "angle":
            stats_lines.append(f"  角度: {self._manual_roi_data.get('angle_deg', '-')}°")
        elif "error" in stats:
            stats_lines.append(f"  {stats['error']}")
        else:
            for k in ["像素数", "面积(mm²)", "均值", "标准差", "最小值", "最大值", "中位数"]:
                if k in stats:
                    stats_lines.append(f"  {k}: {stats[k]}")
        fname = os.path.basename(self._current_file)
        msg = f"确认保存当前ROI？\n文件: {fname}\n\n" + "\n".join(meta_lines[:10]) + "\n" + "\n".join(stats_lines)
        desc, ok = QInputDialog.getText(
            self, "ROI 描述", "请输入ROI描述 (必填):",
            QLineEdit.EchoMode.Normal, "")
        if not ok or not desc.strip():
            QMessageBox.warning(self, "描述为空", "必须输入ROI描述才能保存！")
            self.status_label.setText("保存已取消: 未输入描述")
            return
        roi_desc = desc.strip()
        reply = QMessageBox.question(
            self, "确认保存 ROI", f"描述: {roi_desc}\n\n{msg}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if reply != QMessageBox.StandardButton.Yes:
            self.status_label.setText("保存已取消")
            return

        os.makedirs(self._output_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        json_path = os.path.join(self._output_dir, f"{base}_sam_roi.json")
        png_path = os.path.join(self._output_dir, f"{base}_sam_roi.png")

        if self._viewer is not None and self._viewer.image_rgb is not None:
            overlay = self._viewer.image_rgb.copy()
            m = mask_to_save
            if m is not None and m.shape[:2] != overlay.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (overlay.shape[1], overlay.shape[0])).astype(bool)
            color_mask = np.zeros_like(overlay)
            color_mask[m] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)
            for px, py in self._viewer._points:
                cv2.circle(overlay, (px, py), 5, (0, 255, 0), -1)
            cv2.imwrite(png_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        result = {
            "file": self._current_file,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "source": "magic_seg_interactive" if not is_manual else f"manual_{self._manual_roi_data['shape']}",
            "metadata": meta,
            "roi_statistics": stats,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "保存成功", f"ROI已保存至:\n{json_path}\n{png_path}")
        self.status_label.setText(f"已保存: {base}")

        if is_manual and self._manual_roi_data.get("shape") in ("line", "measure", "angle"):
            shape = self._manual_roi_data["shape"]
            d = self._manual_roi_data
            if shape in ("line", "measure"):
                stats["长度(px)"] = d.get("length_px", "-")
                stats["长度(mm)"] = d.get("length_mm", "-")
            if shape == "angle":
                stats["角度(°)"] = d.get("angle_deg", "-")
            full_meta = dicom_summary_db.extract_dicom_meta(self._current_file)
            dicom_summary_db.insert_roi(
                patient_name=str(meta.get("患者姓名", meta.get("PatientName", "Unknown"))),
                patient_id=str(meta.get("患者ID", meta.get("PatientID", ""))),
                file_path=self._current_file,
                method=f"manual_{shape}",
                source=f"manual_{shape}",
                metadata=meta,
                stats=stats,
                area_mm2=d.get("length_mm") if shape in ("line", "measure") else d.get("angle_deg"),
                pixel_count=len(self._manual_roi_data.get("points", [])),
                roi_description=roi_desc,
                saved_png=png_path,
                saved_json=json_path,
                study_date=str(meta.get("检查日期", "")),
                study_time=full_meta.get("study_time", ""),
                study_description=str(meta.get("检查描述", "")),
                series_description=str(meta.get("系列描述", "")),
                manufacturer=full_meta.get("manufacturer", ""),
                manufacturer_model=full_meta.get("manufacturer_model", ""),
                body_part=full_meta.get("body_part", ""),
                protocol_name=full_meta.get("protocol_name", ""),
                modality=full_meta.get("modality", ""),
                length_mm=d.get("length_mm") if shape in ("line", "measure") else None,
                angle_deg=d.get("angle_deg") if shape == "angle" else None,
            )
        else:
            dicom_summary_db.add_roi_from_sam(
                file_path=self._current_file,
                metadata=meta,
                stats=stats,
                contour=mask_to_save.tolist() if mask_to_save is not None else None,
                saved_png=png_path,
                saved_json=json_path,
                roi_description=roi_desc,
            )
        self._manual_roi_data = None
        self._current_mask = None
