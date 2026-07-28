import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QSplitter, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

import dicom_summary_db as db

C_TOOLBAR = "#1E3A5F"
C_TOOLBAR_TEXT = "#FFFFFF"
C_SIDEBAR = "#F0F4F8"
C_CARD_BG = "#FFFFFF"
C_TITLE = "#0B1E35"
C_TEXT = "#1E293B"
C_SUBTEXT = "#475569"
C_ACCENT = "#2563EB"
C_ACCENT_HOVER = "#1D4ED8"
C_GREEN = "#059669"
C_GREEN_HOVER = "#047857"
C_RED = "#DC2626"
C_SPLITTER = "#CBD5E1"


class SummaryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)

        # Top: patient tree
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(2)
        tree_header = QLabel("患者列表")
        tree_header.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; font-weight: bold;")
        tree_layout.addWidget(tree_header)
        self.patient_tree = QTreeWidget()
        self.patient_tree.setHeaderLabels(["患者", "ROI数"])
        self.patient_tree.setColumnCount(2)
        self.patient_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.patient_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.patient_tree.setStyleSheet(f"""
            QTreeWidget {{ font-size: 11px; color: {C_TEXT}; background-color: {C_CARD_BG};
                border: 1px solid {C_SPLITTER}; border-radius: 4px;
                alternate-background-color: {C_SIDEBAR}; }}
            QTreeWidget::item:selected {{ background-color: {C_ACCENT}; color: white; }}
            QHeaderView::section {{ background-color: {C_TOOLBAR}; color: {C_TOOLBAR_TEXT};
                padding: 2px 4px; font-size: 10px; font-weight: bold; }}
        """)
        self.patient_tree.itemClicked.connect(self._on_patient_clicked)
        self.patient_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree_layout.addWidget(self.patient_tree)
        splitter.addWidget(tree_container)

        # Bottom: ROI table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(2)
        table_header_row = QHBoxLayout()
        table_header_row.setSpacing(4)
        self.table_title = QLabel("ROI 详细数据")
        self.table_title.setStyleSheet(f"color: {C_TITLE}; font-size: 11px; font-weight: bold;")
        table_header_row.addWidget(self.table_title)
        table_header_row.addStretch()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setFixedSize(40, 22)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{ font-size: 10px; color: {C_TEXT}; background: {C_CARD_BG};
                border: 1px solid {C_SPLITTER}; border-radius: 3px; padding: 1px 4px; }}
            QPushButton:hover {{ background: {C_SIDEBAR}; border-color: {C_ACCENT}; }}
        """)
        self.refresh_btn.clicked.connect(self._refresh)
        table_header_row.addWidget(self.refresh_btn)
        self.del_btn = QPushButton("删除选中")
        self.del_btn.setFixedSize(50, 22)
        self.del_btn.setStyleSheet(f"""
            QPushButton {{ font-size: 10px; color: white; background: {C_RED};
                border: 1px solid {C_RED}; border-radius: 3px; padding: 1px 4px; }}
            QPushButton:hover {{ background: #B91C1C; }}
        """)
        self.del_btn.clicked.connect(self._delete_selected)
        table_header_row.addWidget(self.del_btn)
        self.export_btn = QPushButton("导出Excel")
        self.export_btn.setFixedSize(60, 22)
        self.export_btn.setStyleSheet(f"""
            QPushButton {{ font-size: 10px; color: white; background: {C_GREEN};
                border: 1px solid {C_GREEN}; border-radius: 3px; padding: 1px 4px; }}
            QPushButton:hover {{ background: {C_GREEN_HOVER}; }}
        """)
        self.export_btn.clicked.connect(self._export_excel)
        table_header_row.addWidget(self.export_btn)
        table_layout.addLayout(table_header_row)

        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(14)
        self.roi_table.setHorizontalHeaderLabels([
            "编号", "方法", "文件", "设备", "部位", "协议",
            "面积(mm²)", "均值(HU)", "标准差", "Min/Max(HU)", "扫描日期", "时间", "OCR文本", "描述"
        ])
        for ci in [0, 1, 3, 4, 7, 8, 11]:
            self.roi_table.horizontalHeader().setSectionResizeMode(ci, QHeaderView.ResizeMode.ResizeToContents)
        self.roi_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.roi_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.roi_table.horizontalHeader().setSectionResizeMode(12, QHeaderView.ResizeMode.Stretch)
        self.roi_table.horizontalHeader().setSectionResizeMode(13, QHeaderView.ResizeMode.Stretch)
        for ci in [6, 9, 10]:
            self.roi_table.horizontalHeader().setSectionResizeMode(ci, QHeaderView.ResizeMode.ResizeToContents)
        self.roi_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.roi_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.roi_table.setAlternatingRowColors(True)
        self.roi_table.setStyleSheet(f"""
            QTableWidget {{ font-size: 11px; color: {C_TEXT}; background-color: {C_CARD_BG};
                border: 1px solid {C_SPLITTER}; border-radius: 4px;
                alternate-background-color: {C_SIDEBAR};
                gridline-color: {C_SPLITTER}; }}
            QTableWidget::item:selected {{ background-color: {C_ACCENT}; color: white; }}
            QHeaderView::section {{ background-color: {C_TOOLBAR}; color: {C_TOOLBAR_TEXT};
                padding: 3px 4px; font-size: 10px; font-weight: bold; }}
        """)
        self.roi_table.doubleClicked.connect(self._on_cell_double_click)
        table_layout.addWidget(self.roi_table)
        splitter.addWidget(table_container)

        splitter.setSizes([200, 400])
        layout.addWidget(splitter)

    def _refresh(self):
        patients = db.get_patients()
        self.patient_tree.clear()
        for p in patients:
            item = QTreeWidgetItem(self.patient_tree)
            item.setText(0, p["patient_name"])
            item.setText(1, str(p["roi_count"]))
            item.setData(0, Qt.ItemDataRole.UserRole, p["id"])
        self._current_patient_id = None
        self.roi_table.setRowCount(0)

    def _on_patient_clicked(self, item, col):
        pid = item.data(0, Qt.ItemDataRole.UserRole)
        if pid is None:
            return
        self._current_patient_id = pid
        self._load_rois(pid)

    def _load_rois(self, patient_id):
        rois = db.get_patient_rois(patient_id)
        self.roi_table.setRowCount(0)
        self.roi_table.setRowCount(len(rois))
        for i, r in enumerate(rois):
            device = " ".join(filter(None, [
                str(r.get("manufacturer", "")),
                str(r.get("manufacturer_model", "")),
                (str(r.get("magnetic_field_strength", "")) + "T") if r.get("magnetic_field_strength") else "",
            ])).strip()
            items = [
                QTableWidgetItem(str(r.get("roi_number", ""))),
                QTableWidgetItem(str(r.get("method", ""))),
                QTableWidgetItem(str(r.get("file_basename", ""))),
                QTableWidgetItem(device),
                QTableWidgetItem(str(r.get("body_part", ""))),
                QTableWidgetItem(str(r.get("protocol_name", "") or str(r.get("series_description", "")))),
                QTableWidgetItem(f"{r['area_mm2']:.2f}" if r.get("area_mm2") else ""),
                QTableWidgetItem(f"{r['mean_hu']:.2f}" if r.get("mean_hu") else ""),
                QTableWidgetItem(f"{r['std_hu']:.2f}" if r.get("std_hu") else ""),
                QTableWidgetItem(
                    f"{r['min_hu']:.0f}/{r['max_hu']:.0f}"
                    if r.get("min_hu") and r.get("max_hu") else ""),
                QTableWidgetItem(str(r.get("study_date", ""))),
                QTableWidgetItem(str(r.get("created_at", ""))[:16]),
            ]
            # OCR text column
            ocr_text = ""
            if r.get("method") == "ocr":
                stats_json = r.get("stats_json", "{}")
                if stats_json:
                    import json
                    stats = json.loads(stats_json)
                    ocr_text = stats.get("OCR识别完整文本", "")
                    if len(ocr_text) > 120:
                        ocr_text = ocr_text[:120] + "..."
                    ocr_text = ocr_text.replace("\n", " | ")
            ocr_item = QTableWidgetItem(ocr_text)
            ocr_item.setToolTip("双击查看完整OCR文本")
            items.append(ocr_item)
            # Description column
            desc_text = str(r.get("roi_description", ""))
            desc_item = QTableWidgetItem(desc_text)
            items.append(desc_item)
            for j, it in enumerate(items):
                it.setData(Qt.ItemDataRole.UserRole, r["id"])
                self.roi_table.setItem(i, j, it)
            if r.get("method") == "sam_interactive":
                for j in range(self.roi_table.columnCount()):
                    w = self.roi_table.item(i, j)
                    if w:
                        w.setForeground(QColor(C_ACCENT))
            if r.get("method") == "ocr":
                for j in range(self.roi_table.columnCount()):
                    w = self.roi_table.item(i, j)
                    if w:
                        w.setForeground(QColor(C_GREEN))

    def _export_excel(self):
        rois = db.get_all_rois()
        if not rois:
            QMessageBox.warning(self, "提示", "无ROI数据可导出")
            return
        from PySide6.QtWidgets import QFileDialog
        import openpyxl
        from datetime import datetime
        default_name = f"ROI全部数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出Excel", default_name, "Excel文件 (*.xlsx)")
        if not path:
            return
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "ROI数据"
        headers = ["患者", "编号", "方法", "来源", "文件名", "设备", "制造商型号", "场强", "部位",
                   "协议", "序列描述", "层厚", "像素间距", "模态",
                   "面积mm²", "均值HU", "标准差", "最小值HU", "最大值HU", "中位数HU",
                   "像素数", "扫描日期", "扫描时间", "创建时间", "ROI描述", "OCR文本"]
        ws.append(headers)
        for r in rois:
            ocr_text = ""
            if r.get("method") == "ocr":
                stats_json = r.get("stats_json", "{}")
                if stats_json:
                    import json
                    stats = json.loads(stats_json)
                    ocr_text = stats.get("OCR识别完整文本", "")
            ws.append([
                r.get("patient_name", ""),
                r.get("roi_number", ""), r.get("method", ""), r.get("source", ""),
                r.get("file_basename", ""), r.get("manufacturer", ""),
                r.get("manufacturer_model", ""), r.get("magnetic_field_strength", ""),
                r.get("body_part", ""), r.get("protocol_name", ""),
                r.get("series_description", ""), r.get("slice_thickness", ""),
                r.get("pixel_spacing", ""), r.get("modality", ""),
                r.get("area_mm2"), r.get("mean_hu"), r.get("std_hu"),
                r.get("min_hu"), r.get("max_hu"), r.get("median_hu"),
                r.get("pixel_count"), r.get("study_date", ""), r.get("study_time", ""),
                str(r.get("created_at", ""))[:19], r.get("roi_description", ""),
                ocr_text,
            ])
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 14
        ws.column_dimensions['Z'].width = 50  # OCR text column
        ws.column_dimensions['Y'].width = 30  # ROI description column
        wb.save(path)
        QMessageBox.information(self, "导出成功", f"已导出 {len(rois)} 条记录至:\n{path}")

    def _delete_selected(self):
        # Check patient tree selection
        patient_ids = []
        patient_names = []
        for item in self.patient_tree.selectedItems():
            pid = item.data(0, Qt.ItemDataRole.UserRole)
            pname = item.text(0)
            if pid:
                patient_ids.append(pid)
                patient_names.append(pname)

        # Check ROI table selection
        roi_ids = set()
        for item in self.roi_table.selectedItems():
            rid = item.data(Qt.ItemDataRole.UserRole)
            if rid:
                roi_ids.add(rid)

        # Priority: patient-level deletion
        if patient_ids:
            reply = QMessageBox.question(
                self, "确认删除",
                f"确认删除 {len(patient_ids)} 个患者及其全部ROI数据？\n{', '.join(patient_names[:5])}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            conn = db.get_connection()
            for pid in patient_ids:
                conn.execute("DELETE FROM roi_records WHERE patient_id = ?", (pid,))
                conn.execute("DELETE FROM patients WHERE id = ?", (pid,))
            conn.commit()
            conn.close()
            self._refresh()
            return

        if roi_ids:
            reply = QMessageBox.question(
                self, "确认删除", f"确认删除 {len(roi_ids)} 条ROI记录？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            conn = db.get_connection()
            for rid in roi_ids:
                conn.execute("DELETE FROM roi_records WHERE id = ?", (rid,))
            conn.commit()
            conn.close()
            self._refresh()
            if self._current_patient_id:
                self._load_rois(self._current_patient_id)

    def _on_cell_double_click(self, idx):
        row = idx.row()
        item = self.roi_table.item(row, 0)
        if item is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        if rid is None:
            return
        conn = db.get_connection()
        r = conn.execute("SELECT * FROM roi_records WHERE id = ?", (rid,)).fetchone()
        conn.close()
        if not r:
            return
        r = dict(r)
        info_lines = ["═════ 扫描信息 ═════"]
        for k, label in [
            ("modality", "成像模态"), ("manufacturer", "制造商"),
            ("manufacturer_model", "设备型号"),
            ("magnetic_field_strength", "场强(T)"), ("kvp", "kVp"),
            ("xray_tube_current", "管电流(mA)"), ("exposure_time", "曝光时间(ms)"),
            ("body_part", "扫描部位"), ("protocol_name", "扫描协议"),
            ("series_description", "序列描述"),
            ("slice_thickness", "层厚(mm)"), ("pixel_spacing", "像素间距(mm)"),
            ("convolution_kernel", "卷积核"), ("reconstruction_diameter", "重建直径(mm)"),
            ("spiral_pitch_factor", "螺距因子"), ("gantry_detector_tilt", "机架倾斜"),
            ("table_height", "床高(mm)"), ("ctdivol", "CTDIvol(mGy)"),
            ("dlp", "DLP(mGy·cm)"),
            ("study_date", "扫描日期"), ("study_time", "扫描时间"),
        ]:
            v = r.get(k, "")
            if v:
                info_lines.append(f"  {label}: {v}")

        meta = json.loads(r.get("metadata_json", "{}")) if r.get("metadata_json") else {}
        meta_lines = ["───── DICOM 元数据 ─────"]
        for k, v in meta.items():
            if k not in ("制造商", "患者姓名", "患者ID"):
                meta_lines.append(f"  {k}: {v}")
            else:
                pass

        stats_data = json.loads(r.get("stats_json", "{}")) if r.get("stats_json") else {}
        stats_lines = ["───── ROI 统计 ─────"]
        # OCR records: show full text
        if r.get("method") == "ocr" and "OCR识别完整文本" in stats_data:
            ocr_text = stats_data.get("OCR识别完整文本", "")
            stats_lines.append(f"  OCR完整文本:")
            for line in ocr_text.split("\n"):
                if line.strip():
                    stats_lines.append(f"    {line}")
            floats_str = stats_data.get("浮点数", "")
            if floats_str:
                stats_lines.append(f"  浮点数: {floats_str}")
        elif stats_data:
            for k, v in stats_data.items():
                stats_lines.append(f"  {k}: {v}")
        else:
            stats_lines.append(f"  面积: {r.get('area_mm2', '') or '-'}")
            stats_lines.append(f"  均值: {r.get('mean_hu', '') or '-'} HU")
            stats_lines.append(f"  标准差: {r.get('std_hu', '') or '-'}")
            stats_lines.append(f"  Min: {r.get('min_hu', '') or '-'}")
            stats_lines.append(f"  Max: {r.get('max_hu', '') or '-'}")

        msg = (f"ROI #{r['roi_number']} ({r['method']})\n"
               f"患者: {r.get('patient_name', '') or '-'}\n"
               f"文件: {r['file_basename']}\n\n"
               + "\n".join(info_lines) + "\n"
               + "\n".join(meta_lines) + "\n"
               + "\n".join(stats_lines))
        QMessageBox.information(self, "ROI 详情", msg)

    def notify_new_roi(self):
        self._refresh()
