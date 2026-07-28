import os
import sys
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QFileDialog, QMessageBox, QLabel, QProgressBar,
    QSplitter, QFrame, QScrollArea, QSizePolicy, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor

from PyCt6 import (
    CMainWindow, CFrame, CLabel, CButton, CLineEdit,
)

from dicom_cluster import DicomCluster
from ui.dicom_tree import DicomTreeWidget
from ui.roi_panel import RoiPanel
from ui.sam_panel import SamPanel
from ui.summary_panel import SummaryPanel

C_TOOLBAR = "#1E3A5F"
C_TOOLBAR_TEXT = "#FFFFFF"
C_SIDEBAR = "#F0F4F8"
C_CARD_BG = "#FFFFFF"
C_CARD_BORDER = "#93B4E8"
C_TITLE = "#0B1E35"
C_TEXT = "#1E293B"
C_SUBTEXT = "#475569"
C_MUTED = "#94A3B8"
C_ACCENT = "#2563EB"
C_ACCENT_HOVER = "#1D4ED8"
C_GREEN = "#059669"
C_GREEN_HOVER = "#047857"
C_RIGHT_BG = "#E8ECF0"
C_STATUS_BAR = "#E2E8F0"
C_STATUS_TEXT = "#475569"
C_SPLITTER = "#CBD5E1"


class ScanWorker(QThread):
    progress = Signal(int, int)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            cluster = DicomCluster()
            patients = cluster.scan_folder(
                self.folder_path,
                progress_callback=lambda c, t: self.progress.emit(c, t))
            self.done.emit(patients)
        except Exception as e:
            self.error.emit(str(e))


class UnifiedViewer(QScrollArea):
    sam_point_added = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(200, 200)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMouseTracking(True)
        self._label.mousePressEvent = self._on_mouse_press
        self.setWidget(self._label)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            QScrollArea {{ background-color: {C_RIGHT_BG}; border: 1px solid {C_SPLITTER}; border-radius: 6px; }}
        """)
        self._current_path = ""
        self._raw_pixels: np.ndarray = None
        self._image_rgb: np.ndarray = None
        self._window_center: float = 0.0
        self._window_width: float = 0.0
        self._pixel_min: float = 0.0
        self._pixel_max: float = 0.0
        self._sam_mode = False
        self._mask = None
        self._points = []
        self._point_labels = []
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

    def set_sam_mode(self, enabled: bool):
        self._sam_mode = enabled
        if not enabled:
            self._mask = None
            self._points = []
            self._point_labels = []
        self._refresh_display()

    @property
    def sam_mode(self):
        return self._sam_mode

    def set_mask(self, mask):
        self._mask = mask
        self._refresh_display()

    def clear_mask(self):
        self._mask = None
        self._points = []
        self._point_labels = []
        self._refresh_display()

    def clear(self):
        self._label.clear()
        self._label.setText("尚未加载图像")
        self._label.setStyleSheet(f"color: {C_MUTED}; font-size: 16px;")
        self._current_path = ""
        self._raw_pixels = None
        self._image_rgb = None
        self._mask = None
        self._points = []
        self._point_labels = []

    def load_dicom(self, file_path: str):
        try:
            import pydicom
            ds = pydicom.dcmread(file_path, force=True)
            pixels = ds.pixel_array
            if pixels.ndim == 3:
                pixels = pixels[:, :, 0]
            self._raw_pixels = pixels.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                self._raw_pixels = self._raw_pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            self._pixel_min = float(self._raw_pixels.min())
            self._pixel_max = float(self._raw_pixels.max())
            wc = getattr(ds, 'WindowCenter', None)
            ww = getattr(ds, 'WindowWidth', None)
            if wc is not None and ww is not None:
                self._window_center = float(wc[0] if hasattr(wc, '__len__') else wc)
                self._window_width = float(ww[0] if hasattr(ww, '__len__') else ww)
            else:
                self._window_center = (self._pixel_min + self._pixel_max) / 2
                self._window_width = self._pixel_max - self._pixel_min
            self._current_path = file_path
            self._mask = None
            self._points = []
            self._point_labels = []
            self._refresh_display()
            return (self._pixel_min, self._pixel_max, self._window_center, self._window_width)
        except Exception as e:
            self._label.setText(f"加载失败: {str(e)[:100]}")
            self._label.setStyleSheet(f"color: red; font-size: 13px;")
            self._raw_pixels = None
            self._image_rgb = None
            return None

    def set_window(self, center: float, width: float):
        self._window_center = center
        self._window_width = max(width, 1.0)
        self._refresh_display()

    def _refresh_display(self):
        if self._raw_pixels is None:
            return
        wc = self._window_center
        ww = self._window_width
        low = wc - ww / 2
        high = wc + ww / 2
        clipped = np.clip(self._raw_pixels, low, high)
        if high > low:
            gray = ((clipped - low) / (high - low) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(self._raw_pixels, dtype=np.uint8)

        if self._sam_mode or self._mask is not None:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self._image_rgb = display.copy()
            if self._mask is not None:
                m = self._mask
                if m.shape[:2] != display.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (display.shape[1], display.shape[0])).astype(bool)
                color_mask = np.zeros_like(display)
                color_mask[m] = [0, 255, 0]
                display = cv2.addWeighted(display, 1.0, color_mask, 0.5, 0)
            if self._sam_mode:
                for i, (px, py) in enumerate(self._points):
                    color = (0, 255, 0) if self._point_labels[i] == 1 else (255, 0, 0)
                    cv2.circle(display, (px, py), 5, color, -1)
                    cv2.circle(display, (px, py), 6, (255, 255, 255), 1)
            h, w, c = display.shape
            qimg = QImage(display.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
        else:
            self._image_rgb = None
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)

        avail_w = self.viewport().width() - 20
        avail_h = self.viewport().height() - 20
        if avail_w > 0 and avail_h > 0:
            pixmap = pixmap.scaled(avail_w, avail_h,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)

        self._scale = pixmap.width() / (w if pixmap.width() > 0 else 1)
        self._offset_x = (self.viewport().width() - pixmap.width()) // 2 if self.viewport().width() > 0 else 0
        self._offset_y = (self.viewport().height() - pixmap.height()) // 2 if self.viewport().height() > 0 else 0
        self._label.setPixmap(pixmap)

    def _on_mouse_press(self, event):
        if not self._sam_mode or self._raw_pixels is None:
            return
        pos = event.position()
        x = pos.x()
        y = pos.y()
        img_x = int((x - self._offset_x) / self._scale) if self._scale > 0 else 0
        img_y = int((y - self._offset_y) / self._scale) if self._scale > 0 else 0
        h = self._raw_pixels.shape[0]
        w = self._raw_pixels.shape[1]
        if 0 <= img_x < w and 0 <= img_y < h:
            label = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self._points.append((img_x, img_y))
            self._point_labels.append(label)
            self._refresh_display()
            self.sam_point_added.emit(img_x, img_y, label)

    @property
    def current_path(self):
        return self._current_path

    @property
    def raw_window_center(self):
        return self._window_center

    @property
    def raw_window_width(self):
        return self._window_width

    @property
    def image_rgb(self):
        return self._image_rgb


class MainWindow(CMainWindow):
    def __init__(self):
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "logo", "Gemini_Generated_Image_egithcegithcegit.png")
        super().__init__(width=1280, height=820, title="DICOM 自动化分析工具",
                         background_color=("#F8FAFC", "#F8FAFC"),
                         icon=logo_path if os.path.exists(logo_path) else None)
        self._scan_worker = None
        self._patients = None
        self._logo_path = logo_path if os.path.exists(logo_path) else None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{ background-color: {C_TOOLBAR}; border-bottom: 2px solid {C_GREEN}; }}
        """)
        toolbar.setFixedHeight(44)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 0, 12, 0)
        toolbar_layout.setSpacing(8)
        app_title = QLabel("DICOM 自动化分析工具")
        app_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        app_title.setStyleSheet(f"color: {C_TOOLBAR_TEXT};")
        toolbar_layout.addWidget(app_title)
        toolbar_layout.addStretch()
        if self._logo_path:
            logo_label = QLabel()
            logo_pix = QPixmap(self._logo_path).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(logo_pix)
            toolbar_layout.addWidget(logo_label)
        credit = QLabel("design by christ.paul90@gmail.com · all rights reserved")
        credit.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        credit.setStyleSheet(f"color: {C_TOOLBAR_TEXT}; opacity: 0.8; font-style: italic;")
        toolbar_layout.addWidget(credit)
        main_layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {C_SPLITTER}; }}")

        # === COLUMN 1: parameters ===
        left_panel = QFrame()
        left_panel.setMinimumWidth(290)
        left_panel.setStyleSheet(f"background-color: {C_SIDEBAR};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        path_section = QFrame()
        path_section.setStyleSheet(f"""
            QFrame {{ background-color: {C_CARD_BG}; border: 1px solid {C_CARD_BORDER}; border-radius: 6px; }}
        """)
        path_layout = QVBoxLayout(path_section)
        path_layout.setContentsMargins(8, 6, 8, 6)
        path_layout.setSpacing(5)
        path_input_row = QHBoxLayout()
        self.path_input = CLineEdit(master=path_section, width=180, height=24,
                                     placeholder_text="拖入文件夹...")
        self.path_input.line_edit().setStyleSheet(f"""
            QLineEdit {{ font-size: 10px; color: {C_TEXT}; background: {C_SIDEBAR};
                border: 1px solid {C_SPLITTER}; border-radius: 4px; padding: 2px 4px; }}
        """)
        self.path_input.line_edit().setMinimumWidth(140)
        path_input_row.addWidget(self.path_input.line_edit(), 1)
        browse_btn = CButton(master=path_section, text="...", width=28, height=24,
                              background_color=(C_ACCENT, C_ACCENT),
                              hover_color=(C_ACCENT_HOVER, C_ACCENT_HOVER),
                              text_color=("white", "white"), font_size=10)
        browse_btn.button().clicked.connect(self._browse_folder)
        path_input_row.addWidget(browse_btn.button())
        path_layout.addLayout(path_input_row)
        self.scan_btn = CButton(master=path_section, text="扫描DICOM文件",
                                 width=240, height=28,
                                 background_color=(C_GREEN, C_GREEN),
                                 hover_color=(C_GREEN_HOVER, C_GREEN_HOVER),
                                 text_color=("white", "white"),
                                 font_size=11, font_style="bold")
        self.scan_btn.button().clicked.connect(self._scan_folder)
        path_layout.addWidget(self.scan_btn.button())
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        self.scan_progress.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid {C_SPLITTER}; border-radius: 3px;
                background: {C_SIDEBAR}; height: 8px; text-align: center;
                font-size: 8px; color: {C_TITLE}; font-weight: bold; }}
            QProgressBar::chunk {{ background: {C_GREEN}; border-radius: 2px; }}
        """)
        path_layout.addWidget(self.scan_progress)
        left_layout.addWidget(path_section)

        summary_section = QFrame()
        summary_section.setStyleSheet(f"""
            QFrame {{ background-color: {C_CARD_BG}; border: 1px solid {C_SPLITTER}; border-radius: 6px; }}
        """)
        summary_layout = QVBoxLayout(summary_section)
        summary_layout.setContentsMargins(8, 5, 8, 5)
        summary_layout.setSpacing(2)
        self.summary_rows = {}
        for key, label in [("patient", "患者"), ("series", "序列"),
                            ("files", "文件"), ("mod", "模态")]:
            lbl = QLabel(f"{label}: --")
            lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px; font-weight: 600;")
            lbl.setWordWrap(False)
            lbl.setMinimumWidth(120)
            self.summary_rows[key] = lbl
            summary_layout.addWidget(lbl)
        left_layout.addWidget(summary_section)

        action_section = QFrame()
        action_section.setStyleSheet(f"""
            QFrame {{ background-color: {C_CARD_BG}; border: 1px solid {C_CARD_BORDER}; border-radius: 6px; }}
        """)
        action_layout = QVBoxLayout(action_section)
        action_layout.setContentsMargins(8, 5, 8, 6)
        action_layout.setSpacing(5)
        tab_row = QHBoxLayout()
        tab_row.setSpacing(3)
        self.tab_roi = CButton(master=action_section, text="ROI检测", width=100, height=26,
                                background_color=(C_ACCENT, C_ACCENT),
                                hover_color=(C_ACCENT_HOVER, C_ACCENT_HOVER),
                                text_color=("white", "white"), font_size=10, font_style="bold")
        self.tab_roi.button().clicked.connect(lambda: self._switch_action(0))
        self.tab_sam = CButton(master=action_section, text="SAM分割", width=100, height=26, font_size=10)
        self.tab_sam.button().clicked.connect(lambda: self._switch_action(1))

        self.tab_summary = CButton(master=action_section, text="数据汇总", width=100, height=26, font_size=10)
        self.tab_summary.button().clicked.connect(lambda: self._switch_action(2))

        tab_row.addWidget(self.tab_roi.button())
        tab_row.addWidget(self.tab_sam.button())
        tab_row.addWidget(self.tab_summary.button())
        tab_row.addStretch()
        action_layout.addLayout(tab_row)
        self.action_stack = QStackedWidget()
        self.action_stack.setStyleSheet(f"background-color: {C_CARD_BG};")
        self.roi_panel = RoiPanel()
        self.sam_panel = SamPanel()
        self.action_stack.addWidget(self.roi_panel)
        self.action_stack.addWidget(self.sam_panel)
        action_layout.addWidget(self.action_stack)
        left_layout.addWidget(action_section)

        # === COLUMN 2: patient tree ===
        tree_panel = QFrame()
        tree_panel.setMinimumWidth(200)
        tree_panel.setStyleSheet(f"background-color: {C_SIDEBAR};")
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(6, 6, 6, 6)
        tree_layout.setSpacing(0)
        tree_label = QLabel("病人列表")
        tree_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        tree_label.setStyleSheet(f"color: {C_TITLE}; padding-bottom: 4px;")
        tree_layout.addWidget(tree_label)
        self.tree = DicomTreeWidget()
        self.tree.setStyleSheet(f"""
            QTreeWidget {{ font-size: 11px; color: {C_TEXT}; background-color: {C_CARD_BG};
                border: 1px solid {C_SPLITTER}; border-radius: 6px;
                alternate-background-color: {C_SIDEBAR}; }}
            QTreeWidget::item {{ padding: 2px 0; color: {C_TEXT}; }}
            QTreeWidget::item:selected {{ background-color: {C_ACCENT}; color: white; }}
            QHeaderView::section {{ background-color: {C_TOOLBAR}; color: {C_TOOLBAR_TEXT};
                padding: 3px 4px; font-size: 10px; font-weight: bold; border: 1px solid {C_TOOLBAR}; }}
        """)
        tree_layout.addWidget(self.tree)
        self.tree.file_clicked.connect(self._on_file_clicked)

        # === COLUMN 3: unified viewer + controls ===
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background-color: {C_RIGHT_BG};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(3)

        self.viewer = UnifiedViewer()
        self.viewer.clear()
        self.viewer.sam_point_added.connect(self._on_sam_point)
        right_layout.addWidget(self.viewer)

        # File info + reset
        info_row = QHBoxLayout()
        info_row.setSpacing(4)
        self.file_info = QLabel("")
        self.file_info.setWordWrap(True)
        self.file_info.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 9px;")
        info_row.addWidget(self.file_info)
        info_row.addStretch()
        self.reset_btn = QLabel("↺")
        self.reset_btn.setFixedSize(20, 20)
        self.reset_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reset_btn.setStyleSheet(f"""
            QLabel {{ color: {C_ACCENT}; font-size: 14px; font-weight: bold;
                border: 1px solid {C_ACCENT}; border-radius: 10px;
                background: {C_CARD_BG}; }}
            QLabel:hover {{ background: {C_ACCENT}; color: white; }}
        """)
        self.reset_btn.setToolTip("重置窗宽窗位")
        self.reset_btn.mousePressEvent = lambda e: self._reset_windowing()
        info_row.addWidget(self.reset_btn)
        right_layout.addLayout(info_row)

        # Windowing sliders
        window_frame = QFrame()
        window_frame.setStyleSheet(f"""
            QFrame {{ background-color: {C_CARD_BG}; border: 1px solid {C_SPLITTER}; border-radius: 4px; }}
        """)
        window_layout = QVBoxLayout(window_frame)
        window_layout.setContentsMargins(6, 2, 6, 4)
        window_layout.setSpacing(1)

        ww_row = QHBoxLayout()
        ww_label = QLabel("窗宽")
        ww_label.setFixedWidth(26)
        ww_label.setStyleSheet(f"color: {C_TEXT}; font-size: 10px; font-weight: 600;")
        ww_row.addWidget(ww_label)
        self.ww_slider = QSlider(Qt.Orientation.Horizontal)
        self.ww_slider.setMinimum(1)
        self.ww_slider.setMaximum(10000)
        self.ww_slider.setValue(1000)
        self.ww_slider.setStyleSheet(self._slider_style())
        self.ww_slider.valueChanged.connect(self._on_windowing_changed)
        ww_row.addWidget(self.ww_slider)
        self.ww_value = QLabel("0")
        self.ww_value.setFixedWidth(44)
        self.ww_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.ww_value.setStyleSheet(f"color: {C_ACCENT}; font-size: 9px; font-weight: bold;")
        ww_row.addWidget(self.ww_value)
        window_layout.addLayout(ww_row)

        wc_row = QHBoxLayout()
        wc_label = QLabel("窗位")
        wc_label.setFixedWidth(26)
        wc_label.setStyleSheet(f"color: {C_TEXT}; font-size: 10px; font-weight: 600;")
        wc_row.addWidget(wc_label)
        self.wc_slider = QSlider(Qt.Orientation.Horizontal)
        self.wc_slider.setMinimum(-10000)
        self.wc_slider.setMaximum(10000)
        self.wc_slider.setValue(0)
        self.wc_slider.setStyleSheet(self._slider_style())
        self.wc_slider.valueChanged.connect(self._on_windowing_changed)
        wc_row.addWidget(self.wc_slider)
        self.wc_value = QLabel("0")
        self.wc_value.setFixedWidth(44)
        self.wc_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.wc_value.setStyleSheet(f"color: {C_ACCENT}; font-size: 9px; font-weight: bold;")
        wc_row.addWidget(self.wc_value)
        window_layout.addLayout(wc_row)

        right_layout.addWidget(window_frame)

        # Page stack: switch between tree+viewer and summary
        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet(f"background-color: {C_RIGHT_BG};")

        tree_viewer_page = QWidget()
        tv_layout = QHBoxLayout(tree_viewer_page)
        tv_layout.setContentsMargins(0, 0, 0, 0)
        tv_layout.setSpacing(0)
        tv_layout.addWidget(tree_panel)
        tv_layout.addWidget(right_panel)
        self.page_stack.addWidget(tree_viewer_page)

        self.summary_panel = SummaryPanel()
        self.page_stack.addWidget(self.summary_panel)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.page_stack)
        splitter.setSizes([280, 1000])
        main_layout.addWidget(splitter)

        status_frame = QFrame()
        status_frame.setFixedHeight(20)
        status_frame.setStyleSheet(f"""
            background-color: {C_STATUS_BAR}; border-top: 1px solid {C_SPLITTER};
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 0, 8, 0)
        self.status_label = QLabel("就绪 — 请选择 DICOM 文件夹开始扫描")
        self.status_label.setStyleSheet(f"color: {C_STATUS_TEXT}; font-size: 9px; font-weight: 500;")
        status_layout.addWidget(self.status_label)
        main_layout.addWidget(status_frame)

        self.setLayout(main_layout)
        self.setAcceptDrops(True)
        self._suppress_slider = False

        # Wire SAM panel to viewer
        self.sam_panel.set_viewer(self.viewer)
        self.roi_panel.set_viewer(self.viewer)

    def _slider_style(self):
        return f"""
            QSlider::groove:horizontal {{ height: 5px; background: {C_SPLITTER}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -4px 0;
                background: {C_ACCENT}; border: 2px solid white; border-radius: 6px; }}
            QSlider::handle:horizontal:hover {{ background: {C_ACCENT_HOVER}; }}
            QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 2px; }}
        """

    def _on_file_clicked(self, file_path):
        self._display_dicom(file_path)
        siblings = self.tree.get_sibling_files(file_path)
        self.sam_panel.set_file_context(file_path, siblings)

    def _on_sam_point(self, x, y, label):
        self.sam_panel.on_viewer_point(x, y, label)

    def _display_dicom(self, file_path):
        result = self.viewer.load_dicom(file_path)
        if result is None:
            self.file_info.setText(f"加载失败: {os.path.basename(file_path)}")
            return
        pixel_min, pixel_max, wc, ww = result
        bname = os.path.basename(file_path)
        self.file_info.setText(f"📄 {bname}  [{pixel_min:.0f}, {pixel_max:.0f}]")
        self._suppress_slider = True
        range_val = max(pixel_max - pixel_min, 1)
        self.ww_slider.setMinimum(max(1, int(range_val * 0.01)))
        self.ww_slider.setMaximum(int(range_val * 2))
        self.ww_slider.setValue(int(ww))
        self.ww_value.setText(f"{ww:.0f}")
        self.wc_slider.setMinimum(int(pixel_min))
        self.wc_slider.setMaximum(int(pixel_max))
        self.wc_slider.setValue(int(wc))
        self.wc_value.setText(f"{wc:.0f}")
        self._suppress_slider = False

    def _on_windowing_changed(self):
        if self._suppress_slider:
            return
        ww = float(self.ww_slider.value())
        wc = float(self.wc_slider.value())
        self.ww_value.setText(f"{ww:.0f}")
        self.wc_value.setText(f"{wc:.0f}")
        self.viewer.set_window(wc, ww)

    def _reset_windowing(self):
        if self.viewer.current_path:
            self._display_dicom(self.viewer.current_path)

    def _show_first_image(self):
        first = self.tree.get_first_file()
        if first:
            self._on_file_clicked(first)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if path:
            self.path_input.line_edit().setText(path)

    def _scan_folder(self):
        folder = self.path_input.line_edit().text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", "请输入或选择有效的文件夹路径")
            return
        self.scan_btn.button().setEnabled(False)
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        self.status_label.setText("正在扫描DICOM文件...")
        self._scan_worker = ScanWorker(folder)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.start()

    def _on_scan_progress(self, current, total):
        self.scan_progress.setMaximum(total)
        self.scan_progress.setValue(current)
        self.scan_progress.setFormat(f"扫描中... {current}/{total}")

    def _on_scan_done(self, patients):
        self._patients = patients
        self.tree.populate(patients)
        cluster = DicomCluster()
        summary = cluster.get_summary(patients)
        self.summary_rows["patient"].setText(f"患者: {summary['patient_count']}")
        self.summary_rows["series"].setText(f"序列: {summary['series_count']}")
        self.summary_rows["files"].setText(f"文件: {summary['file_count']}")
        self.summary_rows["mod"].setText(f"模态: {', '.join(summary['modalities']) or 'N/A'}")
        self.scan_progress.setVisible(False)
        self.scan_btn.button().setEnabled(True)
        self.status_label.setText(
            f"扫描完成: {summary['patient_count']} 患者, "
            f"{summary['series_count']} 序列, {summary['file_count']} 文件")
        self._show_first_image()

    def _on_scan_error(self, err):
        self.scan_progress.setVisible(False)
        self.scan_btn.button().setEnabled(True)
        self.status_label.setText(f"扫描失败: {err}")
        QMessageBox.critical(self, "错误", f"DICOM扫描失败:\n{err}")

    def _switch_action(self, idx):
        self.action_stack.setCurrentIndex(idx)
        if idx == 0:
            self.page_stack.setCurrentIndex(0)
            self.viewer.set_sam_mode(False)
            self.tab_roi.button().setStyleSheet(
                f"background-color: {C_ACCENT}; color: white; font-weight: bold;")
            self.tab_sam.button().setStyleSheet("")
            self.tab_summary.button().setStyleSheet("")
        elif idx == 1:
            self.page_stack.setCurrentIndex(0)
            self.viewer.set_sam_mode(True)
            self.tab_sam.button().setStyleSheet(
                f"background-color: {C_ACCENT}; color: white; font-weight: bold;")
            self.tab_roi.button().setStyleSheet("")
            self.tab_summary.button().setStyleSheet("")
        else:
            self.viewer.set_sam_mode(False)
            self.page_stack.setCurrentIndex(1)
            self.tab_summary.button().setStyleSheet(
                f"background-color: {C_ACCENT}; color: white; font-weight: bold;")
            self.tab_roi.button().setStyleSheet("")
            self.tab_sam.button().setStyleSheet("")
            self.summary_panel._refresh()

    def get_selected_files(self):
        return self.tree.get_selected_files()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.path_input.line_edit().setText(path)
                self._scan_folder()
                return
