import os
from PySide6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from dicom_cluster import PatientInfo, StudyInfo, SeriesInfo, DicomFileInfo


class DicomTreeWidget(QTreeWidget):
    file_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Name / UID", "Info", "Files"])
        self.setColumnCount(3)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setAnimated(True)
        self.setIndentation(16)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._patients: dict = {}
        self._file_items: dict = {}
        self._first_file: str = ""
        self.itemClicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, item, column):
        d = item.data(0, Qt.ItemDataRole.UserRole)
        if d and d["type"] == "file":
            self.file_clicked.emit(str(d["data"].file_path))

    def populate(self, patients: dict):
        self.clear()
        self._patients = patients
        self._file_items = {}
        self._first_file = ""
        sorted_patients = sorted(patients.values(),
                                  key=lambda p: p.name.lower())

        for p in sorted_patients:
            p_item = QTreeWidgetItem(self)
            p_item.setText(0, f"🏥 {p.name or p.patient_id or 'Unknown'}")
            p_item.setText(1, f"ID: {p.patient_id}")
            p_item.setText(2, str(p.file_count))
            p_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "patient", "data": p})
            p_font = QFont()
            p_font.setBold(True)
            p_font.setPointSize(11)
            p_item.setFont(0, p_font)

            sorted_studies = sorted(p.studies.items(),
                                     key=lambda x: x[1].date or "99999999",
                                     reverse=True)
            for suid, s in sorted_studies:
                s_item = QTreeWidgetItem(p_item)
                label = f"📅 {s.date}" if s.date else "📅 Study"
                if s.description:
                    label += f" - {s.description}"
                s_item.setText(0, label)
                s_item.setText(1, f"UID: {suid[:12]}...")
                s_item.setText(2, str(s.file_count))
                s_item.setData(0, Qt.ItemDataRole.UserRole,
                               {"type": "study", "data": s, "uid": suid})

                sorted_series = sorted(s.series.items(),
                                        key=lambda x: x[1].number or "99999")
                for seuid, se in sorted_series:
                    se_item = QTreeWidgetItem(s_item)
                    se_label = f"📁 Series {se.number}"
                    if se.description:
                        se_label += f" - {se.description}"
                    if se.modality:
                        se_label += f" [{se.modality}]"
                    se_item.setText(0, se_label)
                    se_item.setText(1, f"UID: {seuid[:12]}...")
                    se_item.setText(2, str(se.file_count))
                    se_item.setData(0, Qt.ItemDataRole.UserRole,
                                    {"type": "series", "data": se, "uid": seuid})

                    sorted_files = sorted(se.files,
                                           key=lambda f: f.instance_number or f.file_path)
                    for finfo in sorted_files:
                        f_item = QTreeWidgetItem(se_item)
                        fname = f"📄 {os.path.basename(finfo.file_path)}"
                        if finfo.instance_number:
                            fname += f" [Inst:{finfo.instance_number}]"
                        f_item.setText(0, fname)
                        f_item.setText(1, f"{finfo.file_size_kb:.0f} KB")
                        f_item.setText(2, finfo.modality or "")
                        f_item.setData(0, Qt.ItemDataRole.UserRole,
                                       {"type": "file", "data": finfo, "series_uid": seuid})
                        self._file_items[finfo.file_path] = f_item
                        if not self._first_file:
                            self._first_file = finfo.file_path

        self.expandAll()

    def get_first_file(self) -> str:
        return self._first_file

    def get_sibling_files(self, file_path: str) -> list:
        item = self._file_items.get(file_path)
        if not item:
            return []
        d = item.data(0, Qt.ItemDataRole.UserRole)
        if not d or d.get("type") != "file":
            return []
        se_uid = d.get("series_uid")
        if not se_uid:
            return []
        siblings = []
        def _walk(parent):
            for i in range(parent.childCount()):
                child = parent.child(i)
                cd = child.data(0, Qt.ItemDataRole.UserRole)
                if cd and cd.get("type") == "file" and cd.get("series_uid") == se_uid:
                    siblings.append(cd["data"].file_path)
                _walk(child)
        for i in range(self.topLevelItemCount()):
            _walk(self.topLevelItem(i))
        return siblings

    def get_selected_files(self) -> list:
        files = []
        for item in self.selectedItems():
            d = item.data(0, Qt.ItemDataRole.UserRole)
            if not d:
                continue
            if d["type"] == "file":
                files.append(d["data"])
            elif d["type"] == "series":
                for f in d["data"].files:
                    if f not in files:
                        files.append(f)
            elif d["type"] == "study":
                for se in d["data"].series.values():
                    for f in se.files:
                        if f not in files:
                            files.append(f)
            elif d["type"] == "patient":
                for s in d["data"].studies.values():
                    for se in s.series.values():
                        for f in se.files:
                            if f not in files:
                                files.append(f)
        return files
