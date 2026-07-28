import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


@dataclass
class DicomFileInfo:
    file_path: str
    patient_id: str = ""
    patient_name: str = ""
    study_uid: str = ""
    study_date: str = ""
    study_description: str = ""
    series_uid: str = ""
    series_number: str = ""
    series_description: str = ""
    sop_uid: str = ""
    instance_number: str = ""
    modality: str = ""
    body_part: str = ""
    file_size_kb: float = 0.0


@dataclass
class SeriesInfo:
    uid: str
    number: str = ""
    description: str = ""
    modality: str = ""
    files: List[DicomFileInfo] = field(default_factory=list)
    file_count: int = 0


@dataclass
class StudyInfo:
    uid: str
    date: str = ""
    description: str = ""
    series: Dict[str, SeriesInfo] = field(default_factory=dict)
    file_count: int = 0


@dataclass
class PatientInfo:
    name: str
    patient_id: str = ""
    studies: Dict[str, StudyInfo] = field(default_factory=dict)
    file_count: int = 0
    series_count: int = 0


class DicomCluster:
    def __init__(self):
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required. Install: pip install pydicom")

    def _is_dicom_file(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 132:
            return False
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.pdf', '.doc', '.docx', '.txt', '.log', '.json'):
            return False
        try:
            with open(file_path, 'rb') as f:
                header = f.read(132)
                if len(header) >= 132 and header[128:132] == b'DICM':
                    return True
                f.seek(0)
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                tags_to_check = ['SOPClassUID', 'PatientName', 'Modality', 'StudyInstanceUID']
                for t in tags_to_check:
                    if hasattr(ds, t):
                        return True
                return hasattr(ds, 'file_meta') and ds.file_meta is not None
        except Exception:
            return False

    def _extract_tags(self, file_path: str) -> Optional[DicomFileInfo]:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
        except Exception:
            return None

        info = DicomFileInfo(file_path=file_path)
        info.patient_id = str(getattr(ds, 'PatientID', '')).strip()
        info.patient_name = str(getattr(ds, 'PatientName', '')).strip()
        info.study_uid = str(getattr(ds, 'StudyInstanceUID', '')).strip()
        info.study_date = str(getattr(ds, 'StudyDate', '')).strip()
        info.study_description = str(getattr(ds, 'StudyDescription', '')).strip()
        info.series_uid = str(getattr(ds, 'SeriesInstanceUID', '')).strip()
        info.series_number = str(getattr(ds, 'SeriesNumber', '')).strip()
        info.series_description = str(getattr(ds, 'SeriesDescription', '')).strip()
        info.sop_uid = str(getattr(ds, 'SOPInstanceUID', '')).strip()
        info.instance_number = str(getattr(ds, 'InstanceNumber', '')).strip()
        info.modality = str(getattr(ds, 'Modality', '')).strip()
        info.body_part = str(getattr(ds, 'BodyPartExamined', '')).strip()
        info.file_size_kb = round(os.path.getsize(file_path) / 1024, 1)

        return info

    def _make_patient_key(self, info: DicomFileInfo) -> str:
        ident = info.patient_id or hashlib.md5(info.patient_name.encode()).hexdigest()[:8]
        return ident

    def scan_folder(self, folder_path: str, progress_callback=None) -> Dict[str, PatientInfo]:
        patients: Dict[str, PatientInfo] = {}
        all_files = []

        for root, dirs, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(root, f)
                if self._is_dicom_file(fp):
                    all_files.append(fp)

        total = len(all_files)
        total_patients = 0

        for idx, fp in enumerate(all_files):
            info = self._extract_tags(fp)
            if info is None:
                continue

            pkey = self._make_patient_key(info)
            if pkey not in patients:
                patients[pkey] = PatientInfo(
                    name=info.patient_name or f"Unknown_{pkey}",
                    patient_id=info.patient_id
                )
                total_patients += 1

            patient = patients[pkey]
            patient.file_count += 1
            if not patient.patient_id and info.patient_id:
                patient.patient_id = info.patient_id
            if not patient.name or patient.name.startswith("Unknown"):
                patient.name = info.patient_name or patient.name

            study_uid = info.study_uid or "unknown_study"
            if study_uid not in patient.studies:
                patient.studies[study_uid] = StudyInfo(
                    uid=study_uid,
                    date=info.study_date,
                    description=info.study_description
                )
            study = patient.studies[study_uid]
            study.file_count += 1

            series_uid = info.series_uid or "unknown_series"
            if series_uid not in study.series:
                study.series[series_uid] = SeriesInfo(
                    uid=series_uid,
                    number=info.series_number,
                    description=info.series_description,
                    modality=info.modality
                )
                patient.series_count += 1
            series = study.series[series_uid]
            series.files.append(info)
            series.file_count += 1

            if progress_callback:
                progress_callback(idx + 1, total)

        return patients

    def get_all_files(self, patients: Dict[str, PatientInfo]) -> List[DicomFileInfo]:
        files = []
        for p in patients.values():
            for s in p.studies.values():
                for se in s.series.values():
                    files.extend(se.files)
        return files

    def export_tree_json(self, patients: Dict[str, PatientInfo], output_path: str):
        data = {}
        for pkey, p in patients.items():
            pdata = {"patient_name": p.name, "patient_id": p.patient_id, "studies": {}}
            for suid, s in p.studies.items():
                sdata = {"date": s.date, "description": s.description, "series": {}}
                for seuid, se in s.series.items():
                    sdata["series"][seuid] = {
                        "number": se.number,
                        "description": se.description,
                        "modality": se.modality,
                        "file_count": se.file_count,
                        "files": [os.path.basename(f.file_path) for f in se.files]
                    }
                pdata["studies"][suid] = sdata
            data[pkey] = pdata
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_summary(self, patients: Dict[str, PatientInfo]) -> dict:
        total_files = sum(p.file_count for p in patients.values())
        total_series = sum(p.series_count for p in patients.values())
        modalities = set()
        for p in patients.values():
            for s in p.studies.values():
                for se in s.series.values():
                    if se.modality:
                        modalities.add(se.modality)
        return {
            "patient_count": len(patients),
            "file_count": total_files,
            "series_count": total_series,
            "modalities": sorted(modalities),
        }
