import os
import json
import sqlite3
import time
from typing import Dict, List, Optional, Any

from app_paths import get_db_path

DB_PATH = str(get_db_path())


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT UNIQUE NOT NULL,
            patient_id TEXT,
            patient_sex TEXT,
            patient_birth_date TEXT,
            patient_age TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS roi_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
            study_date TEXT,
            study_time TEXT,
            study_description TEXT,
            series_description TEXT,
            series_number TEXT,
            file_path TEXT,
            file_basename TEXT,
            roi_number INTEGER,
            method TEXT,
            source TEXT,
            manufacturer TEXT,
            manufacturer_model TEXT,
            magnetic_field_strength TEXT,
            body_part TEXT,
            protocol_name TEXT,
            slice_thickness TEXT,
            pixel_spacing TEXT,
            modality TEXT,
            length_mm REAL,
            angle_deg REAL,
            kvp TEXT,
            xray_tube_current TEXT,
            exposure_time TEXT,
            convolution_kernel TEXT,
            reconstruction_diameter TEXT,
            ctdivol TEXT,
            dlp TEXT,
            spiral_pitch_factor TEXT,
            gantry_detector_tilt TEXT,
            table_height TEXT,
            area_mm2 REAL,
            mean_hu REAL,
            std_hu REAL,
            min_hu REAL,
            max_hu REAL,
            median_hu REAL,
            pixel_count INTEGER,
            metadata_json TEXT,
            stats_json TEXT,
            contour_json TEXT,
            saved_png TEXT,
            saved_json TEXT,
            roi_description TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_roi_patient ON roi_records(patient_id);
        CREATE INDEX IF NOT EXISTS idx_roi_method ON roi_records(method);
    """)

    # Migrate existing tables — add columns if missing
    new_cols = {
        "study_time": "TEXT",
        "series_number": "TEXT",
        "manufacturer": "TEXT",
        "manufacturer_model": "TEXT",
        "magnetic_field_strength": "TEXT",
        "body_part": "TEXT",
        "protocol_name": "TEXT",
        "slice_thickness": "TEXT",
        "pixel_spacing": "TEXT",
        "modality": "TEXT",
        "kvp": "TEXT",
        "xray_tube_current": "TEXT",
        "exposure_time": "TEXT",
        "convolution_kernel": "TEXT",
        "reconstruction_diameter": "TEXT",
        "ctdivol": "TEXT",
        "dlp": "TEXT",
        "spiral_pitch_factor": "TEXT",
        "gantry_detector_tilt": "TEXT",
        "table_height": "TEXT",
        "roi_description": "TEXT",
        "length_mm": "REAL",
        "angle_deg": "REAL",
    }
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(roi_records)").fetchall()}
    for col, ctype in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE roi_records ADD COLUMN {col} {ctype}")
    for col in ["patient_sex", "patient_birth_date", "patient_age"]:
        existing_p = {row["name"] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
        if col not in existing_p:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()


def ensure_patient(patient_name: str, patient_id: str = "",
                   patient_sex: str = "", patient_birth_date: str = "",
                   patient_age: str = "") -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM patients WHERE patient_name = ?", (patient_name,)
    ).fetchone()
    if row:
        pid = row["id"]
        updates = []
        vals = []
        if patient_id:
            updates.append("patient_id = ?"); vals.append(patient_id)
        if patient_sex:
            updates.append("patient_sex = ?"); vals.append(patient_sex)
        if patient_birth_date:
            updates.append("patient_birth_date = ?"); vals.append(patient_birth_date)
        if patient_age:
            updates.append("patient_age = ?"); vals.append(patient_age)
        if updates:
            vals.append(pid)
            conn.execute(f"UPDATE patients SET {', '.join(updates)} WHERE id = ?", vals)
    else:
        cur = conn.execute(
            "INSERT INTO patients (patient_name, patient_id, patient_sex, patient_birth_date, patient_age) "
            "VALUES (?, ?, ?, ?, ?)",
            (patient_name, patient_id, patient_sex, patient_birth_date, patient_age))
        pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid


def get_next_roi_number(patient_id: int) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COALESCE(MAX(roi_number), 0) + 1 FROM roi_records WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    num = row[0]
    conn.close()
    return num


def insert_roi(patient_name: str,
               patient_id: str = "",
               patient_sex: str = "",
               patient_birth_date: str = "",
               patient_age: str = "",
               file_path: str = "",
               method: str = "",
               source: str = "",
               metadata: Dict[str, Any] = None,
               stats: Dict[str, Any] = None,
               contour: List = None,
               area_mm2: float = None,
               mean_hu: float = None,
               std_hu: float = None,
               min_hu: float = None,
               max_hu: float = None,
               median_hu: float = None,
               pixel_count: int = None,
               saved_png: str = None,
               saved_json: str = None,
               roi_description: str = None,
               length_mm: float = None,
               angle_deg: float = None,
               study_date: str = None,
               study_time: str = None,
               study_description: str = None,
               series_description: str = None,
               series_number: str = None,
               manufacturer: str = None,
               manufacturer_model: str = None,
               magnetic_field_strength: str = None,
               body_part: str = None,
               protocol_name: str = None,
               slice_thickness: str = None,
               pixel_spacing: str = None,
               modality: str = None,
               kvp: str = None,
               xray_tube_current: str = None,
               exposure_time: str = None,
               convolution_kernel: str = None,
               reconstruction_diameter: str = None,
               ctdivol: str = None,
               dlp: str = None,
               spiral_pitch_factor: str = None,
               gantry_detector_tilt: str = None,
               table_height: str = None) -> int:

    pid = ensure_patient(patient_name, patient_id, patient_sex, patient_birth_date, patient_age)
    roi_num = get_next_roi_number(pid)
    bname = os.path.basename(file_path) if file_path else ""

    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO roi_records (
            patient_id, study_date, study_time, study_description,
            series_description, series_number, file_path, file_basename,
            roi_number, method, source,
            manufacturer, manufacturer_model, magnetic_field_strength,
            body_part, protocol_name, slice_thickness, pixel_spacing,
            modality, kvp, xray_tube_current, exposure_time,
            convolution_kernel, reconstruction_diameter,
            ctdivol, dlp, spiral_pitch_factor,
            gantry_detector_tilt, table_height,
            area_mm2, mean_hu, std_hu, min_hu, max_hu, median_hu, pixel_count,
            metadata_json, stats_json, contour_json, saved_png, saved_json, roi_description, length_mm, angle_deg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?,
                  ?, ?,
                  ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?,
                  ?, ?, ?)
    """, (
        pid,
        study_date or "",
        study_time or "",
        study_description or "",
        series_description or "",
        series_number or "",
        file_path or "",
        bname,
        roi_num,
        method,
        source,
        manufacturer or "",
        manufacturer_model or "",
        magnetic_field_strength or "",
        body_part or "",
        protocol_name or "",
        slice_thickness or "",
        pixel_spacing or "",
        modality or "",
        kvp or "",
        xray_tube_current or "",
        exposure_time or "",
        convolution_kernel or "",
        reconstruction_diameter or "",
        ctdivol or "",
        dlp or "",
        spiral_pitch_factor or "",
        gantry_detector_tilt or "",
        table_height or "",
        area_mm2,
        mean_hu,
        std_hu,
        min_hu,
        max_hu,
        median_hu,
        pixel_count,
        json.dumps(metadata, ensure_ascii=False) if metadata else "{}",
        json.dumps(stats, ensure_ascii=False) if stats else "{}",
        json.dumps(contour, ensure_ascii=False) if contour else "[]",
        saved_png or "",
        saved_json or "",
        roi_description or "",
        length_mm,
        angle_deg,
    ))
    record_id = cur.lastrowid
    conn.commit()
    conn.close()
    return record_id


def update_roi(record_id: int, **kwargs):
    if not kwargs:
        return
    allowed = {
        "area_mm2", "mean_hu", "std_hu", "min_hu", "max_hu", "median_hu",
        "pixel_count", "method", "source", "metadata_json", "stats_json",
        "contour_json", "study_date", "study_description", "series_description",
        "saved_png", "saved_json", "study_time", "series_number",
        "manufacturer", "manufacturer_model", "magnetic_field_strength",
        "body_part", "protocol_name", "slice_thickness", "pixel_spacing",
    }
    sets = []
    vals = []
    for k, v in kwargs.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False))
    if not sets:
        return
    vals.append(record_id)
    conn = get_connection()
    conn.execute(f"UPDATE roi_records SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def delete_roi(record_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM roi_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def delete_patient_rois(patient_name: str):
    conn = get_connection()
    conn.execute("""
        DELETE FROM roi_records WHERE patient_id = (
            SELECT id FROM patients WHERE patient_name = ?
        )
    """, (patient_name,))
    conn.commit()
    conn.close()


def get_patients() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.*, COUNT(r.id) as roi_count
        FROM patients p
        LEFT JOIN roi_records r ON r.patient_id = p.id
        GROUP BY p.id
        ORDER BY p.patient_name
    """).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_patient_rois(patient_id: int) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM roi_records
        WHERE patient_id = ?
        ORDER BY roi_number
    """, (patient_id,)).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_rois() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT r.*, p.patient_name, p.patient_id as patient_dicom_id
        FROM roi_records r
        JOIN patients p ON p.id = r.patient_id
        ORDER BY p.patient_name, r.roi_number
    """).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_patient_by_name(name: str) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM patients WHERE patient_name = ?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def parse_stats_text(stats: Dict) -> Dict:
    result = {}
    if not stats:
        return result
    for k in ["像素数", "面积(mm²)", "均值", "标准差", "最小值", "最大值", "中位数"]:
        v = stats.get(k, "")
        if isinstance(v, str):
            v = v.replace(" HU", "").replace("px", "").replace("mm²", "").strip()
        try:
            result[k] = float(v) if v not in ("", "选区为空") else None
        except (ValueError, TypeError):
            result[k] = None
    return result


def extract_dicom_meta(file_path: str) -> Dict:
    try:
        import pydicom
        ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
        info = {}
        tag_map = {
            "PatientName": "patient_name",
            "PatientID": "patient_id",
            "PatientSex": "patient_sex",
            "PatientBirthDate": "patient_birth_date",
            "PatientAge": "patient_age",
            "StudyDate": "study_date",
            "StudyTime": "study_time",
            "StudyDescription": "study_description",
            "SeriesDescription": "series_description",
            "SeriesNumber": "series_number",
            "Modality": "modality",
            "Manufacturer": "manufacturer",
            "ManufacturerModelName": "manufacturer_model",
            "MagneticFieldStrength": "magnetic_field_strength",
            "BodyPartExamined": "body_part",
            "ProtocolName": "protocol_name",
            "SliceThickness": "slice_thickness",
            "KVP": "kvp",
            "XRayTubeCurrent": "xray_tube_current",
            "ExposureTime": "exposure_time",
            "ConvolutionKernel": "convolution_kernel",
            "ReconstructionDiameter": "reconstruction_diameter",
            "CTDIvol": "ctdivol",
            "DoseLengthProduct": "dlp",
            "SpiralPitchFactor": "spiral_pitch_factor",
            "GantryDetectorTilt": "gantry_detector_tilt",
            "TableHeight": "table_height",
        }
        for tag, key in tag_map.items():
            val = getattr(ds, tag, None)
            if val not in (None, "", "None"):
                info[key] = str(val).strip()
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None:
            try:
                info["pixel_spacing"] = f"{float(ps[0]):.3f}×{float(ps[1]):.3f}"
            except Exception:
                pass
        return info
    except Exception:
        return {}


def add_roi_from_sam(file_path: str, metadata: Dict, stats: Dict,
                     contour: List = None, saved_png: str = None,
                     saved_json: str = None, roi_description: str = None):
    pname = str(metadata.get("患者姓名", metadata.get("PatientName", "Unknown")))
    pid_val = str(metadata.get("患者ID", metadata.get("PatientID", "")))
    sdate = str(metadata.get("检查日期", metadata.get("StudyDate", "")))
    sdesc = str(metadata.get("检查描述", metadata.get("StudyDescription", "")))
    ser_desc = str(metadata.get("系列描述", metadata.get("SeriesDescription", "")))
    parsed = parse_stats_text(stats)

    full_meta = extract_dicom_meta(file_path)
    return insert_roi(
        patient_name=pname,
        patient_id=pid_val,
        patient_sex=full_meta.get("patient_sex", ""),
        patient_birth_date=full_meta.get("patient_birth_date", ""),
        patient_age=full_meta.get("patient_age", ""),
        file_path=file_path,
        method="sam_interactive",
        source="magic_seg_interactive",
        metadata=metadata,
        stats=stats,
        contour=contour,
        area_mm2=parsed.get("面积(mm²)"),
        mean_hu=parsed.get("均值"),
        std_hu=parsed.get("标准差"),
        min_hu=parsed.get("最小值"),
        max_hu=parsed.get("最大值"),
        median_hu=parsed.get("中位数"),
        pixel_count=int(parsed.get("像素数", 0)) if parsed.get("像素数") else None,
        saved_png=saved_png or "",
        saved_json=saved_json or "",
        roi_description=roi_description or "",
        study_date=sdate,
        study_time=full_meta.get("study_time", ""),
        study_description=sdesc,
        series_description=ser_desc,
        series_number=full_meta.get("series_number", ""),
        manufacturer=full_meta.get("manufacturer", ""),
        manufacturer_model=full_meta.get("manufacturer_model", ""),
        magnetic_field_strength=full_meta.get("magnetic_field_strength", ""),
        body_part=full_meta.get("body_part", ""),
        protocol_name=full_meta.get("protocol_name", ""),
        slice_thickness=full_meta.get("slice_thickness", ""),
        pixel_spacing=full_meta.get("pixel_spacing", ""),
    )


def add_rois_from_detection(file_path: str, method: str, rois: List[Dict]):
    meta = extract_dicom_meta(file_path)
    pname = meta.get("patient_name", "Unknown")
    for roi in rois:
        stats = roi.get("statistics", {})
        insert_roi(
            patient_name=pname,
            patient_id=meta.get("patient_id", ""),
            patient_sex=meta.get("patient_sex", ""),
            patient_birth_date=meta.get("patient_birth_date", ""),
            patient_age=meta.get("patient_age", ""),
            file_path=file_path,
            method=method,
            source=roi.get("source", f"roi_{method}"),
            metadata=meta,
            stats=stats,
            contour=roi.get("contour", []),
            area_mm2=roi.get("area_mm2"),
            mean_hu=stats.get("mean"),
            std_hu=stats.get("std"),
            min_hu=stats.get("min"),
            max_hu=stats.get("max"),
            median_hu=None,
            pixel_count=stats.get("pixel_count") or roi.get("area"),
            study_date=meta.get("study_date", ""),
            study_time=meta.get("study_time", ""),
            study_description=meta.get("study_description", ""),
            series_description=meta.get("series_description", ""),
            series_number=meta.get("series_number", ""),
            manufacturer=meta.get("manufacturer", ""),
            manufacturer_model=meta.get("manufacturer_model", ""),
            magnetic_field_strength=meta.get("magnetic_field_strength", ""),
            body_part=meta.get("body_part", ""),
            protocol_name=meta.get("protocol_name", ""),
            slice_thickness=meta.get("slice_thickness", ""),
            pixel_spacing=meta.get("pixel_spacing", ""),
        )


def add_ocr_blocks(file_path: str, blocks: List[Dict]):
    """Store all OCR text from one image as a single merged record."""
    if not blocks:
        return
    meta = extract_dicom_meta(file_path)
    pname = meta.get("patient_name", "Unknown")
    full_text = "\n".join(b.get("text", "") for b in blocks)
    all_floats = []
    for b in blocks:
        all_floats.extend(b.get("floats", []))
    ocr_stats = {
        "OCR识别完整文本": full_text,
        "OCR块数": len(blocks),
        "浮点数": ", ".join(str(f) for f in all_floats),
    }
    insert_roi(
        patient_name=pname, patient_id=meta.get("patient_id", ""),
        file_path=file_path, method="ocr", source="roi_ocr",
        metadata=meta, stats=ocr_stats,
        area_mm2=None, pixel_count=len(blocks),
        study_date=meta.get("study_date", ""),
        study_time=meta.get("study_time", ""),
        study_description=meta.get("study_description", ""),
        series_description=meta.get("series_description", ""),
        manufacturer=meta.get("manufacturer", ""),
        manufacturer_model=meta.get("manufacturer_model", ""),
        body_part=meta.get("body_part", ""),
        protocol_name=meta.get("protocol_name", ""),
        modality=meta.get("modality", ""),
    )


def add_merged_roi(file_path: str, method: str, rois: List[Dict]):
    """Merge all ROIs from one image into a single DB record."""
    if not rois:
        return
    meta = extract_dicom_meta(file_path)
    pname = meta.get("patient_name", "Unknown")
    total_count = len(rois)
    areas = [r.get('area', 0) for r in rois]
    mm2s = [r.get('area_mm2', 0) for r in rois if r.get('area_mm2')]
    circs = [r.get('circularity', 0) for r in rois]
    means = [r.get('statistics', {}).get('mean', 0) for r in rois]
    stds = [r.get('statistics', {}).get('std', 0) for r in rois]
    unique_areas = sorted(set(int(a) for a in areas))
    area_str = ', '.join(str(a) for a in unique_areas[:10])

    merged_stats = {
        "ROI总数": total_count,
        "唯一面积(px)": area_str,
        "总物理面积(mm²)": f"{sum(mm2s):.1f}",
        "均值圆形度": f"{sum(circs)/total_count:.3f}" if total_count > 0 else "0",
        "均值信号": f"{sum(means)/total_count:.1f}" if total_count > 0 else "0",
        "均值SD": f"{sum(stds)/total_count:.1f}" if total_count > 0 else "0",
    }
    insert_roi(
        patient_name=pname,
        patient_id=meta.get("patient_id", ""),
        patient_sex=meta.get("patient_sex", ""),
        patient_birth_date=meta.get("patient_birth_date", ""),
        patient_age=meta.get("patient_age", ""),
        file_path=file_path,
        method=method,
        source="roi_merged",
        metadata=meta,
        stats=merged_stats,
        area_mm2=sum(mm2s) if mm2s else None,
        mean_hu=sum(means)/total_count if total_count > 0 else None,
        std_hu=sum(stds)/total_count if total_count > 0 else None,
        pixel_count=total_count,
        study_date=meta.get("study_date", ""),
        study_time=meta.get("study_time", ""),
        study_description=meta.get("study_description", ""),
        series_description=meta.get("series_description", ""),
        series_number=meta.get("series_number", ""),
        manufacturer=meta.get("manufacturer", ""),
        manufacturer_model=meta.get("manufacturer_model", ""),
        magnetic_field_strength=meta.get("magnetic_field_strength", ""),
        body_part=meta.get("body_part", ""),
        protocol_name=meta.get("protocol_name", ""),
        slice_thickness=meta.get("slice_thickness", ""),
        pixel_spacing=meta.get("pixel_spacing", ""),
        modality=meta.get("modality", ""),
        kvp=meta.get("kvp", ""),
        xray_tube_current=meta.get("xray_tube_current", ""),
        exposure_time=meta.get("exposure_time", ""),
        convolution_kernel=meta.get("convolution_kernel", ""),
        reconstruction_diameter=meta.get("reconstruction_diameter", ""),
        ctdivol=meta.get("ctdivol", ""),
        dlp=meta.get("dlp", ""),
        spiral_pitch_factor=meta.get("spiral_pitch_factor", ""),
        gantry_detector_tilt=meta.get("gantry_detector_tilt", ""),
        table_height=meta.get("table_height", ""),
    )


# Auto-init on import
init_db()
