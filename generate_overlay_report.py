"""生成 Overlay 分析 HTML 报告 — 含SAM种子点分割"""
import os, sys, base64, json, time
import numpy as np, cv2, pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_engine import RoiEngine
from ocr_engine import OcrEngine
from sam_engine import SamEngine

BASE_DIR = os.environ.get("DICOM_ROI_DATA_DIR", "")
if not BASE_DIR:
    _d = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    BASE_DIR = _d if os.path.isdir(_d) else ""

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "overlay_report.html")

engine = RoiEngine()
ocr = OcrEngine()
sam = SamEngine()


def _init_sam():
    if not sam.available:
        print("SAM unavailable, skipping SAM segmentation")
        return False
    if sam.checkpoint_path and sam.load_model():
        print(f"SAM loaded: {sam.model_type} on {sam.device}")
        return True
    return False


def img_to_b64(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode()


def norm_u8(pixels):
    pixels = pixels.astype(np.float32)
    if pixels.ndim == 3:
        pixels = pixels[:, :, 0]
    mn, mx = pixels.min(), pixels.max()
    return ((pixels - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(pixels, dtype=np.uint8)


def build_roi_table(rois):
    if not rois:
        return '<span class="no-roi">无有效ROI</span>'

    total = len(rois)
    areas = [r.get('area', 0) for r in rois]
    mm2s = [r.get('area_mm2', 0) for r in rois]
    circs = [r.get('circularity', 0) for r in rois]
    means = [r.get('statistics', {}).get('mean', 0) for r in rois]
    stds = [r.get('statistics', {}).get('std', 0) for r in rois]

    unique_areas = sorted(set(int(a) for a in areas))
    area_str = ', '.join(str(a) for a in unique_areas[:8])
    if len(unique_areas) > 8:
        area_str += f' ... (+{len(unique_areas)-8})'

    return (f'<span class="roi-summary">ROI: {total}个 | '
            f'面积: {area_str} px | '
            f'总物理面积: {sum(mm2s):.1f} mm² | '
            f'均值圆形度: {sum(circs)/total:.3f} | '
            f'均值信号: {sum(means)/total:.1f} ± {sum(stds)/total:.1f}</span>')


def build_block(patient, files_data):
    rows = ''
    for fd in files_data:
        tbl = build_roi_table(fd['rois'])
        sam_img_html = ''
        if fd.get('b64_sam'):
            sam_img_html = f'<div class="sam"><span>SAM分割</span><img src="data:image/png;base64,{fd["b64_sam"]}"/></div>'
        rows += f'''
        <div class="file-block">
            <div class="fh">
                <b>{fd['fname']}</b>
                <span>{fd['dims']}</span>
                <span>Overlay: {fd['px']} px · ROI: {len(fd['rois'])} · OCR: {len(fd.get("ocr_blocks", []))} blocks{fd.get("sam_info", "")}</span>
                <span>{fd['meta']}</span>
            </div>
            <div class="img-row">
                <div><span>原始</span><img src="data:image/png;base64,{fd['b64_orig']}"/></div>
                <div><span>Overlay</span><img src="data:image/png;base64,{fd['b64_ovl']}"/></div>
                <div><span>叠加</span><img src="data:image/png;base64,{fd['b64_comp']}"/></div>
                {sam_img_html}
            </div>
            <div class="roi-div">{tbl}</div>
            {fd.get('ocr_tbl', '')}
        </div>'''
    return f'''
    <div class="patient">
        <h2>{patient} ({len(files_data)} 文件)</h2>
        {rows}
    </div>'''


OVERLAY_SHAPE_CLOSED = "closed_round"
OVERLAY_SHAPE_CURVE = "continuous_curve"
OVERLAY_SHAPE_LINE = "discrete_line"


def classify_overlay_roi(roi):
    """Classify overlay ROI into 封闭圆形/连续曲线/离散直线 based on circularity + aspect ratio."""
    circ = roi.get('circularity', 0)
    bb = roi.get('bbox', {})
    w, h = bb.get('width', 1) or 1, bb.get('height', 1) or 1
    aspect = max(w / h, h / w) if w > 0 and h > 0 else 99
    area = roi.get('area', 0)

    if circ > 0.55 and aspect < 2.5 and area > 60:
        return OVERLAY_SHAPE_CLOSED
    if aspect > 10 or circ < 0.08 or area < 25:
        return OVERLAY_SHAPE_LINE
    return OVERLAY_SHAPE_CURVE


def _sam_segment_file(orig_gray, orig_rgb, rois):
    """Use overlay ROI centroids as SAM seed points, using only closed round ROIs."""
    if sam.predictor is None or not rois:
        return None, ""

    closed_rois = [r for r in rois if classify_overlay_roi(r) == OVERLAY_SHAPE_CLOSED]
    is_fallback = False
    if not closed_rois:
        closed_rois = sorted(
            [r for r in rois if classify_overlay_roi(r) == OVERLAY_SHAPE_CURVE],
            key=lambda r: r.get('area', 0), reverse=True)[:3]
        is_fallback = True
    if not closed_rois:
        closed_rois = rois[:2]

    areas = [r.get('area', 0) for r in closed_rois if r.get('centroid') and r.get('bbox')]
    if not areas:
        return None, ""
    median_area = float(np.median(areas)) if len(areas) > 1 else areas[0]
    min_area = max(median_area * 0.6, 40)

    centroids = []
    for r in closed_rois:
        cent = r.get('centroid')
        if cent and r.get('area', 0) >= min_area:
            centroids.append((int(cent['x']), int(cent['y'])))
    if not centroids:
        centroids.append((int(closed_rois[0]['centroid']['x']), int(closed_rois[0]['centroid']['y'])))
    if len(centroids) > 6:
        centroids = centroids[:6]

    try:
        sam.set_image(orig_rgb)
    except Exception:
        return None, " | SAM set_image失败"

    points = np.array(centroids, dtype=np.float32)
    labels = np.ones(len(centroids), dtype=np.int32)

    try:
        masks, scores, _ = sam.predict_mask(points, labels)
        best_idx = scores.argmax()
        sam_mask = masks[best_idx]
    except Exception:
        return None, " | SAM预测失败"

    highlight = orig_rgb.copy()
    highlight[sam_mask] = [255, 0, 127]
    alpha = 0.45
    highlight = cv2.addWeighted(orig_rgb, 1.0, highlight, alpha, 0)

    for cx, cy in centroids:
        cv2.circle(highlight, (cx, cy), 4, (255, 255, 0), -1)

    n_closed = len([r for r in rois if classify_overlay_roi(r) == OVERLAY_SHAPE_CLOSED])
    n_curve = len([r for r in rois if classify_overlay_roi(r) == OVERLAY_SHAPE_CURVE])
    n_line = len([r for r in rois if classify_overlay_roi(r) == OVERLAY_SHAPE_LINE])
    fb_tag = "(fallback)" if is_fallback else ""
    info = f" · SAM:{int(sam_mask.sum())}px | seeds:{len(centroids)}{fb_tag} | score:{scores[best_idx]:.3f} | ●{n_closed} ⌒{n_curve} ─{n_line}"
    return img_to_b64(cv2.cvtColor(highlight, cv2.COLOR_RGB2BGR)), info


def process_file(fp, fn):
    try:
        ds = pydicom.dcmread(fp, force=True)
    except Exception:
        return None

    mask = engine.extract_overlay_mask(fp)
    if mask is None or mask.sum() == 0:
        return None

    h, w = mask.shape

    try:
        px = ds.pixel_array
        if px.ndim == 3:
            px = px[:, :, 0]
        orig = norm_u8(px)
    except Exception:
        return None

    if orig.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (orig.shape[1], orig.shape[0])).astype(bool)

    ovl = np.zeros(orig.shape, dtype=np.uint8)
    ovl[mask] = 255

    comp = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    comp[mask] = [0, 255, 0]

    rois = engine.detect(fp, method='overlay', params={'min_area': 1})

    ocr_img = engine.render_overlay_for_ocr(fp)
    ocr_blocks = ocr.extract(ocr_img) if ocr_img is not None else []
    ocr_tbl = ''
    if ocr_blocks:
        ocr_rows = ''
        for b in ocr_blocks:
            f_str = ', '.join(str(f) for f in b.get('floats', [])[:5])
            ocr_rows += f'<tr><td>{b["text"]}</td><td>{b["confidence"]:.0f}%</td><td>{f_str}</td></tr>'
        ocr_tbl = f'<div class="ocr-div"><span class="ocr-hdr">OCR 识别 ({len(ocr_blocks)} blocks)</span><table><thead><tr><th>文本</th><th>置信度</th><th>浮点数</th></tr></thead><tbody>{ocr_rows}</tbody></table></div>'

    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    b64_sam, sam_info = _sam_segment_file(orig, orig_rgb, rois)

    mod = str(getattr(ds, 'Modality', '')).strip() or '-'
    ser = str(getattr(ds, 'SeriesDescription', '')).strip()[:40]

    return {
        'fname': fn,
        'dims': f'{orig.shape[1]}×{orig.shape[0]}',
        'px': int(mask.sum()),
        'meta': f'{mod} · {ser}',
        'rois': rois,
        'ocr_blocks': ocr_blocks,
        'ocr_tbl': ocr_tbl,
        'b64_orig': img_to_b64(orig),
        'b64_ovl': img_to_b64(ovl),
        'b64_comp': img_to_b64(comp),
        'b64_sam': b64_sam,
        'sam_info': sam_info,
    }


def main():
    t0 = time.time()
    sam_ok = _init_sam()
    if not sam_ok:
        print("WARNING: SAM not loaded, report will not include SAM segmentations")

    if not BASE_DIR or not os.path.isdir(BASE_DIR):
        print("ERROR: 未设置数据目录。请设置环境变量 DICOM_ROI_DATA_DIR，"
              "或把数据放在 <repo>/../data/ 下。")
        return

    patient_blocks = []
    patients = sorted(d for d in os.listdir(BASE_DIR)
                      if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith('.'))

    n_total = 0
    n_overlay = 0
    n_sam = 0

    for patient in patients:
        pp = os.path.join(BASE_DIR, patient, 'DICOM')
        if not os.path.exists(pp):
            continue
        file_data = []
        for root, dirs, files in os.walk(pp):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                if os.path.getsize(fp) < 100 or fn.endswith(('.png', '.jpg', '.log', '.txt', '.py', '.json')) or fn == 'DICOMDIR':
                    continue
                n_total += 1
                fd = process_file(fp, fn)
                if fd:
                    if fd.get('b64_sam'):
                        n_sam += 1
                    file_data.append(fd)
                    n_overlay += 1

        if file_data:
            patient_blocks.append(build_block(patient, file_data))

    html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DICOM Overlay + SAM 报告</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",sans-serif;background:#f5f7fa;color:#1e293b;padding:16px}}
h1{{text-align:center;color:#0b1e35;margin-bottom:4px}}
.summary{{text-align:center;color:#64748b;font-size:13px;margin-bottom:20px}}
.patient{{margin-bottom:28px}}
.patient h2{{background:#1e3a5f;color:#fff;padding:6px 14px;font-size:15px;border-radius:6px 6px 0 0}}
.file-block{{background:#fff;border:1px solid #e2e8f0;border-top:none;padding:10px}}
.file-block:last-child{{border-radius:0 0 6px 6px}}
.fh{{display:flex;gap:12px;align-items:center;font-size:12px;margin-bottom:8px;flex-wrap:wrap}}
.fh b{{color:#2563eb}}
.fh span{{color:#64748b;font-size:11px}}
.img-row{{display:flex;gap:10px;margin-bottom:8px;flex-wrap:wrap}}
.img-row>div{{flex:1;min-width:150px;text-align:center}}
.img-row>div.sam{{border:2px solid #7C3AED;border-radius:6px;padding:2px;background:#F5F3FF}}
.img-row span{{display:block;font-size:10px;font-weight:600;color:#64748b;margin-bottom:3px}}
.img-row>div.sam span{{color:#7C3AED;font-size:11px;font-weight:700}}
.img-row img{{width:100%;border:1px solid #e2e8f0;border-radius:4px}}
.roi-div{{margin-top:6px}}
.roi-div table{{width:100%;border-collapse:collapse;font-size:11px}}
.roi-div th{{background:#1e3a5f;color:#fff;padding:3px 5px;text-align:left;font-size:10px}}
.roi-div td{{padding:2px 5px;border-bottom:1px solid #e2e8f0}}
.roi-div tr:nth-child(even){{background:#f8fafc}}
.no-roi{{color:#94a3b8;font-size:11px;font-style:italic}}
.roi-summary{{display:block;font-size:11px;color:#059669;font-weight:600;padding:4px 0}}
.ocr-div{{margin-top:6px}}
.ocr-hdr{{display:block;font-size:11px;font-weight:bold;color:#059669;margin-bottom:3px}}
.ocr-div table{{width:100%;border-collapse:collapse;font-size:11px}}
.ocr-div th{{background:#059669;color:#fff;padding:3px 5px;text-align:left;font-size:10px}}
.ocr-div td{{padding:2px 5px;border-bottom:1px solid #e2e8f0}}
.ocr-div tr:nth-child(even){{background:#f0fdf4}}
</style></head><body>
<h1>DICOM Overlay + SAM 种子点分割报告</h1>
<div class="summary">总文件: {n_total} · 含Overlay: {n_overlay} · SAM分割: {n_sam} · 患者: {len(patient_blocks)} · {time.strftime('%Y-%m-%d %H:%M:%S')} · {time.time()-t0:.1f}s</div>
{''.join(patient_blocks)}
</body></html>'''

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Done: {n_total} files → {n_overlay} with overlay · {n_sam} with SAM ({len(patient_blocks)} patients) · {time.time()-t0:.1f}s')
    print(f'Output: {OUTPUT_HTML}')
    print(f'Size: {os.path.getsize(OUTPUT_HTML)/1024**2:.0f} MB')


if __name__ == '__main__':
    main()
