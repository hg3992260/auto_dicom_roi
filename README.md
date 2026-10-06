# DICOM Analysis Tool

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20%2B%20PyCt6-2E6DA4)](https://doc.qt.io/qtforpython/)
[![SAM](https://img.shields.io/badge/Segmentation-SAM%20%7C%20MedSAM%20%7C%20SAM--Med2D%2F3D-7C3AED)](https://github.com/facebookresearch/segment-anything)

**医学影像 ROI 标注与分析桌面工具**：PySide6 图形界面 + SAM 系列交互式分割 +
传统 CV ROI 检测引擎，并内置 **MCP 桥接层**，可由 AI Agent 读取 GUI 上的全部操作记录、
复现标注过程并驱动后续统计计算。

---

## 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [模型权重](#模型权重)
- [目录结构](#目录结构)
- [输出格式](#输出格式)
- [MCP 接口](#mcp-接口)
- [环境变量](#环境变量)
- [数据与隐私](#数据与隐私)
- [已知限制](#已知限制)

---

## 功能特性

### 1. DICOM 扫描与分组
- 递归扫描文件夹，按 `DICM` 魔数自动识别 DICOM 文件
- 按 `PatientID` → `StudyInstanceUID` → `SeriesInstanceUID` 三级自动分组
- 支持进度回调与拖放文件夹直接扫描

### 2. ROI 检测引擎（5 种算法）

| 方法 | 算法 | 说明 |
|------|------|------|
| `overlay` | DICOM Overlay Plane（6000 组） | 三级 fallback：pydicom 内置 → 手动位解析 → RT Structure Set |
| `otsu` | OTSU 阈值 + 形态学开运算 | 椭圆结构元 |
| `adaptive` | 自适应高斯阈值 + 形态学 | 可调 `block_size`、`c_value` |
| `watershed` | 分水岭算法 | 距离变换 + 标记 |
| `edge` | Canny 边缘检测 + 形态学 | — |

轮廓过滤支持**面积**、**长宽比**（默认 ≤5.0）、**圆形度**（默认 ≥0.1）。

```python
from roi_engine import RoiEngine
engine = RoiEngine()
rois = engine.detect(file_path, method="overlay", params={"min_area": 50})

# 直接取 Overlay 原始位图掩膜
mask = engine.extract_overlay_mask(file_path)   # -> np.ndarray (bool)
```

> **注意**：DICOM Overlay Plane 的位序有 MSB / LSB 两种解包约定。
> 本引擎使用 pydicom 的标准（LSB）解包——若自行实现，用错位序会导致掩膜与图像错位。

### 3. SAM 交互式分割
- 支持 **ViT-B / ViT-L / ViT-H** 三种量级（标准 SAM），并可选
  **MedSAM / SAM-Med2D / SAM-Med3D** 适配器（见 `model_adapters/`）
- 左键 = 前景点，右键 = 背景点，实时返回分割掩膜
- 自动 GPU / CPU fallback；模型单例，切换 checkpoint 自动卸载旧模型

```python
from sam_engine import SamEngine
engine = SamEngine()
engine.set_checkpoint(path)
engine.load_model(callback)
engine.set_image(rgb_image)
masks, scores, logits = engine.predict_mask(points, labels)
rois = engine.auto_segment(image)
```

### 4. 统一查看器（3 种模式）

| 模式 | 触发 | 显示内容 |
|------|------|---------|
| Viewing | 默认 | 灰度图 + 窗宽窗位 |
| Overlay | ROI 检测 | 灰度图 + 绿色掩膜叠加 |
| SAM | 切到「SAM分割」 | RGB 图 + 绿色掩膜 + 前景/背景点 |

窗宽窗位可实时调整，`↺` 按钮重置为 DICOM 默认值。

### 5. 统计汇总与报告
- `dicom_summary_db.py`：SQLite ROI 汇总库（患者 / 方法 / 面积 / 均值 / 标准差 / 扫描信息）
- `generate_overlay_report.py`：批量生成 Overlay + SAM 叠加 HTML 报告
- `ocr_engine.py`：识别烧录在图像上的测量文字（可选）

### 6. MCP 桥接层（`mcp_dicom_tool/`）
GUI 进程内嵌 **recorder**（SQLite + JSONL 双写）与 **TCP bridge**，
用户每次 SAM 交互 / ROI 保存 / 文件加载都会被持久化记录；
MCP server 以 stdio 运行，供 AI Agent 查询状态、截图、导出。

### 7. 模型适配器（`model_adapters/`）
`MedSAM`、`SAM-Med2D`、`SAM-Med3D` 的统一封装（含各自 `segment_anything` 源码）。

---

## 安装

```bash
git clone https://github.com/hg3992260/auto_dicom_roi.git
cd auto_dicom_roi

# 建议使用虚拟环境
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 覆盖 GUI、DICOM、图像处理与 SAM 推理所需依赖。
PyTorch 请按本机 CUDA 版本安装对应的官方 wheel。

---

## 快速开始

```bash
python main.py
```

### 操作流程

1. **扫描** —— 拖入 DICOM 根目录或点击浏览，按「扫描DICOM文件」
2. **浏览** —— 左侧树展开 Patient→Study→Series→File，右侧显示图像
3. **ROI 检测** —— 切到「ROI检测」标签，选方法（默认 `overlay`），点「开始ROI检测」
4. **SAM 分割** —— 切到「SAM分割」标签 → 选择 `.pth` → 「确认加载」
   → 图上左键标前景 / 右键标背景 → 「保存ROI」

### 带 MCP 桥启动

```bash
python main.py --mcp                 # 默认端口 7810
python main.py --mcp --mcp-port 7811
```

---

## 模型权重

模型权重**体积过大，不随仓库分发**。请自行下载后放入 `models/`：

| 文件 | 大小 | 说明 | 来源 |
|------|------|------|------|
| `sam_vit_b_01ec64.pth` | 358 MB | 标准 SAM ViT-B（推荐起点） | [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| `sam_vit_l_0b3195.pth` | 1.2 GB | 标准 SAM ViT-L | 同上 |
| `sam_vit_h_4b8939.pth` | 2.4 GB | 标准 SAM ViT-H（精度最高，显存需求大） | 同上 |
| `medsam_vit_b.pth` | 358 MB | MedSAM（医学影像微调） | [MedSAM](https://github.com/bowang-lab/MedSAM) |
| `sam_med2d_b.pth` | 2.4 GB | SAM-Med2D | [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) |
| `sam_med3d_turbo.pth` | 384 MB | SAM-Med3D（3D 体数据） | [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) |

```bash
mkdir models
# 将下载的权重放入 models/ 后在 GUI 中选择对应文件
```

---

## 目录结构

```
auto_dicom_roi/
├── main.py                     # 入口（CApplication + CMainWindow）
├── app_paths.py                # 运行期路径解析（打包 / 源码两态）
├── dicom_cluster.py            # DICOM 扫描 + Patient/Study/Series 分组
├── dicom_summary_db.py         # SQLite ROI 汇总库
├── roi_engine.py               # ROI 检测引擎（5 算法 + Overlay）
├── sam_engine.py               # SAM 模型单例（ViT-B/L/H）
├── ocr_engine.py               # 烧录文字 OCR
├── utils.py                    # 掩膜渲染 / JSON 保存等工具
├── generate_overlay_report.py  # 批量 HTML 报告
├── interactive_sam_viewer.py   # 独立 SAM 标注窗口
│
├── ui/                         # 界面层
│   ├── main_window.py          # 主窗口（三栏 + UnifiedViewer）
│   ├── dicom_tree.py           # Patient→Study→Series→File 树
│   ├── roi_panel.py            # ROI 检测面板
│   ├── sam_panel.py            # SAM 分割面板
│   └── summary_panel.py        # 汇总表面板
│
├── mcp_dicom_tool/             # MCP 桥接层
│   ├── server.py               # FastMCP 入口（stdio）
│   ├── gui_bridge.py           # GUI 进程内 TCP 桥
│   ├── recorder.py             # 事件记录（SQLite + JSONL）
│   ├── bridge_client.py        # 桥客户端
│   ├── config.py               # 路径 / 端口 / 解释器配置
│   └── tools/                  # lifecycle / inspect / capture / recording
│
├── model_adapters/             # 分割模型适配器
│   ├── medsam/
│   ├── sam_med2d/
│   └── sam_med3d/
│
├── logo/                       # 图标资源
├── requirements.txt
├── pyinstaller.spec            # 打包配置
├── build_exe.bat               # Windows 一键打包
├── 交接文档.md                  # 开发者交接文档
└── README.md
```

---

## 输出格式

### ROI 检测 JSON
```json
{
  "method": "overlay",
  "params": {"min_area": 50},
  "file_count": 6,
  "results": [{
    "file": "path/to/file",
    "rois": [{
      "roi_id": "roi_0",
      "bbox": {"x": 409, "y": 264, "width": 7, "height": 12},
      "area": 36.0,
      "circularity": 0.239,
      "contour": [[0, 0]],
      "centroid": {"x": 412, "y": 270},
      "source": "roi_overlay"
    }]
  }]
}
```

### SAM ROI JSON
```json
{
  "file": "path/to/file",
  "timestamp": "20260728_140000",
  "source": "interactive",
  "metadata": {"患者姓名": "ANON^PATIENT", "检查日期": "YYYYMMDD"},
  "roi_statistics": {
    "像素数": 1234,
    "面积(mm²)": "45.67",
    "均值": "128.34 HU",
    "标准差": "23.45 HU"
  }
}
```

---

## MCP 接口

启动带桥的 GUI 后，MCP server 以 stdio 提供以下工具组：

| 组 | 模块 | 用途 |
|----|------|------|
| `lifecycle` | `tools/lifecycle.py` | 启动 / 重启 / 关闭 GUI，健康检查 |
| `inspect` | `tools/inspect.py` | 查询事件、掩膜、ROI、状态、错误 |
| `capture` | `tools/capture.py` | 截图确认渲染结果 |
| `recording` | `tools/recording.py` | 导出全部记录为 JSONL |

典型调用顺序：

```
dcm_launch → 在 GUI 上做 SAM 交互 / 保存 ROI
          → dcm_query_events / dcm_list_masks / dcm_get_mask / dcm_wait_event
          → dcm_screenshot 截图确认
          → dcm_get_rois 取 ROI 汇总库用于统计计算
```

---

## 环境变量

| 变量 | 作用 | 默认 |
|------|------|------|
| `DICOM_ROI_DATA_DIR` | 默认 DICOM 根目录（`generate_overlay_report.py`、`interactive_sam_viewer.py` 使用） | 空（启动时手动选择） |
| `DICOM_TOOL_PYTHON` | GUI 解释器路径（MCP 桥拉起 GUI 时使用） | 当前解释器 `sys.executable` |
| `DICOM_TOOL_REPO` | 仓库根目录覆盖 | 自动推断 |
| `DICOM_TOOL_MCP_PORT` | MCP 桥端口 | `7810`（占用时向 7810–7910 探测） |
| `KMP_DUPLICATE_LIB_OK` | 规避 numpy/torch OpenMP 运行时冲突 | 由 `main.py` 设为 `TRUE` |

---

## 数据与隐私

本工具面向**临床科研**场景，处理的是受保护的医疗信息，请务必注意：

- 仓库**不包含**任何患者数据、DICOM 文件、ROI 数据库或导出的分析表格；
  `output/`、`mcp_records/`、`*.db`、`*.xlsx`、`*.csv`、`models/` 均已在 `.gitignore` 中排除。
- **不要**把含患者姓名、病历号、检查号、机构名称或原始文件路径的内容提交到公开仓库。
  SAM ROI JSON 的 `metadata` 会写入 DICOM 中的患者信息——导出分享前请先脱敏。
- 示例与文档中的病例标识均为虚构占位（如 `ANON^PATIENT`）。
- 实际使用请遵循所在机构的数据管理规定与伦理审批范围。

---

## 已知限制

1. **SAM 首次加载**约 10–30 s（取决于模型大小与硬件）；ViT-H 加载后约需 8 GB 以上显存。
2. **Overlay 识别**依赖 DICOM 中存在 6000 组 Overlay Plane（常见于后处理工作站的二次捕获图像）。
3. **二次捕获图像**的 `PixelSpacing` 需自行核实是否与显示矩阵匹配——若字段继承自源序列而未随缩放更新，
   由其换算的 mm² / mm 会失真。可在 1 px 比例尺与工作站标尺之间做交叉核对。
4. 无数据库服务依赖，全部结果直接落 JSON / PNG 文件。
5. MCP 桥仅监听 `127.0.0.1`，请勿绑定到外部网卡。

---

## 许可

本仓库包含来自第三方项目的代码（`model_adapters/` 下的
MedSAM、SAM-Med2D、SAM-Med3D 及其 `segment_anything` 源码），
分别遵循其原始许可（多为 Apache-2.0）。使用与再分发前请核对各上游许可条款。

本工具仅用于**科研与工程评估**，不构成医疗器械，不应用于临床诊断决策。
