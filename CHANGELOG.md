# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.1.0] — 2026-10-06

首次公开发布（源码）。

### 新增

- **MCP 桥接层**（`mcp_dicom_tool/`）
  GUI 进程内嵌事件记录器（SQLite + JSONL 双写）与 TCP 桥；MCP server 以 stdio 提供
  `lifecycle` / `inspect` / `capture` / `recording` 四组工具，可由 AI Agent 启动 GUI、
  读取全部操作记录、截图确认并导出 JSONL。
- **模型适配器**（`model_adapters/`）
  MedSAM / SAM-Med2D / SAM-Med3D 统一封装，提供一致的
  `load_model / set_image / predict / predict_boxes / auto_segment` 接口。
- **合成演示数据生成器**（`examples/make_demo_dicom.py`）
  生成完全合成的演示 DICOM（假患者 `DEMO^ANON`，附 6000 组 Overlay Plane），
  用于功能试用、截图复现与 CI 冒烟测试，不含任何真实数据。
- **界面截图**（`docs/screenshots/`）
  主界面 / ROI 检测 / 病人列表 / 查看器与掩膜 / ROI 面板，均基于上述合成数据。
- **第三方许可声明**
  `LICENSE-APACHE-2.0.txt` 与 `THIRD_PARTY_NOTICES.md`（MedSAM / SAM-Med2D / SAM-Med3D /
  TinyViT 的归属与许可）。
- **README.md 完整重写**
  功能特性、安装、快速开始、模型权重下载表、目录结构、输出格式、MCP 接口、
  环境变量、数据与隐私、已知限制。

### 变更

- 内部数据目录硬编码 → 环境变量 `DICOM_ROI_DATA_DIR`（含空值守卫）。
- 开发机解释器绝对路径 → 环境变量 `DICOM_TOOL_PYTHON` / `sys.executable`。
- `build_exe.bat` 的解释器检测改用 `where`，支持虚拟环境。
- `.gitignore` 补齐：模型权重（`*.pth/*.pt/*.bin`）、`output/`、`mcp_records/`、
  本地数据库与表格导出、内部研究脚本。
- 移除已提交的 `ui/__pycache__/*.pyc`（编译产物含绝对路径）。

### 已知问题

- **Windows EXE 的 CI 构建失败**：`build.yml` 的 pip 安装列表缺少 `torchvision`，
  而 `segment_anything` 依赖 `torchvision.transforms`。
  修复只需在该 workflow 增加一行 `pip install torchvision`；
  该文件属于 GitHub Actions，修改需要 `workflow` 权限范围。
  在当前源码树中，`requirements.txt` **已包含** `torchvision>=0.15`，
  本地按 README 安装即可正常构建。
- **`数据汇总` 标签切换无效**：`_switch_action(2)` 调用
  `action_stack.setCurrentIndex(2)`，但栈中只添加了 `roi_panel` 与 `sam_panel`
  两个页面（越界无效）。汇总内容实际由右侧 `page_stack` 承载。
- **`RoiEngine.METHODS` 仅暴露 `overlay` 与 `ocr`**：
  `otsu` / `adaptive` / `watershed` / `edge` 已实现为私有方法
  （`_detect_*`），但受白名单限制，GUI 中不可选。

### 说明

本仓库**不包含**任何患者数据、DICOM 文件、模型权重或研究产物。
详见 README 的「数据与隐私」一节。
