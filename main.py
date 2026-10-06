import os
import sys

# 必须在导入 numpy/torch/opencv 之前设置，避免 OMP 运行时冲突
# （否则 torch 无法初始化 CUDA，SAM 会回退 CPU 推理导致 CPU 占用近 100%）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from PyCt6 import CApplication, set_appearance_mode

set_appearance_mode("light")

from ui.main_window import MainWindow


def main():
    app = CApplication()

    mcp_port = None
    if "--mcp" in sys.argv:
        mcp_port = 7810
        for i, a in enumerate(sys.argv):
            if a == "--mcp-port" and i + 1 < len(sys.argv):
                try:
                    mcp_port = int(sys.argv[i + 1])
                except ValueError:
                    pass
    elif "--mcp-port" in sys.argv:
        for i, a in enumerate(sys.argv):
            if a == "--mcp-port" and i + 1 < len(sys.argv):
                try:
                    mcp_port = int(sys.argv[i + 1])
                except ValueError:
                    pass

    window = MainWindow()
    window.show()

    if mcp_port is not None:
        try:
            from mcp_dicom_tool import config
            from mcp_dicom_tool.gui_bridge import start_bridge
            config.ensure_dirs()
            bridge = start_bridge(window, port=mcp_port)
            window._mcp_bridge_port = bridge.port
            print(f"[mcp] pid={os.getpid()} bridge started on 127.0.0.1:{bridge.port}", flush=True)
        except Exception as e:
            print(f"[mcp] bridge start failed: {e}", flush=True)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
