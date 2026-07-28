import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from PyCt6 import CApplication, set_appearance_mode

set_appearance_mode("light")

from ui.main_window import MainWindow


def main():
    app = CApplication()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
