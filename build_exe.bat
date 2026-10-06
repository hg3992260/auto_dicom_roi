@echo off
REM Build standalone .exe for DICOM Analysis Tool
REM Requires: pip install pyinstaller

REM 优先使用 PATH 中的 python；也可先激活虚拟环境或手动指定
if not defined PYTHON_EXE set "PYTHON_EXE=python"

where %PYTHON_EXE% >nul 2>nul
if errorlevel 1 (
    echo ERROR: "%PYTHON_EXE%" not found in PATH.
    echo         Activate your virtualenv, or set PYTHON_EXE to a full path.
    pause
    exit /b 1
)

echo === Cleaning old builds ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo === Building EXE ===
"%PYTHON_EXE%" -m PyInstaller pyinstaller.spec --clean --noconfirm

if exist models (
    mkdir dist\models 2>nul
    copy models\*.pth dist\models\ >nul
)

echo === Done! ===
echo Output: dist\DICOM_Analysis_Tool.exe
echo Place SAM models in: dist\models\
pause
