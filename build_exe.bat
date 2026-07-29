@echo off
REM Build standalone .exe for DICOM Analysis Tool
REM Requires: pip install pyinstaller

set "PYTHON_EXE=D:\python\envs\RSNA311\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: %PYTHON_EXE%
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
