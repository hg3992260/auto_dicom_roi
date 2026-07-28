@echo off
REM Build standalone .exe for DICOM Analysis Tool
REM Requires: pip install pyinstaller

echo === Cleaning old builds ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo === Building EXE ===
pyinstaller pyinstaller.spec --clean --noconfirm

echo === Done! ===
echo Output: dist\DICOM_Analysis_Tool.exe
echo Place SAM models in: dist\models\
pause
