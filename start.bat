@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала создайте окружение и установите зависимости:
    echo py -m venv .venv
    echo .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py
pause
