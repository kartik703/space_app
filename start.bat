@echo off
echo 🚀 Space Intelligence Platform - Automated Launcher
echo ===================================================

echo 📋 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo 🔄 Starting automated system...

REM Change to script directory
cd /d "%~dp0"

REM Run the automation script
python autostart.py start

if errorlevel 1 (
    echo ❌ Failed to start Space Intelligence Platform
    echo Check the logs for more information
    pause
    exit /b 1
)

echo ✅ Space Intelligence Platform started successfully!
echo 🌐 Open your browser to: http://localhost:8501
pause