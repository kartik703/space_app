@echo off
title Space Intelligence Platform - Ultimate Automation
color 0B

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                🚀 SPACE INTELLIGENCE PLATFORM 🚀              ║
echo ║                     ONE-CLICK LAUNCHER                        ║
echo ║                                                              ║
echo ║  ✅ Automatic Environment Setup                               ║
echo ║  🔄 Real-time Data Pipeline                                   ║
echo ║  📊 Professional Dashboard                                    ║
echo ║  🤖 Full Automation                                           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Change to script directory
cd /d "%~dp0"

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Installing Python...
    echo 📥 Please install Python from: https://python.org/downloads
    echo ⚠️  Make sure to check "Add Python to PATH" during installation
    pause
    start https://python.org/downloads
    exit /b 1
)

echo ✅ Python found!
echo 🚀 Starting Ultimate Space Intelligence Platform...
echo.

REM Run the ultimate launcher
python ultimate_launcher.py

if errorlevel 1 (
    echo.
    echo ❌ Application encountered an error
    echo 🔧 Checking for issues...
    echo.
    echo 📋 Troubleshooting steps:
    echo    1. Check internet connection
    echo    2. Ensure Python 3.8+ is installed
    echo    3. Run as administrator if needed
    echo.
) else (
    echo.
    echo ✅ Application stopped successfully
)

echo.
echo 📚 For help, check the documentation:
echo    - README.md
echo    - DEPLOYMENT.md
echo    - TROUBLESHOOTING.md
echo.
pause