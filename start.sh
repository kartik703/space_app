#!/bin/bash
# 🚀 Space Intelligence Platform - Linux/Mac Launcher

echo "🚀 Space Intelligence Platform - Automated Launcher"
echo "==================================================="

# Check Python installation
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python is not installed or not in PATH"
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python found: $($PYTHON_CMD --version)"

# Change to script directory
cd "$(dirname "$0")"

echo "🔄 Starting automated system..."

# Run the automation script
if $PYTHON_CMD autostart.py start; then
    echo "✅ Space Intelligence Platform started successfully!"
    echo "🌐 Open your browser to: http://localhost:8501"
    
    # Option to open browser automatically on macOS/Linux
    if command -v xdg-open &> /dev/null; then
        read -p "Open browser automatically? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            xdg-open http://localhost:8501
        fi
    elif command -v open &> /dev/null; then
        read -p "Open browser automatically? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            open http://localhost:8501
        fi
    fi
else
    echo "❌ Failed to start Space Intelligence Platform"
    echo "Check the logs for more information"
    exit 1
fi