# 🚀 Space Intelligence Platform - PowerShell Launcher
param(
    [string]$Action = "start"
)

Write-Host "🚀 Space Intelligence Platform - Automated Launcher" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "🔄 Starting automated system..." -ForegroundColor Yellow

try {
    # Run the automation script
    $process = Start-Process -FilePath "python" -ArgumentList "autostart.py", $Action -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -eq 0) {
        Write-Host "✅ Space Intelligence Platform started successfully!" -ForegroundColor Green
        Write-Host "🌐 Open your browser to: http://localhost:8501" -ForegroundColor Cyan
        
        # Option to open browser automatically
        $openBrowser = Read-Host "Open browser automatically? (y/n)"
        if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
            Start-Process "http://localhost:8501"
        }
    }
    else {
        Write-Host "❌ Failed to start Space Intelligence Platform" -ForegroundColor Red
        Write-Host "Check the logs for more information" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ Error starting application: $_" -ForegroundColor Red
}

Read-Host "Press Enter to exit"