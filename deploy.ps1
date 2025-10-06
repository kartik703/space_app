# 🚀 Space Intelligence Platform - Windows PowerShell Deployment Script
# This script handles complete deployment with monitoring and health checks

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "stop", "logs", "status", "clean", "backup", "monitor")]
    [string]$Command = "deploy"
)

# Configuration
$AppName = "space-intelligence-platform"
$DockerImage = "${AppName}:latest"
$ContainerName = "space-app-prod"
$HealthCheckTimeout = 300
$BackupDir = ".\backups\$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Cyan = "Cyan"
}

# Logging functions
function Write-Log {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR: $Message" -ForegroundColor Red
    exit 1
}

# Check prerequisites
function Test-Prerequisites {
    Write-Log "🔍 Checking prerequisites..."
    
    # Check Docker
    try {
        docker --version | Out-Null
        docker info | Out-Null
    }
    catch {
        Write-Error "Docker is not installed or not running"
    }
    
    # Check Docker Compose
    try {
        docker-compose --version | Out-Null
    }
    catch {
        Write-Error "Docker Compose is not installed"
    }
    
    Write-Log "✅ Prerequisites check passed"
}

# Create backup of current deployment
function New-Backup {
    Write-Log "💾 Creating backup of current deployment..."
    
    if (!(Test-Path $BackupDir)) {
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    }
    
    # Backup data directory if exists
    if (Test-Path ".\data") {
        Copy-Item -Recurse -Path ".\data" -Destination $BackupDir
        Write-Log "✅ Data directory backed up"
    }
    
    # Backup logs if exists
    if (Test-Path ".\logs") {
        Copy-Item -Recurse -Path ".\logs" -Destination $BackupDir
        Write-Log "✅ Logs directory backed up"
    }
    
    # Export current container if running
    $runningContainer = docker ps --format "table {{.Names}}" | Select-String $ContainerName
    if ($runningContainer) {
        docker export $ContainerName > "$BackupDir\container_backup.tar"
        Write-Log "✅ Container state backed up"
    }
}

# Build the Docker image
function Build-Image {
    Write-Log "🏗️  Building Docker image..."
    
    $buildResult = docker build --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from $DockerImage -t $DockerImage -f Dockerfile .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
    }
    
    Write-Log "✅ Docker image built successfully"
}

# Stop existing containers
function Stop-Existing {
    Write-Log "🛑 Stopping existing containers..."
    
    $runningContainer = docker ps --format "table {{.Names}}" | Select-String $ContainerName
    if ($runningContainer) {
        docker stop $ContainerName
        docker rm $ContainerName
        Write-Log "✅ Existing containers stopped"
    } else {
        Write-Log "ℹ️  No existing containers to stop"
    }
}

# Deploy with Docker Compose
function Start-Deployment {
    Write-Log "🚀 Deploying Space Intelligence Platform..."
    
    # Pull latest images for dependencies
    try {
        docker-compose pull prometheus grafana
    }
    catch {
        Write-Warning "Could not pull monitoring images"
    }
    
    # Start the application stack
    docker-compose up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker Compose deployment failed"
    }
    
    Write-Log "✅ Deployment started successfully"
}

# Health check
function Test-Health {
    Write-Log "🏥 Performing health checks..."
    
    $counter = 0
    $maxAttempts = [math]::Floor($HealthCheckTimeout / 10)
    
    while ($counter -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Log "✅ Application is healthy and responding"
                return
            }
        }
        catch {
            # Continue trying
        }
        
        $counter++
        Write-Host "." -NoNewline
        Start-Sleep 10
    }
    
    Write-Error "Health check failed after ${HealthCheckTimeout}s"
}

# Post-deployment verification
function Test-Deployment {
    Write-Log "🔍 Verifying deployment..."
    
    # Check if containers are running
    $runningContainers = docker-compose ps | Select-String "Up"
    if (!$runningContainers) {
        Write-Error "Some containers are not running"
    }
    
    # Check application accessibility
    try {
        Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 10 | Out-Null
    }
    catch {
        Write-Error "Application is not accessible"
    }
    
    # Check monitoring stack (if enabled)
    $monitoringContainers = docker-compose --profile monitoring ps | Select-String "Up"
    if ($monitoringContainers) {
        Write-Log "📊 Monitoring stack is running"
        Write-Log "   - Grafana: http://localhost:3000 (admin/space123)"
        Write-Log "   - Prometheus: http://localhost:9090"
    }
    
    Write-Log "✅ Deployment verification passed"
}

# Display status and next steps
function Show-Status {
    Write-Log "🎉 Deployment completed successfully!"
    Write-Host ""
    Write-Host "🌟 Space Intelligence Platform is now running!" -ForegroundColor Blue
    Write-Host "   Application: http://localhost:8501" -ForegroundColor Blue
    Write-Host "   Health Check: http://localhost:8501/_stcore/health" -ForegroundColor Blue
    Write-Host ""
    Write-Host "📊 Optional Monitoring Stack:" -ForegroundColor Yellow
    Write-Host "   Start with: docker-compose --profile monitoring up -d" -ForegroundColor Yellow
    Write-Host "   Grafana: http://localhost:3000 (admin/space123)" -ForegroundColor Yellow
    Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 Management Commands:" -ForegroundColor Green
    Write-Host "   View logs: docker-compose logs -f" -ForegroundColor Green
    Write-Host "   Stop: docker-compose down" -ForegroundColor Green
    Write-Host "   Update: .\deploy.ps1" -ForegroundColor Green
    Write-Host ""
    Write-Host "📁 Backup created at: $BackupDir" -ForegroundColor Blue
}

# Main deployment workflow
function Start-MainDeployment {
    Write-Log "🚀 Starting Space Intelligence Platform deployment..."
    
    Test-Prerequisites
    New-Backup
    Build-Image
    Stop-Existing
    Start-Deployment
    Test-Health
    Test-Deployment
    Show-Status
    
    Write-Log "✨ Deployment completed successfully!"
}

# Command handling
switch ($Command) {
    "deploy" {
        Start-MainDeployment
    }
    "stop" {
        Write-Log "Stopping Space Intelligence Platform..."
        docker-compose down
        Write-Log "✅ Application stopped"
    }
    "logs" {
        docker-compose logs -f
    }
    "status" {
        docker-compose ps
    }
    "clean" {
        Write-Log "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker image prune -f
        Write-Log "✅ Cleanup completed"
    }
    "backup" {
        New-Backup
        Write-Log "✅ Backup created at: $BackupDir"
    }
    "monitor" {
        Write-Log "Starting monitoring stack..."
        docker-compose --profile monitoring up -d
        Write-Log "✅ Monitoring stack started"
    }
    default {
        Write-Host "Usage: .\deploy.ps1 [deploy|stop|logs|status|clean|backup|monitor]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  deploy   - Deploy the application (default)"
        Write-Host "  stop     - Stop all services"
        Write-Host "  logs     - View live logs"
        Write-Host "  status   - Show service status"
        Write-Host "  clean    - Clean up Docker resources"
        Write-Host "  backup   - Create backup of current state"
        Write-Host "  monitor  - Start monitoring stack"
    }
}