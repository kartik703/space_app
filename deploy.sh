#!/bin/bash

# 🚀 Space Intelligence Platform - Production Deployment Script
# This script handles complete deployment with monitoring and health checks

set -euo pipefail

# Configuration
APP_NAME="space-intelligence-platform"
DOCKER_IMAGE="$APP_NAME:latest"
CONTAINER_NAME="space-app-prod"
HEALTH_CHECK_TIMEOUT=300
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"
    
    # Check if Docker daemon is running
    docker info >/dev/null 2>&1 || error "Docker daemon is not running"
    
    log "✅ Prerequisites check passed"
}

# Create backup of current deployment
backup_current() {
    log "💾 Creating backup of current deployment..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup data directory if exists
    if [ -d "./data" ]; then
        cp -r ./data "$BACKUP_DIR/"
        log "✅ Data directory backed up"
    fi
    
    # Backup logs if exists
    if [ -d "./logs" ]; then
        cp -r ./logs "$BACKUP_DIR/"
        log "✅ Logs directory backed up"
    fi
    
    # Export current container if running
    if docker ps --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
        docker export "$CONTAINER_NAME" > "$BACKUP_DIR/container_backup.tar"
        log "✅ Container state backed up"
    fi
}

# Build the Docker image
build_image() {
    log "🏗️  Building Docker image..."
    
    # Build with build args for optimization
    docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from "$DOCKER_IMAGE" \
        -t "$DOCKER_IMAGE" \
        -f Dockerfile \
        . || error "Docker build failed"
    
    log "✅ Docker image built successfully"
}

# Stop existing containers
stop_existing() {
    log "🛑 Stopping existing containers..."
    
    if docker ps --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
        docker stop "$CONTAINER_NAME" || warn "Could not stop existing container"
        docker rm "$CONTAINER_NAME" || warn "Could not remove existing container"
        log "✅ Existing containers stopped"
    else
        log "ℹ️  No existing containers to stop"
    fi
}

# Deploy with Docker Compose
deploy() {
    log "🚀 Deploying Space Intelligence Platform..."
    
    # Pull latest images for dependencies
    docker-compose pull prometheus grafana || warn "Could not pull monitoring images"
    
    # Start the application stack
    docker-compose up -d || error "Docker Compose deployment failed"
    
    log "✅ Deployment started successfully"
}

# Health check
health_check() {
    log "🏥 Performing health checks..."
    
    local counter=0
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / 10))
    
    while [ $counter -lt $max_attempts ]; do
        if curl -f -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            log "✅ Application is healthy and responding"
            return 0
        fi
        
        counter=$((counter + 1))
        echo -n "."
        sleep 10
    done
    
    error "Health check failed after ${HEALTH_CHECK_TIMEOUT}s"
}

# Post-deployment verification
verify_deployment() {
    log "🔍 Verifying deployment..."
    
    # Check if containers are running
    if ! docker-compose ps | grep -q "Up"; then
        error "Some containers are not running"
    fi
    
    # Check application accessibility
    if ! curl -f -s http://localhost:8501 >/dev/null 2>&1; then
        error "Application is not accessible"
    fi
    
    # Check monitoring stack (if enabled)
    if docker-compose --profile monitoring ps | grep -q "Up"; then
        log "📊 Monitoring stack is running"
        log "   - Grafana: http://localhost:3000 (admin/space123)"
        log "   - Prometheus: http://localhost:9090"
    fi
    
    log "✅ Deployment verification passed"
}

# Display status and next steps
show_status() {
    log "🎉 Deployment completed successfully!"
    echo
    echo -e "${BLUE}🌟 Space Intelligence Platform is now running!${NC}"
    echo -e "${BLUE}   Application: http://localhost:8501${NC}"
    echo -e "${BLUE}   Health Check: http://localhost:8501/_stcore/health${NC}"
    echo
    echo -e "${YELLOW}📊 Optional Monitoring Stack:${NC}"
    echo -e "${YELLOW}   Start with: docker-compose --profile monitoring up -d${NC}"
    echo -e "${YELLOW}   Grafana: http://localhost:3000 (admin/space123)${NC}"
    echo -e "${YELLOW}   Prometheus: http://localhost:9090${NC}"
    echo
    echo -e "${GREEN}🔧 Management Commands:${NC}"
    echo -e "${GREEN}   View logs: docker-compose logs -f${NC}"
    echo -e "${GREEN}   Stop: docker-compose down${NC}"
    echo -e "${GREEN}   Update: ./deploy.sh${NC}"
    echo
    echo -e "${BLUE}📁 Backup created at: $BACKUP_DIR${NC}"
}

# Cleanup function for graceful exit
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed. Check logs for details."
        log "🔄 To rollback, run: docker-compose down && docker-compose up -d"
    fi
}

# Main deployment workflow
main() {
    trap cleanup EXIT
    
    log "🚀 Starting Space Intelligence Platform deployment..."
    
    check_prerequisites
    backup_current
    build_image
    stop_existing
    deploy
    health_check
    verify_deployment
    show_status
    
    log "✨ Deployment completed successfully!"
}

# Command line argument handling
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "🛑 Stopping Space Intelligence Platform..."
        docker-compose down
        log "✅ Application stopped"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        log "🧹 Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker image prune -f
        log "✅ Cleanup completed"
        ;;
    "backup")
        backup_current
        log "✅ Backup created at: $BACKUP_DIR"
        ;;
    "monitor")
        log "📊 Starting monitoring stack..."
        docker-compose --profile monitoring up -d
        log "✅ Monitoring stack started"
        ;;
    *)
        echo "Usage: $0 [deploy|stop|logs|status|clean|backup|monitor]"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy the application (default)"
        echo "  stop     - Stop all services"
        echo "  logs     - View live logs"
        echo "  status   - Show service status"
        echo "  clean    - Clean up Docker resources"
        echo "  backup   - Create backup of current state"
        echo "  monitor  - Start monitoring stack"
        exit 1
        ;;
esac