"""
Production API System - RESTful API for Fusion AI Space Risk Platform
FastAPI-based enterprise-grade API with authentication, rate limiting, and comprehensive endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import logging
import json
import hashlib
import hmac
import time
import io
import csv
import base64
from contextlib import asynccontextmanager

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our AI systems
from .fusion_ai_system import SpaceRiskFusionAI, FusionPrediction
from .risk_engine import AdvancedRiskEngine, RiskAlert, DecisionRecommendation
from .advanced_space_cv import AdvancedSpaceCV
from .time_series_forecaster import SpaceWeatherForecaster
from .orbital_intelligence import OrbitalIntelligence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global AI systems (will be initialized on startup)
fusion_ai: Optional[SpaceRiskFusionAI] = None
risk_engine: Optional[AdvancedRiskEngine] = None

# Pydantic Models for API
class RiskAnalysisRequest(BaseModel):
    """Request model for risk analysis"""
    image_path: Optional[str] = Field(None, description="Path to image for CV analysis")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    include_forecasting: bool = Field(True, description="Include time-series forecasting")
    include_orbital: bool = Field(True, description="Include orbital intelligence")
    priority: str = Field("normal", regex="^(low|normal|high|urgent)$")
    client_id: Optional[str] = Field(None, description="Client identifier")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if v is not None:
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError('Invalid base64 image data')
        return v

class RiskAnalysisResponse(BaseModel):
    """Response model for risk analysis"""
    analysis_id: str
    timestamp: datetime
    overall_risk_score: float
    risk_category: str
    confidence_level: float
    cv_analysis: Dict[str, Any]
    time_series_forecast: Dict[str, Any]
    orbital_assessment: Dict[str, Any]
    financial_impact: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time_ms: float

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    image_paths: List[str] = Field(..., description="List of image paths")
    analysis_params: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = Field(None, description="Webhook URL for results")
    batch_name: Optional[str] = Field(None, description="Batch identifier")

class AlertsResponse(BaseModel):
    """Response model for alerts"""
    alerts: List[Dict[str, Any]]
    total_count: int
    critical_count: int
    high_count: int
    last_updated: datetime

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]
    uptime_seconds: float

class WebhookPayload(BaseModel):
    """Webhook payload model"""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    signature: str

# Authentication
class APIKeyAuth:
    """Simple API key authentication"""
    
    def __init__(self):
        # In production, these would be stored securely
        self.api_keys = {
            "test_key_123": {"client_id": "test_client", "rate_limit": "100/hour"},
            "prod_key_456": {"client_id": "prod_client", "rate_limit": "1000/hour"},
            "enterprise_key_789": {"client_id": "enterprise_client", "rate_limit": "10000/hour"}
        }
    
    def verify_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify API key"""
        api_key = credentials.credentials
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return self.api_keys[api_key]

# Global instances
security = HTTPBearer()
auth = APIKeyAuth()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global fusion_ai, risk_engine
    
    try:
        logger.info("Initializing AI systems...")
        fusion_ai = SpaceRiskFusionAI()
        risk_engine = AdvancedRiskEngine()
        risk_engine.start_real_time_monitoring()
        logger.info("AI systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI systems: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        if risk_engine:
            risk_engine.stop_real_time_monitoring()
        logger.info("AI systems shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Space Risk Fusion AI API",
    description="Enterprise-grade API for space risk assessment and prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Store for tracking requests and analytics
request_analytics = {}
start_time = time.time()

@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    """Middleware for request analytics"""
    start_time_req = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time_req
    
    # Track analytics
    endpoint = request.url.path
    if endpoint not in request_analytics:
        request_analytics[endpoint] = {"count": 0, "total_time": 0}
    
    request_analytics[endpoint]["count"] += 1
    request_analytics[endpoint]["total_time"] += process_time
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health and Status Endpoints

@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    current_time = time.time()
    
    # Check service status
    services_status = {
        "fusion_ai": "healthy" if fusion_ai else "unhealthy",
        "risk_engine": "healthy" if risk_engine else "unhealthy",
        "api": "healthy"
    }
    
    overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        services=services_status,
        uptime_seconds=current_time - start_time
    )

@app.get("/status")
@limiter.limit("10/minute")
async def status_check(request: Request):
    """Detailed status check"""
    return {
        "api_status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints_analytics": request_analytics,
        "active_connections": "monitoring_enabled" if risk_engine else "monitoring_disabled"
    }

# Core Analysis Endpoints

@app.post("/api/v1/analyze", response_model=RiskAnalysisResponse)
@limiter.limit("50/hour")
async def analyze_risk(
    request: Request,
    analysis_request: RiskAnalysisRequest,
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Comprehensive risk analysis endpoint"""
    start_time_analysis = time.time()
    
    try:
        if not fusion_ai:
            raise HTTPException(status_code=503, detail="Fusion AI system not available")
        
        # Handle image data
        image_path = None
        if analysis_request.image_data:
            # Decode base64 image and save temporarily
            image_bytes = base64.b64decode(analysis_request.image_data)
            temp_path = f"temp_image_{int(time.time())}.jpg"
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            image_path = temp_path
        elif analysis_request.image_path:
            image_path = analysis_request.image_path
        
        # Perform analysis
        prediction = fusion_ai.comprehensive_analysis(
            image_path=image_path,
            include_forecasting=analysis_request.include_forecasting,
            include_orbital=analysis_request.include_orbital
        )
        
        # Clean up temporary file
        if analysis_request.image_data and image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception:
                pass
        
        processing_time = (time.time() - start_time_analysis) * 1000
        
        return RiskAnalysisResponse(
            analysis_id=f"analysis_{int(time.time())}_{hash(str(prediction.timestamp))}",
            timestamp=prediction.timestamp,
            overall_risk_score=prediction.overall_risk_score,
            risk_category=prediction.risk_category,
            confidence_level=prediction.confidence_level,
            cv_analysis=prediction.cv_analysis,
            time_series_forecast=prediction.time_series_forecast,
            orbital_assessment=prediction.orbital_assessment,
            financial_impact=prediction.financial_impact,
            alerts=[alert.__dict__ if hasattr(alert, '__dict__') else alert for alert in prediction.alerts],
            recommendations=prediction.recommendations,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/batch-analyze")
@limiter.limit("5/hour")
async def batch_analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchAnalysisRequest,
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Batch analysis endpoint"""
    
    batch_id = f"batch_{int(time.time())}_{hash(str(batch_request.image_paths))}"
    
    # Start background processing
    background_tasks.add_task(
        process_batch_analysis,
        batch_id,
        batch_request.image_paths,
        batch_request.analysis_params,
        batch_request.callback_url,
        api_key_data["client_id"]
    )
    
    return {
        "batch_id": batch_id,
        "status": "accepted",
        "image_count": len(batch_request.image_paths),
        "estimated_completion": (datetime.now() + timedelta(minutes=len(batch_request.image_paths) * 2)).isoformat()
    }

async def process_batch_analysis(batch_id: str, image_paths: List[str], 
                               params: Dict[str, Any], callback_url: Optional[str],
                               client_id: str):
    """Background task for batch processing"""
    try:
        results = []
        
        for image_path in image_paths:
            try:
                prediction = fusion_ai.comprehensive_analysis(
                    image_path=image_path,
                    include_forecasting=params.get("include_forecasting", True),
                    include_orbital=params.get("include_orbital", True)
                )
                
                results.append({
                    "image_path": image_path,
                    "analysis": prediction.__dict__ if hasattr(prediction, '__dict__') else str(prediction),
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Send webhook if provided
        if callback_url:
            await send_webhook(callback_url, {
                "event_type": "batch_analysis_complete",
                "batch_id": batch_id,
                "client_id": client_id,
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"Batch analysis {batch_id} completed: {len(results)} items processed")
        
    except Exception as e:
        logger.error(f"Batch analysis {batch_id} failed: {e}")

# Risk Management Endpoints

@app.get("/api/v1/alerts", response_model=AlertsResponse)
@limiter.limit("100/hour")
async def get_alerts(
    request: Request,
    risk_level: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Get active alerts"""
    try:
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk engine not available")
        
        alerts = risk_engine.get_active_alerts()
        
        # Filter alerts
        if risk_level:
            alerts = [a for a in alerts if a.risk_level.value == risk_level]
        if category:
            alerts = [a for a in alerts if a.category.value == category]
        
        # Limit results
        alerts = alerts[:limit]
        
        # Count by priority
        critical_count = len([a for a in alerts if a.risk_level.value == "critical"])
        high_count = len([a for a in alerts if a.risk_level.value == "high"])
        
        return AlertsResponse(
            alerts=[{
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "risk_level": alert.risk_level.value,
                "category": alert.category.value,
                "priority": alert.priority.value,
                "title": alert.title,
                "description": alert.description,
                "affected_systems": alert.affected_systems,
                "recommended_actions": alert.recommended_actions,
                "resolved": alert.resolved
            } for alert in alerts],
            total_count=len(alerts),
            critical_count=critical_count,
            high_count=high_count,
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.post("/api/v1/alerts/{alert_id}/resolve")
@limiter.limit("100/hour")
async def resolve_alert(
    request: Request,
    alert_id: str,
    resolution_notes: str = "",
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Resolve an alert"""
    try:
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk engine not available")
        
        risk_engine.resolve_alert(alert_id, resolution_notes)
        
        return {
            "alert_id": alert_id,
            "status": "resolved",
            "timestamp": datetime.now().isoformat(),
            "resolved_by": api_key_data["client_id"]
        }
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@app.get("/api/v1/risk-score")
@limiter.limit("200/hour")
async def get_risk_score(
    request: Request,
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Get current composite risk score"""
    try:
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk engine not available")
        
        composite_risk = risk_engine.calculate_composite_risk_score()
        return composite_risk
        
    except Exception as e:
        logger.error(f"Error getting risk score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk score: {str(e)}")

@app.get("/api/v1/dashboard-data")
@limiter.limit("60/hour")
async def get_dashboard_data(
    request: Request,
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Get comprehensive dashboard data"""
    try:
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk engine not available")
        
        dashboard_data = risk_engine.export_risk_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

# Data Export Endpoints

@app.get("/api/v1/export/alerts")
@limiter.limit("10/hour")
async def export_alerts(
    request: Request,
    format: str = "json",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    api_key_data: dict = Depends(auth.verify_api_key)
):
    """Export alerts data"""
    try:
        alerts = risk_engine.get_active_alerts() if risk_engine else []
        
        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Alert ID", "Timestamp", "Risk Level", "Category", "Title", "Description"])
            
            for alert in alerts:
                writer.writerow([
                    alert.alert_id,
                    alert.timestamp.isoformat(),
                    alert.risk_level.value,
                    alert.category.value,
                    alert.title,
                    alert.description
                ])
            
            response = StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=alerts_export.csv"}
            )
            return response
        
        else:  # JSON format
            return {
                "export_timestamp": datetime.now().isoformat(),
                "export_format": "json",
                "alerts": [{
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "risk_level": alert.risk_level.value,
                    "category": alert.category.value,
                    "title": alert.title,
                    "description": alert.description
                } for alert in alerts]
            }
        
    except Exception as e:
        logger.error(f"Error exporting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Webhook functionality
async def send_webhook(url: str, payload: Dict[str, Any]):
    """Send webhook notification"""
    try:
        import aiohttp
        
        # Add signature for security
        secret = "webhook_secret_key"  # Should be configurable
        signature = hmac.new(
            secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        
        webhook_payload = WebhookPayload(
            event_type=payload["event_type"],
            timestamp=datetime.now(),
            data=payload,
            signature=signature
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=webhook_payload.dict(),
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info(f"Webhook sent successfully to {url}")
                else:
                    logger.warning(f"Webhook failed: {response.status} - {await response.text()}")
                    
    except Exception as e:
        logger.error(f"Error sending webhook: {e}")

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Space Risk Fusion AI API",
        version="1.0.0",
        description="Enterprise-grade API for space risk assessment and prediction using fusion AI",
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )