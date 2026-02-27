"""
AstroGuard API — Space Debris Collision Risk Intelligence Platform
FastAPI application serving the REST API and the web dashboard frontend.
"""
import time
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .data_fetcher import fetch_all_objects, fetch_demo_data, get_data_timestamp
from .analyzer import analyze_conjunctions, analyze_single_satellite, build_space_objects, generate_demo_conjunctions
from .risk_model import AstroGuardRiskModel
from .models import (
    ConjunctionEvent, DashboardStats, AnalyzeRequest,
    APIStatus, SpaceObject, RiskAssessment,
)
from .orbital import get_orbit_points_for_visualization, build_satellite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.start_time = time.time()
        self.risk_model = AstroGuardRiskModel()
        self.satellites: list = []
        self.debris: list = []
        self.conjunctions: List[ConjunctionEvent] = []
        self.last_analysis_time: Optional[str] = None
        self.data_source: str = "pending"
        self.is_ready: bool = False


state = AppState()

# ─── FastAPI Setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AstroGuard API",
    description="AI-powered space debris collision risk intelligence platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=== AstroGuard starting up ===")

    # Train AI model
    accuracy = state.risk_model.train()
    logger.info(f"Risk model ready (accuracy={accuracy:.3f})")

    # Load TLE data
    use_demo = os.getenv("ASTROGUARD_DEMO", "false").lower() == "true"
    if use_demo:
        state.satellites, state.debris = fetch_demo_data()
        state.data_source = "demo"
    else:
        state.satellites, state.debris = fetch_all_objects(
            max_satellites=200, max_debris=400
        )
        state.data_source = "celestrak" if len(state.satellites) > 15 else "demo"

    logger.info(f"Loaded {len(state.satellites)} satellites and {len(state.debris)} debris objects")

    # Run initial conjunction analysis
    logger.info("Running initial conjunction analysis...")
    state.conjunctions = analyze_conjunctions(
        satellites=state.satellites,
        debris_objects=state.debris,
        risk_model=state.risk_model,
        hours_ahead=24,
        max_conjunctions=50,
    )
    state.last_analysis_time = get_data_timestamp()
    state.is_ready = True

    high_risk = sum(1 for c in state.conjunctions if c.risk_level >= 2)
    logger.info(f"Analysis complete: {len(state.conjunctions)} conjunctions found ({high_risk} high/critical risk)")

    # Fall back to demo conjunctions if no real ones found (demo data mode)
    if len(state.conjunctions) == 0 and state.data_source == "demo":
        logger.info("No real conjunctions found — using simulated demo events for dashboard")
        state.conjunctions = generate_demo_conjunctions(state.risk_model)

    logger.info("=== AstroGuard ready ===")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serve the main web dashboard."""
    html_path = frontend_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AstroGuard</h1><p>Frontend not found. See /api/docs</p>")


@app.get("/api/status", response_model=APIStatus, tags=["System"])
async def get_status():
    """API health check and status."""
    return APIStatus(
        status="ready" if state.is_ready else "initializing",
        version="1.0.0",
        data_source=state.data_source,
        objects_loaded=len(state.satellites) + len(state.debris),
        model_trained=state.risk_model.is_trained,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/api/statistics", response_model=DashboardStats, tags=["Dashboard"])
async def get_statistics():
    """Get dashboard statistics for the main display."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing")

    high_risk = sum(1 for c in state.conjunctions if c.risk_level == 2)
    critical = sum(1 for c in state.conjunctions if c.risk_level == 3)
    anomalies = sum(1 for c in state.conjunctions if c.is_anomaly)

    return DashboardStats(
        total_objects_tracked=len(state.satellites) + len(state.debris),
        active_satellites=len(state.satellites),
        debris_objects=len(state.debris),
        high_risk_conjunctions=high_risk,
        critical_conjunctions=critical,
        total_conjunctions_24h=len(state.conjunctions),
        anomalies_detected=anomalies,
        last_updated_iso=state.last_analysis_time or datetime.now(timezone.utc).isoformat(),
        model_accuracy=round(state.risk_model.train_accuracy, 4),
    )


@app.get("/api/conjunctions", response_model=List[ConjunctionEvent], tags=["Analysis"])
async def get_conjunctions(
    min_risk: int = Query(default=0, ge=0, le=3, description="Minimum risk level (0=LOW, 3=CRITICAL)"),
    limit: int = Query(default=20, ge=1, le=50, description="Maximum number of results"),
    anomalies_only: bool = Query(default=False, description="Return only anomalous events"),
):
    """Get upcoming conjunction events sorted by risk level."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing, please wait")

    filtered = [c for c in state.conjunctions if c.risk_level >= min_risk]
    if anomalies_only:
        filtered = [c for c in filtered if c.is_anomaly]
    return filtered[:limit]


@app.get("/api/conjunctions/{event_id}", response_model=ConjunctionEvent, tags=["Analysis"])
async def get_conjunction_detail(event_id: str):
    """Get details for a specific conjunction event."""
    for c in state.conjunctions:
        if c.id == event_id:
            return c
    raise HTTPException(status_code=404, detail=f"Conjunction event '{event_id}' not found")


@app.post("/api/analyze", response_model=List[ConjunctionEvent], tags=["Analysis"])
async def analyze_custom_satellite(request: AnalyzeRequest):
    """
    Analyze a custom satellite (provided via TLE) against all tracked debris objects.
    Useful for satellite operators to check their own assets.
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing")
    if not state.debris:
        raise HTTPException(status_code=503, detail="No debris data loaded")

    events = analyze_single_satellite(
        sat_name=request.name,
        sat_l1=request.tle_line1,
        sat_l2=request.tle_line2,
        debris_objects=state.debris,
        risk_model=state.risk_model,
        hours_ahead=request.hours_ahead,
    )
    return events


@app.get("/api/satellites", response_model=List[SpaceObject], tags=["Data"])
async def get_satellites(limit: int = Query(default=50, ge=1, le=300)):
    """List tracked active satellites with orbital elements."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing")
    objects = build_space_objects(state.satellites[:limit], "SATELLITE")
    return objects


@app.get("/api/debris", response_model=List[SpaceObject], tags=["Data"])
async def get_debris(limit: int = Query(default=50, ge=1, le=600)):
    """List tracked debris objects with orbital elements."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing")
    objects = build_space_objects(state.debris[:limit], "DEBRIS")
    return objects


@app.get("/api/model/features", tags=["AI Model"])
async def get_feature_importances():
    """Get ML model feature importances (explainability endpoint)."""
    if not state.risk_model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    return {
        "feature_importances": state.risk_model.get_feature_importances(),
        "model_type": "RandomForestClassifier",
        "anomaly_model": "IsolationForest",
        "training_accuracy": round(state.risk_model.train_accuracy, 4),
        "n_risk_classes": 4,
        "risk_labels": {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"},
    }


@app.get("/api/visualize/orbits", tags=["Visualization"])
async def get_orbit_data(limit: int = Query(default=10, ge=1, le=30)):
    """
    Get orbit ground track data for visualization.
    Returns lat/lon points for the first N satellites and high-risk objects.
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing")

    result = []
    for name, l1, l2 in state.satellites[:limit]:
        sat = build_satellite(l1, l2)
        if sat is None:
            continue
        points = get_orbit_points_for_visualization(sat, n_points=60)
        result.append({"name": name, "type": "SATELLITE", "points": points})

    # Add high-risk debris
    high_risk_debris_names = {c.object2_name for c in state.conjunctions if c.risk_level >= 2}
    for name, l1, l2 in state.debris:
        if name in high_risk_debris_names and len(result) < limit + 5:
            sat = build_satellite(l1, l2)
            if sat is None:
                continue
            points = get_orbit_points_for_visualization(sat, n_points=60)
            result.append({"name": name, "type": "DEBRIS", "points": points})

    return {"orbits": result}


@app.post("/api/refresh", tags=["System"])
async def refresh_data(background_tasks: BackgroundTasks):
    """Trigger a fresh data fetch and conjunction analysis in the background."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing")

    async def _refresh():
        logger.info("Background refresh started")
        new_sats, new_debris = fetch_all_objects(max_satellites=200, max_debris=400)
        if new_sats:
            state.satellites = new_sats
        if new_debris:
            state.debris = new_debris
        state.conjunctions = analyze_conjunctions(
            satellites=state.satellites,
            debris_objects=state.debris,
            risk_model=state.risk_model,
            hours_ahead=24,
        )
        state.last_analysis_time = get_data_timestamp()
        logger.info("Background refresh complete")

    background_tasks.add_task(_refresh)
    return {"message": "Data refresh started in background", "current_data_age": state.last_analysis_time}
