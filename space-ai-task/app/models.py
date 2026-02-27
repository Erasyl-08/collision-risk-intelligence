from pydantic import BaseModel, Field
from typing import Optional
from enum import IntEnum


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}[self.value]

    @property
    def color(self) -> str:
        return {0: "#00ff88", 1: "#ffcc00", 2: "#ff6b35", 3: "#ff3355"}[self.value]


class SpaceObject(BaseModel):
    norad_id: int
    name: str
    tle_line1: str
    tle_line2: str
    object_type: str = "SATELLITE"  # SATELLITE, DEBRIS, ROCKET_BODY
    altitude_km: float = 0.0
    inclination_deg: float = 0.0
    period_min: float = 0.0
    rcs_m2: float = 1.0  # Radar cross-section estimate


class ConjunctionEvent(BaseModel):
    id: str
    object1_id: int
    object1_name: str
    object2_id: int
    object2_name: str
    object2_type: str
    tca_unix: float  # Time of Closest Approach (Unix timestamp)
    tca_iso: str
    miss_distance_km: float
    relative_velocity_km_s: float
    collision_probability: float
    risk_level: int
    risk_label: str
    risk_color: str
    altitude_km: float
    recommendation: str
    is_anomaly: bool = False


class RiskAssessment(BaseModel):
    risk_level: int
    risk_label: str
    risk_color: str
    collision_probability: float
    confidence: float
    features_used: dict
    recommendation: str
    explanation: str


class DashboardStats(BaseModel):
    total_objects_tracked: int
    active_satellites: int
    debris_objects: int
    high_risk_conjunctions: int
    critical_conjunctions: int
    total_conjunctions_24h: int
    anomalies_detected: int
    last_updated_iso: str
    model_accuracy: float


class AnalyzeRequest(BaseModel):
    tle_line1: str = Field(..., description="TLE Line 1 of the satellite to analyze")
    tle_line2: str = Field(..., description="TLE Line 2 of the satellite to analyze")
    name: str = Field(default="CUSTOM SAT", description="Satellite name")
    hours_ahead: int = Field(default=24, ge=1, le=72, description="Hours to look ahead")


class APIStatus(BaseModel):
    status: str
    version: str
    data_source: str
    objects_loaded: int
    model_trained: bool
    uptime_seconds: float
