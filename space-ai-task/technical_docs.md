# AstroGuard — Technical Documentation

## 1. System Overview

AstroGuard is a real-time AI-powered space debris collision risk intelligence platform. It combines classical orbital mechanics (SGP4 propagation) with modern machine learning to detect and assess collision risks between active satellites and catalogued space debris.

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        ASTROGUARD PLATFORM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Data Layer  │    │ Physics Layer│    │    AI Layer      │  │
│  │              │    │              │    │                  │  │
│  │  Celestrak   │───▶│  SGP4        │───▶│  Random Forest   │  │
│  │  TLE API     │    │  Propagator  │    │  Classifier      │  │
│  │              │    │              │    │                  │  │
│  │  Demo Data   │    │  Conjunction │    │  Isolation       │  │
│  │  (fallback)  │    │  Detector    │    │  Forest Anomaly  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│          │                  │                     │             │
│          └──────────────────┴─────────────────────┘             │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │   FastAPI REST  │                          │
│                    │   API Server   │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │  Web Dashboard  │                          │
│                    │  (Plotly + JS)  │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI 0.109+ | REST API + async serving |
| ASGI Server | Uvicorn | Production-grade HTTP server |
| Orbital Mechanics | sgp4 2.22 | SGP4/SDP4 orbit propagation |
| ML Framework | scikit-learn 1.4+ | Random Forest, Isolation Forest |
| Numerical Computing | NumPy 1.26+ | Vector math, ECI coordinates |
| Data Processing | Pandas 2.1+ | TLE catalog management |
| Visualization | Plotly 5.18+ | Interactive charts |
| Data Validation | Pydantic 2.5+ | API models + validation |
| HTTP Client | Requests | Celestrak API calls |
| Frontend | HTML5 + CSS3 + Vanilla JS | Zero-dependency dashboard |

---

## 3. Data Sources

### 3.1 Celestrak TLE Catalog
- **URL**: `https://celestrak.org/CTLE/GP.php`
- **Format**: Two-Line Element (TLE) — NORAD standard
- **Groups fetched**:
  - `active` — all operational satellites (~9,000 objects)
  - `debris` — catalogued debris and rocket bodies (~20,000+ objects)
- **Update frequency**: Celestrak updates TLEs daily from US Space Force

### 3.2 TLE Format
```
SATELLITE NAME
1 NNNNNC YYDDDHHH.HHHHHHHH +.HHHHHHHH +HHHHH-H +HHHHH-H H NNNNN
2 NNNNN HHH.HHHH HHH.HHHH HHHHHHH HHH.HHHH HHH.HHHH HH.HHHHHHHHNNNNN
```
- Line 1: NORAD ID, international designator, epoch, drag terms
- Line 2: Inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion

### 3.3 Demo Data
For offline operation, 20 real TLEs (well-known satellites + debris objects) are embedded in `app/data_fetcher.py`.

---

## 4. Orbital Mechanics Module (`app/orbital.py`)

### 4.1 SGP4 Propagation
Uses the `sgp4` Python library (wrapping C++ Vallado implementation):

```python
from sgp4.api import Satrec, jday

sat = Satrec.twoline2rv(line1, line2)
jd, fr = jday(year, month, day, hour, minute, second)
error, position_km, velocity_km_s = sat.sgp4(jd, fr)
```

**Output**: Earth-Centered Inertial (ECI) coordinate frame
- Position vector `r` in km
- Velocity vector `v` in km/s
- Error code 0 = success

### 4.2 Altitude Computation
```
altitude_km = |r| - R_Earth
```
where `R_Earth = 6371.0 km`

### 4.3 Orbital Period
Derived from mean motion `n₀` (rad/min):
```
T = 1440 / (n₀ · 1440/(2π)) minutes
```

---

## 5. Conjunction Detection Algorithm (`app/analyzer.py`)

### 5.1 Pre-filtering (Orbital Regime)
Reduces O(N×M) pairs to a manageable set by only propagating pairs where:
- `|alt₁ - alt₂| < 150 km`
- `|inc₁ - inc₂| < 25°` (accounting for retrograde orbits)

### 5.2 Trajectory Propagation
For each candidate pair:
- Time step: **5 minutes** (balances accuracy vs speed)
- Window: **24 hours** (1440 steps / 5 = 288 time points)
- Both objects propagated to same time grid

### 5.3 Closest Approach Detection
For each common timestamp:
```
d(t) = |r₁(t) - r₂(t)|    (Euclidean distance in ECI)
```
The minimum `d` over the time window defines the **miss distance** at **Time of Closest Approach (TCA)**.

Only pairs with `miss_distance < 10 km` proceed to risk assessment.

### 5.4 Relative Velocity
```
v_rel = |v₁(TCA) - v₂(TCA)|    (km/s)
```

---

## 6. AI Risk Model (`app/risk_model.py`)

### 6.1 Chan Collision Probability Formula
Physics-based probability used as a feature and for training label generation:

```
Pc = (A_comb / (2π · σ_r²)) · exp(-d² / (2σ_r²))
```

Where:
- `A_comb = π · (r₁ + r₂)²` — combined hard-body cross-section
- `r₁, r₂` — object radii estimated from radar cross-section (RCS)
- `σ_r` — combined 1-sigma position uncertainty (100–500 m typical)
- `d` — miss distance (m)

### 6.2 Training Data Generation
6,000 synthetic samples generated using varied physics parameters:

| Feature | Distribution | Range |
|---------|-------------|-------|
| Miss distance | Exponential(μ=20) | 0.05 – 200 km |
| Relative velocity | Uniform | 0.1 – 15 km/s |
| Altitude | Uniform | 200 – 2000 km |
| Object 1 RCS | Exponential(μ=2) | 0.01 – 400 m² |
| Object 2 RCS | Exponential(μ=2) | 0.01 – 400 m² |
| Position uncertainty σ | Uniform | 50 – 500 m |

**Risk labels** assigned by Pc thresholds (NASA CARA guidelines):

| Level | Threshold | Int |
|-------|-----------|-----|
| LOW | Pc < 10⁻⁵ | 0 |
| MEDIUM | Pc ≥ 10⁻⁵ | 1 |
| HIGH | Pc ≥ 10⁻⁴ | 2 |
| CRITICAL | Pc ≥ 10⁻³ | 3 |

### 6.3 Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",  # handles class imbalance
    random_state=42,
)
```

**Input features (8)**:
1. `miss_distance_km`
2. `relative_velocity_km_s`
3. `altitude_km`
4. `object1_rcs_m2`
5. `object2_rcs_m2`
6. `position_sigma_m`
7. `collision_probability` (Chan formula, pre-computed)
8. `kinetic_energy_factor` (= min(v_rel/10, 1.0))

**Why ML over pure Chan formula?**
- Captures non-linear interactions (e.g., high-altitude + polar crossing = different risk profile)
- Can learn from historical collision data when available (future improvement)
- Provides confidence scores and feature importances for explainability
- Faster inference than iterative Monte Carlo methods

### 6.4 Isolation Forest Anomaly Detection
```python
IsolationForest(
    n_estimators=100,
    contamination=0.05,  # 5% of events flagged as anomalous
    random_state=42,
)
```

Detects conjunctions with unusual characteristics not covered by the training distribution:
- Unexpected orbital plane intersections
- Unusual object size/mass combinations
- Novel debris cloud patterns
- Events that warrant independent human verification

### 6.5 Model Performance
- Validation accuracy: ~94–97% (depends on class balance in split)
- Training time: < 5 seconds on modern hardware
- Inference time: < 1ms per conjunction event

---

## 7. REST API (`app/main.py`)

### 7.1 Application Lifecycle
1. `startup`: Train AI model → Fetch TLE data → Run conjunction analysis
2. `request`: Serve cached results (updated on background refresh)
3. `background`: `/api/refresh` triggers new data fetch and analysis

### 7.2 Key Endpoints

#### GET /api/conjunctions
```json
[
  {
    "id": "a3f1b2c4",
    "object1_name": "ISS (ZARYA)",
    "object2_name": "COSMOS 1408 DEB",
    "object2_type": "DEBRIS",
    "tca_iso": "2026-02-27T14:32:00+00:00",
    "miss_distance_km": 0.847,
    "relative_velocity_km_s": 7.23,
    "collision_probability": 3.2e-4,
    "risk_level": 2,
    "risk_label": "HIGH",
    "altitude_km": 421.5,
    "recommendation": "Maneuver assessment required.",
    "is_anomaly": false
  }
]
```

#### POST /api/analyze
Accepts custom TLE input for on-demand analysis of any satellite:
```json
{
  "name": "MY SAT",
  "tle_line1": "1 25544U ...",
  "tle_line2": "2 25544 ...",
  "hours_ahead": 24
}
```

---

## 8. Frontend Dashboard (`frontend/index.html`)

Single-file dashboard with zero external JS dependencies except Plotly CDN:

- **Stats row**: 7 real-time KPIs
- **Conjunction table**: Sortable, filterable by risk level, row click for detail modal
- **Risk distribution**: Animated bar chart by risk level
- **AI Feature Importance**: Horizontal bar chart (Plotly)
- **Orbit Ground Tracks**: Geo scatter plot (Plotly scattergeo)
- **Altitude Distribution**: Histogram of conjunction altitudes
- **Loading sequence**: Animated boot sequence reflecting actual backend initialization

---

## 9. Performance Characteristics

| Metric | Value |
|--------|-------|
| Objects analyzed (demo) | 15 sats × 6 debris |
| Objects analyzed (live) | 200 sats × 400 debris |
| Pre-filter reduction | ~80–95% pair reduction |
| Propagation step | 5 minutes |
| Analysis window | 24 hours |
| Total analysis time (demo) | < 5 seconds |
| Total analysis time (full) | 30–120 seconds |
| API response time (cached) | < 50ms |
| Model inference | < 1ms per event |

---

## 10. Known Limitations & Future Work

### Current Limitations
1. **5-minute time step** may miss very short close approaches (< 1 minute)
2. **Position uncertainty** is assumed fixed (200m); operational systems use covariance matrices
3. **No maneuver optimization**: AstroGuard identifies risk but doesn't compute optimal delta-V
4. **Single-thread propagation**: Could be parallelized for larger catalogs

### Roadmap
- **v1.1**: Parallel propagation with multiprocessing (10x speed)
- **v1.2**: Covariance matrix support for accurate Pc computation
- **v1.3**: Maneuver recommendation engine (optimal ΔV calculation)
- **v2.0**: GPT-4 integration for natural language risk reports
- **v2.1**: Historical collision data fine-tuning for ML model
- **v2.2**: Real-time TLE update subscription (Space-Track.org API)
- **v3.0**: Constellation-level fleet monitoring with custom alerting

---

## 11. Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest

# Run with demo data (no network required)
ASTROGUARD_DEMO=true pytest
```

---

## 12. Deployment

### Local
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (future)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud
Compatible with Railway, Render, Fly.io, AWS Lambda (with Mangum adapter), GCP Cloud Run.
