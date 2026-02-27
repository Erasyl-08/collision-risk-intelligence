# 🛡 AstroGuard — AI-Powered Space Debris Collision Risk Intelligence

> **Aeroo Space AI Competition 2026 — MVP**  
> Real-time AI analysis of satellite conjunction events using live orbital data and machine learning.

---

## Problem

Over **27,000 tracked objects** and millions of untracked fragments orbit Earth. A single high-velocity collision (Kessler Syndrome trigger) could render entire orbital shells unusable for decades. Commercial satellite operators lack affordable, real-time, AI-enhanced conjunction risk assessment tools.

## Solution

AstroGuard is an AI-powered platform that:
- **Ingests live TLE data** from NASA/Celestrak orbital catalogs
- **Propagates orbits** using the industry-standard SGP4 algorithm
- **Detects conjunction events** (close approaches) over the next 24–72 hours
- **Assesses collision risk** with a two-layer AI model:
  - **Random Forest classifier** trained on physics-based synthetic data (Chan formula)
  - **Isolation Forest** for anomaly detection of unusual orbital patterns
- **Recommends actions**: from monitoring to emergency avoidance maneuvers
- **Serves results** via REST API and an interactive web dashboard

---

## Quick Start

### Prerequisites

- Python 3.9+
- Internet connection (optional — works offline with demo data)

### Installation

```bash
git clone https://github.com/your-team/astroguard
cd astroguard
pip install -r requirements.txt
```

### Run CLI Demo (fastest way to see the AI)

```bash
python demo.py
```

With live Celestrak data:

```bash
python demo.py --live --hours 48
```

### Run Web Dashboard

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

The dashboard will:
1. Train the AI model (~5 seconds)
2. Fetch TLE data from Celestrak (or use demo data)
3. Run conjunction analysis
4. Display the interactive dashboard

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/statistics` | Dashboard statistics |
| GET | `/api/conjunctions` | Conjunction events (filter by risk level) |
| GET | `/api/conjunctions/{id}` | Event detail |
| POST | `/api/analyze` | Analyze your satellite (TLE input) |
| GET | `/api/satellites` | List tracked satellites |
| GET | `/api/debris` | List tracked debris objects |
| GET | `/api/model/features` | AI model feature importances |
| GET | `/api/visualize/orbits` | Orbit ground track data |
| POST | `/api/refresh` | Trigger data refresh |
| GET | `/api/docs` | Swagger UI |

### Example: Analyze your satellite

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MY SATELLITE",
    "tle_line1": "1 25544U 98067A   24056.30463572  .00018151  00000-0  32822-3 0  9990",
    "tle_line2": "2 25544  51.6416 109.4938 0005770  25.8920  87.4420 15.50024695440375",
    "hours_ahead": 24
  }'
```

---

## Project Structure

```
astroguard/
├── app/
│   ├── main.py            # FastAPI REST API server
│   ├── models.py          # Pydantic data models
│   ├── data_fetcher.py    # Celestrak TLE data fetcher (with demo fallback)
│   ├── orbital.py         # SGP4 orbital propagation module
│   ├── analyzer.py        # Conjunction analysis pipeline
│   └── risk_model.py      # AI risk model (Random Forest + Isolation Forest)
├── frontend/
│   └── index.html         # Space-themed web dashboard (Plotly, vanilla JS)
├── demo.py                # CLI demo script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── technical_docs.md      # Technical documentation
└── pitch_deck.md          # Investor pitch deck
```

---

## AI Architecture

```
TLE Data (Celestrak)
        │
        ▼
SGP4 Propagation (sgp4 library)
        │
        ▼
Orbital Regime Pre-filter
  (altitude ±150km, inclination ±25°)
        │
        ▼
Trajectory Propagation (5-min steps, 24h window)
        │
        ▼
Closest Approach Detection (TCA, miss distance, relative velocity)
        │
        ├──────────────────────┐
        ▼                      ▼
Chan Collision          Feature Engineering
Probability (Pc)       [miss_dist, vel, alt, RCS, Pc, KE]
        │                      │
        └──────────┬───────────┘
                   ▼
         Random Forest Classifier
         (Risk: LOW/MEDIUM/HIGH/CRITICAL)
                   │
         Isolation Forest
         (Anomaly Detection)
                   │
                   ▼
         Risk Assessment + Recommendation
```

---

## Risk Levels

| Level | Collision Probability | Action |
|-------|----------------------|--------|
| LOW | Pc < 1×10⁻⁵ | Routine monitoring |
| MEDIUM | Pc ≥ 1×10⁻⁵ | Pre-alert operator |
| HIGH | Pc ≥ 1×10⁻⁴ | Maneuver assessment |
| CRITICAL | Pc ≥ 1×10⁻³ | Emergency maneuver |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASTROGUARD_DEMO` | `false` | Force demo data (skip Celestrak fetch) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `sgp4` | SGP4/SDP4 orbital propagation |
| `scikit-learn` | Random Forest + Isolation Forest |
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `plotly` | Interactive charts |
| `requests` | Celestrak API calls |
| `pydantic` | Data validation |

---

## Team

Aeroo Space AI Competition 2026 Team

---

## License

MIT License
