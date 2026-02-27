# AstroGuard — AI-Powered Space Debris Collision Risk Intelligence

**Aeroo Space AI Competition 2026**

AstroGuard is a real-time AI platform that detects collision risks between active satellites and space debris. It combines industry-standard orbital mechanics (SGP4) with machine learning to assess danger levels and recommend actions — served through a REST API and an interactive web dashboard.

---

## The Problem

There are over **27,000 tracked objects** in Earth orbit, plus millions of untracked fragments moving at 7–15 km/s. A single unmitigated collision can generate thousands of new debris pieces, triggering a cascade that makes entire orbital shells unusable for decades (Kessler Syndrome). Satellite operators need affordable, real-time, AI-enhanced conjunction risk tools — and today they largely don't have them.

---

## How It Works

```
Live TLE Data (Celestrak / NASA)
          │
          ▼
  SGP4 Orbital Propagation
  (position & velocity every 5 min, 24h ahead)
          │
          ▼
  Orbital Regime Pre-filter
  (altitude ±150 km, inclination ±25°)
          │
          ▼
  Closest Approach Detection
  (miss distance, relative velocity, TCA)
          │
          ├─────────────────────┐
          ▼                     ▼
  Chan Collision           Feature Engineering
  Probability (Pc)         [miss_dist, vel, alt,
  (physics baseline)        RCS, Pc, KE factor]
          │                     │
          └──────────┬──────────┘
                     ▼
         Random Forest Classifier
         → Risk Level: LOW / MEDIUM / HIGH / CRITICAL
                     │
         Isolation Forest
         → Anomaly flag for unusual patterns
                     │
                     ▼
         REST API + Web Dashboard
```

---

## Risk Levels

| Level    | Collision Probability | Recommended Action               |
|----------|-----------------------|----------------------------------|
| LOW      | Pc < 1×10⁻⁵           | Routine monitoring               |
| MEDIUM   | Pc ≥ 1×10⁻⁵           | Pre-alert satellite operator     |
| HIGH     | Pc ≥ 1×10⁻⁴           | Maneuver assessment required     |
| CRITICAL | Pc ≥ 1×10⁻³           | Execute emergency avoidance now  |

---

## Quick Start

**Requirements:** Python 3.9+, internet connection optional (works offline with demo data)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI demo

The fastest way to see the full AI pipeline in action — no server needed:

```bash
python demo.py
```

This will:
- Train the Random Forest + Isolation Forest models (~5 sec)
- Load orbital data (embedded demo TLEs)
- Run conjunction analysis with SGP4 propagation
- Print a full risk report in the terminal

To fetch live data from Celestrak and look 48 hours ahead:

```bash
python demo.py --live --hours 48
```

### 3. Run the web dashboard

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** — the dashboard auto-initializes, loads data, runs analysis, and displays results. Interactive orbit ground tracks, risk distribution charts, and a conjunction event table with click-through detail panels are all included.

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: **http://localhost:8000/api/docs**

| Method | Endpoint                  | Description                                      |
|--------|---------------------------|--------------------------------------------------|
| GET    | `/api/status`             | Health check and system status                   |
| GET    | `/api/statistics`         | Dashboard KPIs (object counts, risk summary)     |
| GET    | `/api/conjunctions`       | Conjunction events, filterable by risk level     |
| GET    | `/api/conjunctions/{id}`  | Detail for a single event                        |
| POST   | `/api/analyze`            | Analyze any satellite by providing its TLE       |
| GET    | `/api/satellites`         | List all tracked active satellites               |
| GET    | `/api/debris`             | List all tracked debris objects                  |
| GET    | `/api/model/features`     | AI model feature importances (explainability)    |
| GET    | `/api/visualize/orbits`   | Orbit ground track data for visualization        |
| POST   | `/api/refresh`            | Trigger fresh data fetch + analysis (background) |

### Example — analyze your own satellite

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
space-ai-task/
├── app/
│   ├── main.py            FastAPI application (API server + startup logic)
│   ├── models.py          Pydantic data models
│   ├── data_fetcher.py    Celestrak TLE fetcher with embedded demo fallback
│   ├── orbital.py         SGP4 propagation, altitude/element extraction
│   ├── analyzer.py        Conjunction detection pipeline
│   └── risk_model.py      AI risk model (Random Forest + Isolation Forest)
├── frontend/
│   └── index.html         Interactive web dashboard (Plotly + vanilla JS)
├── demo.py                CLI demo — full pipeline without a running server
├── requirements.txt       Python dependencies
├── README.md              This file
├── technical_docs.md      Architecture and algorithm documentation
└── pitch_deck.md          Investor pitch deck (15 slides)
```

---

## AI Components

### Random Forest Classifier
- Trained on **6,000 synthetic samples** generated using Chan's collision probability formula
- **8 input features**: miss distance, relative velocity, altitude, object sizes (RCS), position uncertainty, pre-computed Pc, kinetic energy factor
- **4 output classes**: LOW / MEDIUM / HIGH / CRITICAL
- Validation accuracy: ~99% on held-out set
- Uses physics-based hard floor — if Chan formula says HIGH, the model cannot output LOW

### Isolation Forest Anomaly Detector
- Trained on the same feature space
- Flags conjunctions outside the normal distribution as anomalies (5% contamination rate)
- Catches unusual orbital geometries that warrant independent human verification

### Data Sources
- **Celestrak TLE catalog** — live orbital data for 27,000+ objects (`celestrak.org`)
- **Embedded demo TLEs** — 24 real historical TLEs bundled for offline use

---

## Environment Variables

| Variable           | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `ASTROGUARD_DEMO`  | `false` | Skip Celestrak fetch, use embedded data  |

---

## Dependencies

| Package         | Version  | Purpose                        |
|----------------|----------|--------------------------------|
| `fastapi`       | ≥0.109   | REST API framework             |
| `uvicorn`       | ≥0.27    | ASGI server                    |
| `sgp4`          | ≥2.22    | SGP4/SDP4 orbital propagation  |
| `scikit-learn`  | ≥1.4     | Random Forest + Isolation Forest |
| `numpy`         | ≥1.26    | Numerical computing            |
| `pandas`        | ≥2.1     | Data handling                  |
| `plotly`        | ≥5.18    | Interactive charts             |
| `requests`      | ≥2.31    | Celestrak API calls            |
| `pydantic`      | ≥2.5     | Data validation                |

---

## Documentation

- `technical_docs.md` — full architecture, algorithm descriptions, API schema, performance characteristics
- `pitch_deck.md` — 15-slide investor presentation: problem, solution, market, business model, roadmap

---

## License

MIT
