#!/usr/bin/env python3
"""
AstroGuard CLI Demo
Demonstrates the full AI pipeline without requiring a running server.

Usage:
    python demo.py
    python demo.py --live     # Fetch live TLE data from Celestrak
    python demo.py --hours 48 # Look ahead 48 hours
"""
import argparse
import sys
import time
from datetime import datetime, timezone

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

RISK_COLORS = {0: GREEN, 1: YELLOW, 2: f"{BOLD}{YELLOW}", 3: f"{BOLD}{RED}"}
RISK_LABELS = {0: "LOW     ", 1: "MEDIUM  ", 2: "HIGH    ", 3: "CRITICAL"}


def print_banner():
    print(f"""
{CYAN}{BOLD}
  █████╗ ███████╗████████╗██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
 ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
 ███████║███████╗   ██║   ██████╔╝██║   ██║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
 ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
 ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
 ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
{RESET}
{DIM}  AI-Powered Space Debris Collision Risk Intelligence Platform{RESET}
{DIM}  Aeroo Space AI Competition 2026 — MVP Demo{RESET}
""")


def print_section(title: str):
    width = 72
    print(f"\n{CYAN}{'─' * width}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{'─' * width}{RESET}")


def print_step(emoji: str, text: str, done: bool = False):
    status = f"{GREEN}✓{RESET}" if done else f"{CYAN}⟳{RESET}"
    print(f"  {status} {emoji} {text}")


def main():
    parser = argparse.ArgumentParser(description="AstroGuard CLI Demo")
    parser.add_argument("--live", action="store_true", help="Fetch live TLE data from Celestrak")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look ahead (default: 24)")
    parser.add_argument("--max-sats", type=int, default=150, help="Max satellites to analyze")
    parser.add_argument("--max-debris", type=int, default=300, help="Max debris objects to analyze")
    args = parser.parse_args()

    print_banner()

    # ── Step 1: Import ────────────────────────────────────────────────────────
    print_section("STEP 1 — Loading AI & Orbital Modules")
    print_step("📦", "Importing SGP4, scikit-learn, numpy...")
    try:
        from app.risk_model import AstroGuardRiskModel
        from app.data_fetcher import fetch_all_objects, fetch_demo_data
        from app.analyzer import analyze_conjunctions, build_space_objects
        from app.orbital import get_orbital_elements_from_tle
        print_step("📦", "All modules loaded successfully", done=True)
    except ImportError as e:
        print(f"\n{RED}Import error: {e}{RESET}")
        print(f"{YELLOW}Run: pip install -r requirements.txt{RESET}")
        sys.exit(1)

    # ── Step 2: Train Model ────────────────────────────────────────────────────
    print_section("STEP 2 — Training AI Risk Model")
    print_step("🧠", "Generating 6,000 synthetic training samples (Chan collision probability)...")
    t0 = time.time()
    model = AstroGuardRiskModel()
    accuracy = model.train()
    elapsed = time.time() - t0
    print_step("🧠", f"Random Forest classifier trained in {elapsed:.2f}s", done=True)
    print_step("🔍", f"Isolation Forest anomaly detector trained", done=True)
    print_step("📊", f"Validation accuracy: {GREEN}{BOLD}{accuracy*100:.1f}%{RESET}", done=True)

    fi = model.get_feature_importances()
    top_features = sorted(fi.items(), key=lambda x: -x[1])[:3]
    print(f"\n  {DIM}Top predictive features:{RESET}")
    for feat, imp in top_features:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {CYAN}{bar:<25}{RESET} {imp:.4f}")

    # ── Step 3: Load Data ──────────────────────────────────────────────────────
    print_section("STEP 3 — Loading Orbital Data")
    if args.live:
        print_step("🌐", "Fetching live TLE data from Celestrak.org...")
        satellites, debris = fetch_all_objects(
            max_satellites=args.max_sats, max_debris=args.max_debris
        )
        source = "Celestrak (live)" if len(satellites) > 15 else "Demo (Celestrak unavailable)"
    else:
        print_step("💾", "Loading embedded demo TLE data (use --live for Celestrak)...")
        satellites, debris = fetch_demo_data()
        source = "Embedded demo data"

    print_step("🛰 ", f"Active satellites loaded: {GREEN}{BOLD}{len(satellites)}{RESET}", done=True)
    print_step("🗑 ", f"Debris objects loaded:     {GREEN}{BOLD}{len(debris)}{RESET}", done=True)
    print_step("📡", f"Data source: {DIM}{source}{RESET}", done=True)

    # Show sample orbital elements
    if satellites:
        name, l1, l2 = satellites[0]
        elems = get_orbital_elements_from_tle(l1, l2)
        print(f"\n  {DIM}Sample: {name}{RESET}")
        print(f"    Altitude:    {elems.get('altitude_approx_km', 0):.0f} km")
        print(f"    Inclination: {elems.get('inclination_deg', 0):.2f}°")
        print(f"    Period:      {elems.get('period_min', 0):.1f} min")

    # ── Step 4: Conjunction Analysis ───────────────────────────────────────────
    print_section(f"STEP 4 — Conjunction Analysis (next {args.hours}h)")
    print_step("⚡", f"Pre-filtering by orbital regime (±150km altitude, ±25° inclination)...")
    print_step("📐", "Running SGP4 propagation (5-min time steps)...")
    print_step("🎯", "Finding closest approaches (TCA detection)...")

    t0 = time.time()
    events = analyze_conjunctions(
        satellites=satellites,
        debris_objects=debris,
        risk_model=model,
        hours_ahead=args.hours,
        max_conjunctions=30,
    )
    elapsed = time.time() - t0
    print_step("✅", f"Analysis complete in {elapsed:.2f}s — {len(events)} conjunction events found", done=True)

    # ── Step 5: Results ────────────────────────────────────────────────────────
    print_section("STEP 5 — Conjunction Event Report")

    if not events:
        print(f"\n  {YELLOW}No close conjunctions in demo data. Generating simulation events...{RESET}")
        from app.analyzer import generate_demo_conjunctions
        events = generate_demo_conjunctions(model)
        print(f"  {DIM}[SIMULATION MODE — use --live for real Celestrak data]{RESET}")

    if events:
        # Summary counts
        by_risk = {0: 0, 1: 0, 2: 0, 3: 0}
        anomalies = 0
        for e in events:
            by_risk[e.risk_level] += 1
            if e.is_anomaly:
                anomalies += 1

        print(f"\n  {'Risk Level':<12} {'Count':>6}  {'Bar'}")
        print(f"  {'─'*50}")
        for level in [3, 2, 1, 0]:
            cnt = by_risk[level]
            bar = "█" * min(cnt * 3, 30)
            color = RISK_COLORS[level]
            print(f"  {color}{RISK_LABELS[level]:<12}{RESET} {cnt:>6}  {color}{bar}{RESET}")
        print(f"\n  {YELLOW}Anomalies detected by AI: {anomalies}{RESET}")

        # Top events table
        print(f"\n  {BOLD}Top Conjunction Events:{RESET}")
        header = f"  {'#':<3} {'RISK':<10} {'SATELLITE':<22} {'DEBRIS':<22} {'MISS DIST':>10} {'REL VEL':>9} {'Pc':>10} {'ALT':>7}"
        print(f"\n{DIM}{header}{RESET}")
        print(f"  {'─'*110}")

        for i, event in enumerate(events[:15], 1):
            color = RISK_COLORS[event.risk_level]
            risk_label = RISK_LABELS[event.risk_level].strip()
            sat_name = event.object1_name[:20]
            deb_name = event.object2_name[:20]
            dist = f"{event.miss_distance_km:.3f} km"
            vel = f"{event.relative_velocity_km_s:.2f} km/s"
            pc_str = f"{event.collision_probability:.2e}"
            alt = f"{event.altitude_km:.0f} km"
            anomaly_flag = " ⚠" if event.is_anomaly else "  "

            print(
                f"  {i:<3} "
                f"{color}{risk_label:<10}{RESET} "
                f"{sat_name:<22} "
                f"{DIM}{deb_name:<22}{RESET} "
                f"{dist:>10} "
                f"{vel:>9} "
                f"{CYAN}{pc_str:>10}{RESET} "
                f"{alt:>7}"
                f"{YELLOW}{anomaly_flag}{RESET}"
            )

        # Show highest risk detail
        if events:
            top = events[0]
            print(f"\n{BOLD}  Highest Risk Event Detail:{RESET}")
            print(f"  {'─'*50}")
            print(f"  Satellite:          {top.object1_name}")
            print(f"  Debris Object:      {top.object2_name} ({top.object2_type})")
            print(f"  Miss Distance:      {top.miss_distance_km:.4f} km")
            print(f"  Relative Velocity:  {top.relative_velocity_km_s:.4f} km/s")
            print(f"  Collision Prob (Pc): {top.collision_probability:.3e}")
            print(f"  Altitude at TCA:    {top.altitude_km} km")
            tca_dt = datetime.fromtimestamp(top.tca_unix, tz=timezone.utc)
            print(f"  Time of Closest Approach: {tca_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            color = RISK_COLORS[top.risk_level]
            print(f"  AI Risk Assessment: {color}{BOLD}{RISK_LABELS[top.risk_level].strip()}{RESET}")
            if top.is_anomaly:
                print(f"  {YELLOW}⚠ Anomaly: Unusual conjunction pattern flagged by Isolation Forest{RESET}")
            print(f"\n  {CYAN}Recommendation: {top.recommendation}{RESET}")

    # ── Step 6: API Info ───────────────────────────────────────────────────────
    print_section("STEP 6 — Web Dashboard & API")
    print(f"""
  {BOLD}Start the full web dashboard:{RESET}

    {CYAN}uvicorn app.main:app --reload{RESET}

  Then open:  {GREEN}http://localhost:8000{RESET}

  {BOLD}REST API Endpoints:{RESET}
    GET  /api/statistics          Dashboard statistics
    GET  /api/conjunctions        Conjunction events (filterable)
    POST /api/analyze             Analyze your own satellite (TLE input)
    GET  /api/model/features      AI model explainability
    GET  /api/visualize/orbits    Orbit ground track data
    GET  /api/docs                Interactive Swagger documentation

  {DIM}For live data: set ASTROGUARD_DEMO=false (default) or use --live flag{RESET}
""")

    print(f"\n{GREEN}{BOLD}  AstroGuard demo complete!{RESET}\n")


if __name__ == "__main__":
    main()
