"""
Conjunction Analysis Pipeline for AstroGuard.

Workflow:
1. Pre-filter candidate pairs by orbital regime similarity
2. Propagate filtered pairs over 24h using SGP4
3. Find closest approach for each pair
4. Score risk using ML model
5. Return ranked conjunction events
"""
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional

import numpy as np

from .orbital import (
    build_satellite,
    propagate_trajectory,
    find_closest_approach,
    get_orbital_elements_from_tle,
    estimate_rcs_from_type,
)
from .models import ConjunctionEvent, SpaceObject
from .risk_model import AstroGuardRiskModel

logger = logging.getLogger(__name__)

ORBITAL_REGIME_ALT_TOLERANCE_KM = 150.0
ORBITAL_REGIME_INC_TOLERANCE_DEG = 25.0
MAX_MISS_DISTANCE_KM = 10.0
PROPAGATION_STEP_MINUTES = 5.0


def objects_in_same_regime(elements1: dict, elements2: dict) -> bool:
    """Quick pre-filter: are two objects close enough in orbital regime to warrant full propagation?"""
    if not elements1 or not elements2:
        return False
    alt_diff = abs(elements1.get("altitude_approx_km", 0) - elements2.get("altitude_approx_km", 0))
    inc_diff = abs(elements1.get("inclination_deg", 0) - elements2.get("inclination_deg", 0))
    inc_diff = min(inc_diff, 180.0 - inc_diff)
    return alt_diff < ORBITAL_REGIME_ALT_TOLERANCE_KM and inc_diff < ORBITAL_REGIME_INC_TOLERANCE_DEG


def analyze_conjunctions(
    satellites: List[Tuple[str, str, str]],
    debris_objects: List[Tuple[str, str, str]],
    risk_model: AstroGuardRiskModel,
    hours_ahead: int = 24,
    max_conjunctions: int = 50,
) -> List[ConjunctionEvent]:
    """
    Full conjunction analysis pipeline.
    Returns sorted list of conjunction events by risk (highest first).
    """
    now = datetime.now(timezone.utc)
    duration_minutes = hours_ahead * 60
    events: List[ConjunctionEvent] = []

    logger.info(f"Starting conjunction analysis: {len(satellites)} sats vs {len(debris_objects)} debris")

    # Extract orbital elements for all objects (fast, no propagation)
    sat_elements = []
    for name, l1, l2 in satellites:
        elems = get_orbital_elements_from_tle(l1, l2)
        elems["name"] = name
        elems["l1"] = l1
        elems["l2"] = l2
        sat_elements.append(elems)

    debris_elements = []
    for name, l1, l2 in debris_objects:
        elems = get_orbital_elements_from_tle(l1, l2)
        elems["name"] = name
        elems["l1"] = l1
        elems["l2"] = l2
        debris_elements.append(elems)

    total_pairs_checked = 0
    propagations_done = 0

    for sat_idx, sat_elem in enumerate(sat_elements):
        sat_name = sat_elem.get("name", f"SAT-{sat_idx}")
        sat_l1 = sat_elem.get("l1", "")
        sat_l2 = sat_elem.get("l2", "")
        if not sat_l1 or not sat_l2:
            continue

        sat_obj = build_satellite(sat_l1, sat_l2)
        if sat_obj is None:
            continue

        # Pre-filter debris by orbital regime
        candidate_debris = [
            d for d in debris_elements
            if objects_in_same_regime(sat_elem, d)
        ]

        if not candidate_debris:
            continue

        # Propagate satellite trajectory once
        sat_traj = propagate_trajectory(sat_obj, now, duration_minutes, PROPAGATION_STEP_MINUTES)
        if not sat_traj:
            continue

        for deb_idx, deb_elem in enumerate(candidate_debris):
            deb_name = deb_elem.get("name", f"DEB-{deb_idx}")
            deb_l1 = deb_elem.get("l1", "")
            deb_l2 = deb_elem.get("l2", "")
            if not deb_l1 or not deb_l2:
                continue

            total_pairs_checked += 1
            deb_obj = build_satellite(deb_l1, deb_l2)
            if deb_obj is None:
                continue

            deb_traj = propagate_trajectory(deb_obj, now, duration_minutes, PROPAGATION_STEP_MINUTES)
            if not deb_traj:
                continue

            propagations_done += 1
            closest = find_closest_approach(sat_traj, deb_traj)
            if closest is None:
                continue

            tca_unix, miss_dist_km, rel_vel_km_s, tca_alt_km = closest

            if miss_dist_km > MAX_MISS_DISTANCE_KM:
                continue

            sat_rcs = estimate_rcs_from_type("SATELLITE", sat_name)
            deb_rcs = estimate_rcs_from_type("DEBRIS", deb_name)

            try:
                assessment = risk_model.predict(
                    miss_distance_km=miss_dist_km,
                    relative_velocity_km_s=rel_vel_km_s,
                    altitude_km=tca_alt_km,
                    object1_rcs_m2=sat_rcs,
                    object2_rcs_m2=deb_rcs,
                )
            except Exception as e:
                logger.warning(f"Risk model failed for {sat_name}/{deb_name}: {e}")
                continue

            tca_dt = datetime.fromtimestamp(tca_unix, tz=timezone.utc)
            sat_norad = int(sat_l1[2:7])
            deb_norad = int(deb_l1[2:7])

            obj2_type = "DEBRIS" if ("DEB" in deb_name.upper()) else "ROCKET_BODY" if ("R/B" in deb_name.upper()) else "OBJECT"

            event = ConjunctionEvent(
                id=str(uuid.uuid4())[:8],
                object1_id=sat_norad,
                object1_name=sat_name,
                object2_id=deb_norad,
                object2_name=deb_name,
                object2_type=obj2_type,
                tca_unix=tca_unix,
                tca_iso=tca_dt.isoformat(),
                miss_distance_km=round(miss_dist_km, 4),
                relative_velocity_km_s=round(rel_vel_km_s, 4),
                collision_probability=assessment["collision_probability"],
                risk_level=assessment["risk_level"],
                risk_label=assessment["risk_label"],
                risk_color=assessment["risk_color"],
                altitude_km=round(tca_alt_km, 1),
                recommendation=assessment["recommendation"],
                is_anomaly=assessment.get("is_anomaly", False),
            )
            events.append(event)

            if len(events) >= max_conjunctions * 3:
                break

        if len(events) >= max_conjunctions * 3:
            break

    logger.info(f"Checked {total_pairs_checked} pairs, ran {propagations_done} propagations, found {len(events)} conjunctions")

    # Sort by risk level (desc) then collision probability (desc)
    events.sort(key=lambda e: (-e.risk_level, -e.collision_probability))
    return events[:max_conjunctions]


def analyze_single_satellite(
    sat_name: str,
    sat_l1: str,
    sat_l2: str,
    debris_objects: List[Tuple[str, str, str]],
    risk_model: AstroGuardRiskModel,
    hours_ahead: int = 24,
) -> List[ConjunctionEvent]:
    """Analyze one satellite against all debris objects."""
    return analyze_conjunctions(
        satellites=[(sat_name, sat_l1, sat_l2)],
        debris_objects=debris_objects,
        risk_model=risk_model,
        hours_ahead=hours_ahead,
        max_conjunctions=20,
    )


def generate_demo_conjunctions(risk_model: AstroGuardRiskModel) -> List[ConjunctionEvent]:
    """
    Generate realistic synthetic conjunction events for offline demonstration.
    Clearly labeled as SIMULATION data. Used when live Celestrak data unavailable.
    """
    import uuid
    from datetime import datetime, timezone, timedelta
    import random

    random.seed(42)
    now = datetime.now(timezone.utc)

    # (sat_name, sat_id, deb_name, deb_id, miss_km, vel_km_s, alt_km, sat_rcs, deb_rcs)
    scenarios = [
        ("ISS (ZARYA)",            25544, "COSMOS 1408 DEB",     49271, 0.045, 7.83,  421.5, 400.0, 0.5),
        ("SENTINEL-1A",            39634, "COSMOS 1408 DEB 2",   49272, 0.031, 10.54, 693.0, 80.0,  0.3),
        ("STARLINK-1007",          44713, "SL-8 R/B",            22830, 0.18,  4.21,  556.0, 2.0,   5.0),
        ("TERRA",                  25994, "FENGYUN 1C DEB",       29228, 0.52,  9.12,  710.0, 30.0,  0.2),
        ("AQUA",                   27424, "IRIDIUM 33 DEB",       34506, 1.24,  3.67,  705.0, 30.0,  0.5),
        ("SENTINEL-2A",            40697, "FENGYUN 1C DEB 2",    29229, 2.10,  8.90,  786.0, 80.0,  0.2),
        ("HUBBLE SPACE TELESCOPE", 20580, "SL-16 R/B (ZENIT)",   21897, 3.45,  2.34,  538.0, 20.0,  8.0),
        ("STARLINK-1008",          44714, "STARLINK DEBRIS 001",  47800, 0.09,  0.87,  549.0, 2.0,   0.1),
        ("NOAA 18",                28654, "SL-8 R/B",             22830, 4.88,  6.78,  860.0, 10.0,  5.0),
        ("GLOBALSTAR M081",        40076, "IRIDIUM 33 DEB 2",     34507, 3.67,  5.12, 1414.0, 5.0,   0.5),
    ]

    events = []
    for i, (sat_name, sat_id, deb_name, deb_id, miss_km, vel_km_s, alt_km, sat_rcs, deb_rcs) in enumerate(scenarios):
        tca_dt = now + timedelta(hours=random.uniform(1, 23))

        try:
            assessment = risk_model.predict(
                miss_distance_km=miss_km,
                relative_velocity_km_s=vel_km_s,
                altitude_km=alt_km,
                object1_rcs_m2=sat_rcs,
                object2_rcs_m2=deb_rcs,
            )
        except Exception:
            from .risk_model import RISK_LABELS, RISK_COLORS, RECOMMENDATIONS, pc_to_risk_level, chan_collision_probability
            import math
            r1 = math.sqrt(sat_rcs / math.pi)
            r2 = math.sqrt(deb_rcs / math.pi)
            pc = chan_collision_probability(miss_km, r1 + r2, 200.0)
            risk = pc_to_risk_level(pc)
            assessment = {
                "risk_level": risk,
                "risk_label": RISK_LABELS[risk],
                "risk_color": RISK_COLORS[risk],
                "collision_probability": pc,
                "is_anomaly": False,
                "recommendation": RECOMMENDATIONS[risk],
            }

        is_anom = assessment.get("is_anomaly", False)

        obj2_type = "DEBRIS" if "DEB" in deb_name else "ROCKET_BODY"
        event = ConjunctionEvent(
            id=str(uuid.uuid4())[:8],
            object1_id=sat_id,
            object1_name=f"[SIM] {sat_name}",
            object2_id=deb_id,
            object2_name=deb_name,
            object2_type=obj2_type,
            tca_unix=tca_dt.timestamp(),
            tca_iso=tca_dt.isoformat(),
            miss_distance_km=round(miss_km, 4),
            relative_velocity_km_s=round(vel_km_s, 4),
            collision_probability=assessment["collision_probability"],
            risk_level=assessment["risk_level"],
            risk_label=assessment["risk_label"],
            risk_color=assessment["risk_color"],
            altitude_km=alt_km,
            recommendation=assessment["recommendation"],
            is_anomaly=bool(assessment.get("is_anomaly", is_anom)),
        )
        events.append(event)

    events.sort(key=lambda e: (-e.risk_level, -e.collision_probability))
    return events


def build_space_objects(tle_list: List[Tuple[str, str, str]], object_type: str = "SATELLITE") -> List[SpaceObject]:
    """Convert TLE tuples to SpaceObject models with computed orbital elements."""
    objects = []
    for name, l1, l2 in tle_list:
        try:
            norad_id = int(l1[2:7])
        except (ValueError, IndexError):
            continue
        elems = get_orbital_elements_from_tle(l1, l2)
        obj = SpaceObject(
            norad_id=norad_id,
            name=name,
            tle_line1=l1,
            tle_line2=l2,
            object_type=object_type,
            altitude_km=round(elems.get("altitude_approx_km", 0.0), 1),
            inclination_deg=round(elems.get("inclination_deg", 0.0), 2),
            period_min=round(elems.get("period_min", 0.0), 2),
            rcs_m2=estimate_rcs_from_type(object_type, name),
        )
        objects.append(obj)
    return objects
