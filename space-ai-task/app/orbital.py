"""
SGP4-based orbital propagation module.
Converts TLE data into ECI positions and velocities for conjunction analysis.
"""
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List

import numpy as np
from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
MU_KM3_S2 = 398600.4418  # Earth's gravitational parameter


def build_satellite(line1: str, line2: str) -> Optional[Satrec]:
    """Build an sgp4 Satrec object from TLE lines."""
    try:
        return Satrec.twoline2rv(line1, line2)
    except Exception as e:
        logger.warning(f"Failed to parse TLE: {e}")
        return None


def propagate(sat: Satrec, dt: datetime) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Propagate satellite to given datetime.
    Returns (position_km, velocity_km_s) in ECI frame, or None on error.
    """
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None
    return np.array(r), np.array(v)


def get_altitude_km(position_eci: np.ndarray) -> float:
    """Compute altitude above Earth's surface from ECI position."""
    return float(np.linalg.norm(position_eci)) - EARTH_RADIUS_KM


def get_orbital_elements_from_tle(line1: str, line2: str) -> dict:
    """
    Extract mean orbital elements from TLE for pre-filtering.
    Returns dict with: inclination_deg, mean_motion_rev_day, altitude_approx_km, period_min, eccentricity
    """
    try:
        sat = build_satellite(line1, line2)
        if sat is None:
            return {}
        inclination_deg = math.degrees(sat.inclo)
        eccentricity = sat.ecco
        mean_motion_rad_min = sat.no  # rad/min
        mean_motion_rev_day = mean_motion_rad_min * (1440.0 / (2 * math.pi))
        period_min = 1440.0 / mean_motion_rev_day if mean_motion_rev_day > 0 else 0
        semi_major_axis_km = (MU_KM3_S2 / ((mean_motion_rad_min / 60.0) ** 2)) ** (1.0 / 3.0)
        altitude_approx_km = semi_major_axis_km - EARTH_RADIUS_KM
        return {
            "inclination_deg": inclination_deg,
            "eccentricity": eccentricity,
            "mean_motion_rev_day": mean_motion_rev_day,
            "period_min": period_min,
            "altitude_approx_km": altitude_approx_km,
            "semi_major_axis_km": semi_major_axis_km,
        }
    except Exception as e:
        logger.warning(f"Could not extract orbital elements: {e}")
        return {}


def propagate_trajectory(
    sat: Satrec,
    start_time: datetime,
    duration_minutes: int,
    step_minutes: float = 5.0,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Propagate a satellite over a time range.
    Returns list of (unix_timestamp, position_eci_km, velocity_eci_km_s).
    """
    trajectory = []
    n_steps = int(duration_minutes / step_minutes)
    for i in range(n_steps + 1):
        dt = start_time + timedelta(minutes=i * step_minutes)
        result = propagate(sat, dt)
        if result is not None:
            pos, vel = result
            trajectory.append((dt.timestamp(), pos, vel))
    return trajectory


def find_closest_approach(
    traj1: List[Tuple[float, np.ndarray, np.ndarray]],
    traj2: List[Tuple[float, np.ndarray, np.ndarray]],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Find the time of closest approach (TCA) between two trajectories.
    Returns (tca_unix, miss_distance_km, rel_velocity_km_s, altitude_km) or None.
    Aligns trajectories by timestamp.
    """
    if not traj1 or not traj2:
        return None

    t1_map = {t: (p, v) for t, p, v in traj1}
    t2_map = {t: (p, v) for t, p, v in traj2}
    common_times = sorted(set(t1_map.keys()) & set(t2_map.keys()))

    if not common_times:
        return None

    min_dist = float("inf")
    tca = None
    tca_vel = 0.0
    tca_alt = 0.0

    for t in common_times:
        p1, v1 = t1_map[t]
        p2, v2 = t2_map[t]
        dist = float(np.linalg.norm(p1 - p2))
        if dist < min_dist:
            min_dist = dist
            tca = t
            rel_vel = np.linalg.norm(v1 - v2)
            tca_vel = float(rel_vel)
            tca_alt = get_altitude_km(p1)

    if tca is None or min_dist > 100.0:
        return None

    return tca, min_dist, tca_vel, tca_alt


def estimate_rcs_from_type(object_type: str, name: str) -> float:
    """
    Estimate radar cross-section (m²) as proxy for object size.
    Used in collision probability calculation.
    """
    name_upper = name.upper()
    if "DEB" in name_upper:
        return 0.1
    if "R/B" in name_upper or "ROCKET" in name_upper:
        return 10.0
    if "ISS" in name_upper or "STATION" in name_upper:
        return 400.0
    if "STARLINK" in name_upper or "ONEWEB" in name_upper:
        return 2.0
    if "HUBBLE" in name_upper:
        return 20.0
    return 1.0


def get_orbit_points_for_visualization(sat: Satrec, n_points: int = 90) -> List[dict]:
    """
    Generate orbit ground track points for visualization.
    Returns list of {lat, lon, alt} dicts.
    """
    now = datetime.now(timezone.utc)
    elements = {}
    try:
        result = propagate(sat, now)
        if result is None:
            return []
        period_min = 90.0
        sat_obj = sat
        mean_motion = sat_obj.no * (1440.0 / (2 * math.pi))
        if mean_motion > 0:
            period_min = 1440.0 / mean_motion
    except Exception:
        period_min = 90.0

    points = []
    for i in range(n_points):
        dt = now + timedelta(minutes=i * period_min / n_points)
        result = propagate(sat, dt)
        if result is None:
            continue
        pos, _ = result
        x, y, z = pos
        r = np.linalg.norm(pos)
        lat = math.degrees(math.asin(z / r))
        lon = math.degrees(math.atan2(y, x))
        alt = r - EARTH_RADIUS_KM
        # Adjust longitude for Earth's rotation
        gst_offset = (dt.timestamp() - now.timestamp()) * 360.0 / 86400.0
        lon = (lon - gst_offset + 180) % 360 - 180
        points.append({"lat": lat, "lon": lon, "alt": alt})
    return points
