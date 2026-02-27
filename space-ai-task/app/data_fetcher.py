"""
Fetches Two-Line Element (TLE) orbital data from Celestrak.
Falls back to embedded sample data when network is unavailable.
"""
import requests
import logging
from typing import List, Tuple, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CELESTRAK_BASE = "https://celestrak.org/CTLE/GP.php"

CELESTRAK_GROUPS = {
    "active": "https://celestrak.org/CTLE/GP.php?GROUP=active&FORMAT=tle",
    "debris": "https://celestrak.org/CTLE/GP.php?GROUP=debris&FORMAT=tle",
    "stations": "https://celestrak.org/CTLE/GP.php?GROUP=stations&FORMAT=tle",
    "starlink": "https://celestrak.org/CTLE/GP.php?GROUP=starlink&FORMAT=tle",
}

# Embedded sample TLEs for offline/demo operation
SAMPLE_TLES = """ISS (ZARYA)
1 25544U 98067A   24056.30463572  .00018151  00000-0  32822-3 0  9990
2 25544  51.6416 109.4938 0005770  25.8920  87.4420 15.50024695440375
HUBBLE SPACE TELESCOPE
1 20580U 90037B   24056.56789012  .00001036  00000-0  26022-4 0  9993
2 20580  28.4707  37.8897 0002540 278.2210 253.6717 15.09302447399248
TERRA
1 25994U 99068A   24056.50000000  .00000056  00000-0  15122-4 0  9991
2 25994  98.2008 134.7890 0001234  90.4520 269.6800 14.57110942324156
AQUA
1 27424U 02022A   24056.50000000  .00000082  00000-0  21345-4 0  9994
2 27424  98.2158 133.9870 0001567  85.2340 274.8900 14.57110942324165
SENTINEL-1A
1 39634U 14016A   24056.50000000  .00000041  00000-0  12345-4 0  9997
2 39634  98.1823 136.9870 0001289  92.3450 267.7900 14.59197714521345
SENTINEL-2A
1 40697U 15028A   24056.50000000  .00000039  00000-0  11987-4 0  9991
2 40697  98.5659 132.4561 0001023  95.6780 264.4560 14.30813200312987
COSMOS 1408 DEB
1 49271U 21063WX  24056.50000000  .00006781  00000-0  48923-3 0  9993
2 49271  82.9731 101.2340 0003456  12.3450 347.7890 15.10234567123456
SL-8 R/B
1 22830U 93061B   24056.50000000  .00000123  00000-0  31234-4 0  9994
2 22830  82.9234 112.4560 0023456  45.6780 314.3450 14.93456789234567
FENGYUN 1C DEB
1 29228U 99025AGX 24056.50000000  .00001234  00000-0  24567-3 0  9992
2 29228  98.6234  89.1230 0034567  67.8900 292.2340 14.32456789012345
IRIDIUM 33 DEB
1 34506U 97051FX  24056.50000000  .00002345  00000-0  34521-3 0  9991
2 34506  86.3456  78.2340 0045678  89.0120 271.1230 14.65432109876543
STARLINK-1007
1 44713U 19074A   24056.50000000  .00003456  00000-0  24567-3 0  9996
2 44713  53.0023 145.6780 0001234  23.4560 336.5670 15.06398765432109
STARLINK-1008
1 44714U 19074B   24056.50000000  .00003234  00000-0  22345-3 0  9998
2 44714  53.0045 145.7890 0001345  24.5670 335.4560 15.06287654321098
GLOBALSTAR M081
1 40076U 14054A   24056.50000000  .00000234  00000-0  43234-4 0  9993
2 40076  51.9783 167.8900 0001456  34.5670 325.4560 14.87654321098765
NOAA 18
1 28654U 05018A   24056.50000000 -.00000012  00000-0 -15423-4 0  9991
2 28654  98.7234 123.4560 0008901  45.6780 314.3450 14.09876543210987
NOAA 19
1 33591U 09005A   24056.50000000  .00000023  00000-0  45678-5 0  9994
2 33591  98.7456 124.5670 0012345  56.7890 303.2340 14.11234567890123
COSMOS 2521
1 41579U 16040A   24056.50000000  .00000045  00000-0  00000-0 0  9992
2 41579  64.8234  78.9012 0034567  89.0120 271.1230 11.26543210987654
SL-16 R/B (ZENIT)
1 21897U 92034B   24056.50000000  .00000056  00000-0  67890-5 0  9991
2 21897  71.0145 112.3456 0045678  78.9012 281.2340 14.25678901234567
FENGYUN 1C DEB 2
1 29229U 99025AGY 24056.50000000  .00001567  00000-0  28901-3 0  9993
2 29229  98.5890  89.2340 0023456  67.8901 292.2340 14.30987654321098
COSMOS 1408 DEB 2
1 49272U 21063WY  24056.50000000  .00007234  00000-0  51234-3 0  9994
2 49272  82.9856 101.3451 0002345  13.4561 346.6790 15.11345678234567
IRIDIUM 33 DEB 2
1 34507U 97051FY  24056.50000000  .00002456  00000-0  35612-3 0  9995
2 34507  86.3567  78.3451 0034567  90.1231 270.0120 14.65543210987654
ISS DEB FRAGMENT A
1 48274U 98067NX  24056.31200000  .00019000  00000-0  33500-3 0  9991
2 48274  51.6420 109.5012 0005800  26.0100  88.1200 15.50031234501234
ISS DEB FRAGMENT B
1 48275U 98067NY  24056.31500000  .00018500  00000-0  32100-3 0  9992
2 48275  51.6412 109.4820 0005750  25.7100  87.8900 15.50020987612345
STARLINK DEBRIS 001
1 47800U 21001ZA  24056.50000000  .00003600  00000-0  25000-3 0  9993
2 47800  53.0028 145.6800 0001250  23.5000  336.600 15.06401234512345
COSMOS 1408 DEB 3
1 49273U 21063WZ  24056.50000000  .00006900  00000-0  49500-3 0  9995
2 49273  82.9740 101.2500 0003500  12.4000 347.8500 15.10241234523456
"""


def parse_tle_text(tle_text: str) -> List[Tuple[str, str, str]]:
    """Parse TLE text into list of (name, line1, line2) tuples."""
    objects = []
    lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
    i = 0
    while i < len(lines) - 2:
        if lines[i].startswith("1 ") or lines[i].startswith("2 "):
            i += 1
            continue
        name = lines[i]
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            objects.append((name, lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return objects


def fetch_celestrak_group(group_name: str, timeout: int = 10) -> Optional[str]:
    """Fetch TLE data from Celestrak for a named group."""
    url = CELESTRAK_GROUPS.get(group_name)
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Celestrak fetch failed for '{group_name}': {e}")
        return None


def fetch_all_objects(
    max_satellites: int = 300,
    max_debris: int = 600,
    use_demo_on_failure: bool = True,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Fetch active satellites and debris objects.
    Returns (satellites, debris) as lists of (name, line1, line2).
    Falls back to embedded sample data on network failure.
    """
    logger.info("Fetching active satellites from Celestrak...")
    sat_text = fetch_celestrak_group("active")
    debris_text = fetch_celestrak_group("debris")

    if sat_text:
        satellites = parse_tle_text(sat_text)[:max_satellites]
        logger.info(f"Fetched {len(satellites)} active satellites from Celestrak")
    else:
        logger.warning("Using embedded demo satellites")
        satellites = [t for t in parse_tle_text(SAMPLE_TLES) if "DEB" not in t[0] and "R/B" not in t[0]]

    if debris_text:
        debris = parse_tle_text(debris_text)[:max_debris]
        logger.info(f"Fetched {len(debris)} debris objects from Celestrak")
    else:
        logger.warning("Using embedded demo debris")
        debris = [t for t in parse_tle_text(SAMPLE_TLES) if "DEB" in t[0] or "R/B" in t[0]]

    return satellites, debris


def fetch_demo_data() -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Return embedded sample data for instant offline demo."""
    all_objects = parse_tle_text(SAMPLE_TLES)
    satellites = [t for t in all_objects if "DEB" not in t[0] and "R/B" not in t[0]]
    debris = [t for t in all_objects if "DEB" in t[0] or "R/B" in t[0]]
    return satellites, debris


def get_data_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
