"""AIS ingestion from the Danish Maritime Authority open archives.

Downloads daily archives, parses position reports, filters to the study area and time window.
Raw AIS is noisy - duplicated messages, implausible jumps, missing MMSI - and cleaning is part
of the work rather than a preliminary to it.

At this level only the last step is implemented: reading position reports that are already on
disk and putting them in the working CRS. Downloading and cleaning arrive with the level that
needs real AIS.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# AIS reports positions in WGS84; everything downstream works in the projected CRS.
AIS_CRS = "EPSG:4326"


def load_ais(path: Path, crs: str) -> gpd.GeoDataFrame:
    """Read position reports from CSV and return them in `crs`.

    Expects columns `mmsi`, `timestamp`, `lon`, `lat`. MMSI is kept as text: it is an
    identifier, never a quantity, and reading it as a number invites a leading zero to be lost
    or a missing value to turn it into a float.
    """
    reports = pd.read_csv(path, dtype={"mmsi": "string"})
    reports["timestamp"] = pd.to_datetime(reports["timestamp"], utc=True, format="ISO8601")

    return gpd.GeoDataFrame(
        reports.drop(columns=["lon", "lat"]),
        geometry=gpd.points_from_xy(reports["lon"], reports["lat"]),
        crs=AIS_CRS,
    ).to_crs(crs)
