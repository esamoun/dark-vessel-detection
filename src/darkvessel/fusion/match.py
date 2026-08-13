"""Spatio-temporal matching, and the dark vessel decision.

Detections are matched to declared AIS positions within a tolerance derived from position
uncertainty and geolocation error. What remains unmatched is reported as a candidate, with its
tolerance stated - a claim about evidence, not a verdict. Geometry-critical: covered by tests.

The position matched against is the report nearest in time to the acquisition, taken as it
stands. That is deliberately naive and it is the weakest link in the chain: a vessel moves
between its last report and the moment the radar imaged it, so a stale position can manufacture
a dark vessel that does not exist. Interpolating along the track to the acquisition time
replaces this, and is the whole subject of `interpolate.py`.
"""

from collections.abc import Iterator
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

MATCHED = "matched"
DARK = "dark"
# No declarations were supplied, so nothing was compared against anything. Distinct from `dark`,
# which says a search happened and came back empty: a run with no AIS that marked its detections
# dark would produce a layer that reads, to anyone opening it, as a sea full of undeclared
# vessels. That is the most confident wrong answer this chain could give.
UNSEARCHED = "unsearched"


def classify(
    detections: gpd.GeoDataFrame,
    ais: gpd.GeoDataFrame | None,
    acquired_at: datetime,
    tolerance_m: float,
) -> gpd.GeoDataFrame:
    """Mark each detection matched or dark against the declared positions.

    The tolerance travels with the result rather than staying in a config file: "dark" is a
    claim about what was searched, and it means nothing without the radius that produced it.

    `ais` of None is not an empty AIS slice. An empty slice is a search that returned nothing,
    and its detections are honestly dark; None is no search at all, and its detections are
    `unsearched`, carrying no radius because no radius was applied.
    """
    searched = ais is not None

    classified = detections.copy()
    classified["status"] = DARK if searched else UNSEARCHED
    classified["mmsi"] = pd.Series(pd.NA, index=classified.index, dtype="string")
    classified["match_distance_m"] = np.nan
    classified["tolerance_m"] = float(tolerance_m) if searched else np.nan
    classified["acquired_at"] = acquired_at

    declared = _positions_at_acquisition(ais, acquired_at, classified.crs)
    for detection_idx, declared_idx, distance_m in _closest_pairs(
        classified, declared, tolerance_m
    ):
        classified.loc[detection_idx, "status"] = MATCHED
        classified.loc[detection_idx, "mmsi"] = declared.loc[declared_idx, "mmsi"]
        classified.loc[detection_idx, "match_distance_m"] = distance_m

    return classified


def _positions_at_acquisition(
    ais: gpd.GeoDataFrame | None,
    acquired_at: datetime,
    crs: str,
) -> gpd.GeoDataFrame:
    """One declared position per vessel: its report nearest in time to the acquisition."""
    if ais is None or ais.empty:
        return gpd.GeoDataFrame({"mmsi": []}, geometry=[], crs=crs)

    if ais.crs != crs:
        ais = ais.to_crs(crs)

    staleness = (ais["timestamp"] - pd.Timestamp(acquired_at)).abs()
    # Sorting on the timestamp as well, with a stable sort, keeps the choice deterministic when
    # a vessel reported equally far either side of the acquisition.
    nearest_first = ais.assign(_staleness=staleness).sort_values(
        ["_staleness", "timestamp"], kind="stable"
    )
    # Whole rows, rather than a per-column reduction: a groupby aggregate would happily pair one
    # report's timestamp with another report's position.
    return nearest_first.drop_duplicates(subset="mmsi", keep="first")


def _closest_pairs(
    detections: gpd.GeoDataFrame,
    declared: gpd.GeoDataFrame,
    tolerance_m: float,
) -> Iterator[tuple[int, int, float]]:
    """Pair detections with declared positions, closest first, each used at most once.

    One declared position cannot explain two detections: two hulls that close together are two
    vessels, and only one of them declared itself. Quadratic in the number of detections and of
    declarations, which is fine for the tens of each a synthetic scene produces. A real scene
    against a day of Danish AIS is a different size of problem and wants a spatial index; that
    belongs with the level that first runs one, not before.
    """
    if declared.empty:
        return

    candidates = []
    for detection_idx, detection in detections.geometry.items():
        distances = declared.geometry.distance(detection)
        for declared_idx, distance_m in distances[distances <= tolerance_m].items():
            candidates.append((float(distance_m), detection_idx, declared_idx))

    claimed_detections: set[int] = set()
    claimed_declarations: set[int] = set()
    for distance_m, detection_idx, declared_idx in sorted(candidates):
        if detection_idx in claimed_detections or declared_idx in claimed_declarations:
            continue
        claimed_detections.add(detection_idx)
        claimed_declarations.add(declared_idx)
        yield detection_idx, declared_idx, distance_m
