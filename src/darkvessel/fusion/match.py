"""Spatio-temporal matching, and the dark vessel decision.

Detections are matched to declared AIS positions within a tolerance derived from position
uncertainty and geolocation error. What remains unmatched is reported as a candidate, with its
tolerance stated - a claim about evidence, not a verdict. Geometry-critical: covered by tests.

The positions matched against are not the reports as they stand. Each vessel is placed at the
acquisition instant by `interpolate.py` before anything is compared, because a vessel moves
between its last report and the moment the radar imaged it and a stale position manufactures a
dark vessel that does not exist. Where a track gives nothing to interpolate between, the nearest
report is used and the row says so: `position_basis` travels with every match, so the strength
of the evidence behind a match is in the layer rather than in an assumption.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd

from darkvessel.fusion.interpolate import positions_at

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
    max_gap: timedelta,
) -> gpd.GeoDataFrame:
    """Mark each detection matched or dark against the declared positions.

    The tolerance travels with the result rather than staying in a config file: "dark" is a
    claim about what was searched, and it means nothing without the radius that produced it. So
    does the basis of the position that explained a match, for the same reason — a match against
    an interpolated position and one against a report five minutes old are different claims.

    `ais` of None is not an empty AIS slice. An empty slice is a search that returned nothing,
    and its detections are honestly dark; None is no search at all, and its detections are
    `unsearched`, carrying no radius because no radius was applied.
    """
    searched = ais is not None
    declared = _positions_at_acquisition(ais, acquired_at, detections.crs, max_gap)

    classified = detections.copy()
    classified["status"] = DARK if searched else UNSEARCHED
    classified["mmsi"] = pd.Series(pd.NA, index=classified.index, dtype="string")
    classified["match_distance_m"] = np.nan
    classified["tolerance_m"] = float(tolerance_m) if searched else np.nan
    # How many declarations the radius was applied to. `dark` is a claim about a search, and a
    # search of nothing is the one case where the claim is technically true and reads as its
    # opposite: a layer of a hundred dark vessels over a quiet sea, with nothing in it to say
    # that no vessel declared itself there in the first place. The first real slice this project
    # ingested was exactly that, so the count travels with the verdict, like the radius.
    classified["declarations_searched"] = float(len(declared)) if searched else np.nan
    classified["position_basis"] = pd.Series(pd.NA, index=classified.index, dtype="string")
    classified["position_age_s"] = np.nan
    classified["acquired_at"] = acquired_at

    for detection_idx, declared_idx, distance_m in _closest_pairs(
        classified, declared, tolerance_m
    ):
        classified.loc[detection_idx, "status"] = MATCHED
        classified.loc[detection_idx, "mmsi"] = declared.loc[declared_idx, "mmsi"]
        classified.loc[detection_idx, "match_distance_m"] = distance_m
        classified.loc[detection_idx, "position_basis"] = declared.loc[
            declared_idx, "position_basis"
        ]
        classified.loc[detection_idx, "position_age_s"] = declared.loc[
            declared_idx, "position_age_s"
        ]

    return classified


def _positions_at_acquisition(
    ais: gpd.GeoDataFrame | None,
    acquired_at: datetime,
    crs: str,
    max_gap: timedelta,
) -> gpd.GeoDataFrame:
    """One declared position per vessel, placed at the acquisition instant.

    Reprojected first: interpolation is linear in whatever CRS it is handed, and a straight line
    in degrees is not the straight line in metres that the tolerance is compared against.
    """
    if ais is None or ais.empty:
        return gpd.GeoDataFrame(
            {"mmsi": [], "position_basis": [], "position_age_s": []}, geometry=[], crs=crs
        )

    if ais.crs != crs:
        ais = ais.to_crs(crs)

    return positions_at(ais, acquired_at, max_gap)


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
