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
from pyproj import Transformer
from shapely import Point

from darkvessel.fusion.azimuth import Geometry
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
    geometry: Geometry | None = None,
) -> gpd.GeoDataFrame:
    """Mark each detection matched or dark against the declared positions.

    The tolerance travels with the result rather than staying in a config file: "dark" is a
    claim about what was searched, and it means nothing without the radius that produced it. So
    does the basis of the position that explained a match, for the same reason — a match against
    an interpolated position and one against a report five minutes old are different claims.

    `ais` of None is not an empty AIS slice. An empty slice is a search that returned nothing,
    and its detections are honestly dark; None is no search at all, and its detections are
    `unsearched`, carrying no radius because no radius was applied.

    `geometry` is the orbit the scene was acquired from. Given one, each declared position is
    moved to where the radar would have drawn the vessel before anything is compared — a moving
    target is imaged displaced along the satellite's track, and on the first real scene that put
    four declared vessels 420 to 490 m from their detections and had the chain report all four as
    dark. Left out, no correction is applied and the run matches against the positions as placed,
    which is what the synthetic scene wants: it has no orbit.
    """
    searched = ais is not None
    declared = _drawn_by_the_radar(
        _positions_at_acquisition(ais, acquired_at, detections.crs, max_gap), geometry
    )

    classified = detections.copy()
    classified["status"] = DARK if searched else UNSEARCHED
    classified["mmsi"] = pd.Series(pd.NA, index=classified.index, dtype="string")
    # How big the vessel that explained this detection said it was. Blank on a dark detection,
    # because nothing explained it and there is no vessel to have a size — and blank on a match
    # against a vessel that never declared one.
    classified["length_m"] = np.nan
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
    # How far the declaration had to be moved to sit where the radar drew it. On the layer for
    # the same reason the tolerance is: a match made across four hundred metres of azimuth
    # correction and one made across none are different claims, and only one of them depends on
    # a geometry someone chose.
    classified["azimuth_shift_m"] = np.nan
    classified["acquired_at"] = acquired_at

    for detection_idx, declared_idx, distance_m in _closest_pairs(
        classified, declared, tolerance_m
    ):
        classified.loc[detection_idx, "status"] = MATCHED
        classified.loc[detection_idx, "mmsi"] = declared.loc[declared_idx, "mmsi"]
        classified.loc[detection_idx, "length_m"] = declared.loc[declared_idx, "length_m"]
        classified.loc[detection_idx, "match_distance_m"] = distance_m
        classified.loc[detection_idx, "position_basis"] = declared.loc[
            declared_idx, "position_basis"
        ]
        classified.loc[detection_idx, "position_age_s"] = declared.loc[
            declared_idx, "position_age_s"
        ]
        classified.loc[detection_idx, "azimuth_shift_m"] = declared.loc[
            declared_idx, "azimuth_shift_m"
        ]

    return classified


def _drawn_by_the_radar(declared: gpd.GeoDataFrame, geometry: Geometry | None) -> gpd.GeoDataFrame:
    """Move each declared position to where the radar would have drawn that vessel.

    A vessel with no velocity behind it is left alone and its shift is NaN rather than zero. The
    two are different claims: zero says the chain worked out that this vessel was not moving,
    NaN says it does not know. A position placed from a single report is the second, and moving
    it by nothing would put a fast ship back exactly where the radar did not draw it while
    looking, in the layer, like a correction that had been applied.
    """
    moved = declared.copy()
    if geometry is None or moved.empty:
        moved["azimuth_shift_m"] = np.nan
        return moved

    latitude = _latitude_of(moved)
    shifts = [
        geometry.displacement(east, north, latitude)
        if np.isfinite(east) and np.isfinite(north)
        else None
        for east, north in zip(moved["velocity_east_ms"], moved["velocity_north_ms"], strict=True)
    ]

    moved["azimuth_shift_m"] = [
        float(np.hypot(*shift)) if shift is not None else np.nan for shift in shifts
    ]
    moved.geometry = gpd.GeoSeries(
        [
            point if shift is None else Point(point.x + shift[0], point.y + shift[1])
            for point, shift in zip(moved.geometry, shifts, strict=True)
        ],
        index=moved.index,
        crs=declared.crs,
    )
    return moved


def _latitude_of(positions: gpd.GeoDataFrame) -> float:
    """One latitude for the whole scene, taken from the middle of the declarations.

    The ground track's bearing changes with latitude, but over an eighteen-kilometre box it
    changes by hundredths of a degree — far less than the incidence angle this correction is
    already approximating. One latitude, taken once, rather than a reprojection per vessel.
    """
    centre = positions.geometry.union_all().centroid
    _, latitude = Transformer.from_crs(positions.crs, "EPSG:4326", always_xy=True).transform(
        centre.x, centre.y
    )
    return float(latitude)


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
            {"mmsi": [], "length_m": [], "position_basis": [], "position_age_s": []},
            geometry=[],
            crs=crs,
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
