"""Which zone a detection stands in, decided from a file rather than from a catalogue.

The other three contextual variables are rasters, reduced at a point by Earth Engine in one round
trip. This one is not a reduction at all — `gee_layers.py` says so itself, that a polygon
membership is not a reducer, and does it as a spatial join. A join against polygons on the disk is
the same join, minus an account, a project and a network.

That is the whole of why this module exists apart from that one. Filling the EEZ column through
`darkvessel context` would mean re-sampling shore, depth and fishing effort to write a variable
that depends on none of them, and would put an Earth Engine credential between a reader and an
answer that a GeoPackage and a point-in-polygon test can give in a second. So this writes one
column and leaves the other three exactly as it found them.

**Three answers, not two.** A detection inside a zone carries its name. A detection inside the
water this file covers but inside no zone carries `high seas`, which is an answer. A detection
outside what the file covers carries `unavailable`, which is not — it is a statement about this
run. The distinction is older than this module and `tests/test_context.py` has held it since the
sampling was written; what is new is that there is now a third case, a detection beyond the
fetched rectangle, and it belongs on the `unavailable` side.

**Two zones claiming one position is reported as two zones claiming it**, joined into a single
name rather than resolved. A rule that took the first, or the smallest, or the one with the lower
identifier would be inventing a maritime boundary in a tie-break — and boundaries genuinely do
overlap, which is why Marine Regions carries `Joint regime` polygons at all. It does not arise in
this study area: measured over the archive's 189 detections, none stands in two zones. The
behaviour is written down and tested rather than left to be discovered by the first run where it
does.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import box as rectangle

from darkvessel.context.gee_layers import DEGREES, EEZ, HIGH_SEAS, UNAVAILABLE
from darkvessel.data.eez import EEZ_LAYER, MARINE_REGIONS, Zones

# What separates two zones that both claim one position, in the name that goes on the row.
CLAIMED_BY = " / "


def attach(detections: gpd.GeoDataFrame, boundaries: Zones, *, field: str) -> gpd.GeoDataFrame:
    """Put the zone each detection stands in on its row, and touch nothing else.

    By spatial join in degrees, which is where the boundaries are published and where a
    point-in-polygon test is exact regardless of projection — unlike a distance, which is why the
    rest of this chain works in metres.
    """
    if detections.crs is None:
        raise ValueError(
            "these detections carry no CRS, so there is no way to say which water they are in"
        )

    carried = detections.copy()
    if len(detections) == 0:
        carried[EEZ] = []
        return carried

    placed = detections.to_crs(DEGREES) if str(detections.crs) != DEGREES else detections
    name = boundaries.column(field)
    covered = placed.geometry.within(rectangle(*boundaries.covers.as_rectangle()))

    claims = _claims(placed, boundaries.zones[[name, "geometry"]], name)
    carried[EEZ] = [
        claims.get(index) or (HIGH_SEAS if inside else UNAVAILABLE)
        for index, inside in zip(placed.index, covered, strict=True)
    ]
    return carried


def _claims(placed: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, name: str) -> dict[object, str]:
    """Every zone that claims each position, by the index of the detection it claims.

    A position claimed twice keeps both names. See the module docstring: a tie-break here would
    be this repository deciding a maritime boundary, and the two words the column is made of are
    about evidence rather than about tidiness.
    """
    joined = gpd.sjoin(
        placed[["geometry"]], zones.rename(columns={name: EEZ}), how="inner", predicate="within"
    )
    found: dict[object, set[str]] = {}
    for index, zone in zip(joined.index, joined[EEZ], strict=True):
        found.setdefault(index, set()).add(str(zone))
    return {index: CLAIMED_BY.join(sorted(names)) for index, names in found.items()}


def zones_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """Which boundaries answer this column, and where the fetched copy of them lives.

    The same shape as the other `*_request_from` functions, and the path this one resolves is the
    one `context_request_from` said it would resolve "the day an EEZ layer is ingested from a file
    rather than named as a remote asset". It is a file now, and it is not in the repository:
    Marine Regions asks not to be redistributed, so `data/` is where it lands, beside the scenes
    and the AIS archive that are fetched and ignored for their own reasons.
    """
    settings = config.get("context", {}).get("eez", {})
    if "reference" not in settings:
        raise ValueError(
            "context.eez.reference is missing: it is the file `darkvessel eez` fetches the "
            "boundaries into and `darkvessel zones` reads them back out of"
        )
    margin_m = float(settings.get("margin_m", 0.0))
    if margin_m < 0:
        raise ValueError(f"a margin of {margin_m} m cannot be negative")

    return {
        "source": str(settings.get("source", MARINE_REGIONS)),
        "layer": str(settings.get("layer", EEZ_LAYER)),
        "field": str(settings.get("property", "SOVEREIGN1")),
        "margin_m": margin_m,
        "reference": (relative_to / str(settings["reference"])).resolve(),
    }
