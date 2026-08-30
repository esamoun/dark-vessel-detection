"""Contextual variables sampled from the Earth Engine catalogue.

Distance to shore, bathymetry, EEZ boundaries, fishing effort. Turns a set of detections into
an answer to a question: where do undeclared vessels concentrate, and under what conditions.

A detection on its own is a coordinate and a score. Whether it is interesting depends on where it
is standing: eight kilometres off a coast in twelve metres of water inside a national EEZ, in a
square where fishing effort has always been recorded, is a different object from the same score in
four hundred metres of water on the high seas. None of that is in the radar scene, and all of it
is in somebody's published raster.

**The sampling happens on Google's side of the connection.** Every detection of a scene is sent as
one feature collection and reduced against the stacked layers there; what crosses the network is a
table of values, never a raster. The alternative — fetching four global products and sampling them
on a laptop — is tens of gigabytes to answer a question about a few hundred points, and it is the
same reason `data/gee_export.py` clips and reprojects server-side.

**A value nobody could sample is missing, never zero.** This is the whole constraint of the
module, and it is not pedantry: no fishing effort recorded in a square is a real and useful fact
about that water, zero metres from shore is a detection aground, and a depth of zero is the
waterline. Every one of those numbers is a plausible answer, so an unanswered layer that filled in
a zero would be indistinguishable from a finding. Numbers come back NaN, which a GeoPackage
carries as NULL and QGIS shows as empty; the EEZ comes back as one of two words — `high seas`,
meaning the position is outside every zone, which is an answer, and `unavailable`, meaning the
layer did not give one.

The layers arrive as a parameter, the way the catalogue does in `data/gee_export.py` and the
detector does in `pipeline.py`. Everything that can be decided without a network is decided on
this side of that seam and tested there: the frame the points are asked in, the row each answer
lands on, and what a missing value looks like by the time it reaches the file.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

# What Earth Engine is addressed in. The chain works in metres in a projected CRS; the catalogue
# is asked in degrees, and a UTM easting handed over as a longitude is not an error anything
# downstream could see — it samples a position in the Gulf of Guinea and returns a number.
DEGREES = "EPSG:4326"

# The columns a sampled detection carries. Ordered the way a reader would ask the questions:
# how far from land, how deep, whose water, and how much fishing has been recorded there.
DISTANCE_TO_SHORE = "distance_to_shore_m"
DEPTH = "depth_m"
EEZ = "eez"
FISHING_HOURS = "fishing_hours"

CONTEXT = (DISTANCE_TO_SHORE, DEPTH, EEZ, FISHING_HOURS)
# The three that are numbers. The EEZ is a name, and the difference matters here rather than
# being an implementation detail: a name has no NaN, so it carries its own word for missing.
MEASURED = (DISTANCE_TO_SHORE, DEPTH, FISHING_HOURS)

# Outside every exclusive economic zone. An answer, not the absence of one.
HIGH_SEAS = "high seas"
# The layer was not asked, or was asked and said nothing. Never confused with the above: one is a
# statement about the sea and the other is a statement about this run.
UNAVAILABLE = "unavailable"

# What each column is called when a run talks about it out loud.
LABELS = {
    DISTANCE_TO_SHORE: "distance to shore",
    DEPTH: "water depth",
    FISHING_HOURS: "fishing effort",
}


@dataclass(frozen=True)
class Context:
    """What the catalogue said about one position.

    `None` means the layer did not answer, in every field. It is spelled that way rather than as
    NaN because this is the boundary a sampler writes to, and a sampler that has to remember which
    sentinel each field uses is a sampler that will eventually pick the wrong one.

    There is no zone here, and there is no `eez=None` waiting for a sampler that might answer it
    one day either. A catalogue of rasters cannot answer a polygon membership, `context/zones.py`
    can and does, and a field on this boundary that no implementation ever fills would be a second
    route into the one column whose two missing-value words have to stay apart. See #35.
    """

    distance_to_shore_m: float | None = None
    depth_m: float | None = None
    fishing_hours: float | None = None


class Layers(Protocol):
    """What the contextual sampling needs of Earth Engine, and nothing beyond it."""

    def sample(self, points: Sequence[tuple[float, float]]) -> Sequence[Context]:
        """One answer per point, in the order given. Points are `(longitude, latitude)`."""
        ...


@dataclass(frozen=True)
class LayerSources:
    """Which published product answers each question, named in the config rather than in here.

    Asset identifiers belong in the file that describes a run, for the same reason the checkpoint
    path does: the catalogue is somebody else's, it moves, and a hard-coded identifier is a claim
    this repository cannot keep true. What is here instead is the shape of the request and the
    checks that can be made before it is sent.

    `effort` may be `None`, and that is a configuration rather than a failure: a source left null
    means the run cannot sample that variable, and every row then says `unavailable` rather than
    carrying a zero nobody measured.

    There is no EEZ source here, and its absence is the decision of #35. Earth Engine's public
    catalogue carries no EEZ boundaries, and the conclusion drawn from that until then — ingest
    them as a table asset and name the asset here — does not follow from it. A zone is a polygon
    and membership is a point-in-polygon test, which needs no catalogue: `context/zones.py` does
    it against a file, and `darkvessel zones` fills that column without an account. What is left
    here is the three variables that really are rasters reduced at a point.
    """

    shore: str
    search_radius_m: float
    depth: str
    depth_band: str
    effort: str | None
    effort_bands: tuple[str, ...]
    effort_start: str
    effort_end: str
    scale_m: float

    def __post_init__(self) -> None:
        if self.scale_m <= 0:
            raise ValueError(
                f"a sampling scale of {self.scale_m} m samples nothing; it is the grid the "
                "layers are reduced on, and Earth Engine needs it in metres"
            )
        if self.search_radius_m <= 0:
            raise ValueError(
                f"a search radius of {self.search_radius_m} m finds no shore; beyond it the "
                "distance is unmeasured, which this reports as missing rather than as far"
            )
        if self.effort_start >= self.effort_end:
            raise ValueError(
                f"the fishing-effort window ends before it starts: {self.effort_start} to "
                f"{self.effort_end}"
            )
        if self.effort is not None and not self.effort_bands:
            raise ValueError(
                f"{self.effort} is named as the fishing-effort source with no bands to read; "
                "the collection carries one band per gear type and none of them is the total"
            )


def without_context(detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The columns a sampled run carries, on a run that sampled nothing.

    Present and empty rather than absent. The chain writes them on every layer, including the
    synthetic demo that has never seen a network, because a layer whose schema depends on which
    stages were switched on is a layer that cannot be stacked with the one beside it — the same
    promise `embedder.attach` and `register.without_a_register` make.
    """
    carried = detections.copy()
    for name in MEASURED:
        carried[name] = np.full(len(detections), np.nan)
    carried[EEZ] = np.full(len(detections), UNAVAILABLE, dtype=object)
    return carried


def attach(detections: gpd.GeoDataFrame, layers: Layers) -> gpd.GeoDataFrame:
    """Sample every layer at every detection, and put the answers on the rows they belong to.

    One call for the whole scene rather than one per detection: the sampling is a reduction over
    a feature collection, and a loop of round trips would turn a few seconds into a few minutes
    and bill each one separately.

    By position, and the length is checked rather than trusted. Earth Engine's own reducers make
    no promise about the order a feature collection comes back in — the real sampler carries an
    index through and reorders on it — and a sampler that quietly returned one row fewer would
    put every value on the wrong vessel and still write a layer that opens.
    """
    if detections.crs is None:
        raise ValueError(
            "these detections carry no CRS, so there is no way to say where they are in the "
            "degrees the catalogue is addressed in"
        )
    if len(detections) == 0:
        return without_context(detections)

    placed = detections.to_crs(DEGREES) if str(detections.crs) != DEGREES else detections
    points = list(zip(placed.geometry.x, placed.geometry.y, strict=True))

    sampled = list(layers.sample(points))
    if len(sampled) != len(points):
        raise ValueError(
            f"the layers answered {len(sampled)} of {len(points)} detections; the answers are "
            "attached by position, so a short reply would put each value on the wrong row "
            "rather than on none"
        )

    carried = detections.copy()
    carried[DISTANCE_TO_SHORE] = [_number(one.distance_to_shore_m) for one in sampled]
    carried[DEPTH] = [_number(one.depth_m) for one in sampled]
    carried[FISHING_HOURS] = [_number(one.fishing_hours) for one in sampled]
    # The zone column is **not** written here, only guaranteed to exist. This sampler does not
    # answer it — `context/zones.py` does, from published polygons — and a version of this that
    # wrote `unavailable` over whatever it found would silently undo `darkvessel zones` every
    # time somebody re-sampled the rasters. The schema promise is kept without the overwrite:
    # a layer that arrives without the column leaves with it, empty.
    if EEZ not in carried.columns:
        carried[EEZ] = np.full(len(carried), UNAVAILABLE, dtype=object)
    return carried


def coverage(detections: gpd.GeoDataFrame) -> list[str]:
    """What each layer answered, as the lines a run prints.

    Counted off the layer that was written rather than off a tally kept while sampling, for the
    reason `register.reduction` is: what someone opening the output can reproduce is the only
    honest version of this figure. It is also the only place a reader is told that a variable came
    back empty everywhere — a column of NULLs in an attribute table is easy to scroll past, and a
    run that sampled nothing and a run whose layer had no coverage here look identical in QGIS.
    """
    total = len(detections)
    if total == 0:
        return ["no detections to sample the contextual layers at"]

    lines = []
    for name in MEASURED:
        answered = int(np.isfinite(pd.to_numeric(detections[name], errors="coerce")).sum())
        lines.append(f"{LABELS[name]}: {answered} of {total} detection(s) carry a value")

    lines.append(zone_coverage(detections))
    return lines


def zone_coverage(detections: gpd.GeoDataFrame) -> str:
    """What the zone column answered, as one line.

    Its own function because two commands report it — `context`, which no longer fills it, and
    `zones`, which does. One sentence with one owner: a second copy of it would be free to start
    counting `high seas` and `unavailable` as the same thing, which is the one mistake this
    variable exists to avoid.
    """
    total = len(detections)
    zones = detections[EEZ].astype(object)
    unavailable = int((zones == UNAVAILABLE).sum())
    high_seas = int((zones == HIGH_SEAS).sum())
    return (
        f"EEZ: {total - unavailable - high_seas} in a named EEZ, {high_seas} on the high seas, "
        f"{unavailable} unavailable"
    )


def _number(value: float | None) -> float:
    """A sampled number, or NaN. The one place `None` becomes the missing value of a column."""
    return float("nan") if value is None else float(value)


# The property each point carries through the reduction, so that what comes back can be put in the
# order it was asked in. Prefixed, because it travels through somebody else's feature collection
# alongside whatever properties that collection already has.
INDEX = "darkvessel_index"


def earth_engine_layers(sources: LayerSources, project: str | None = None) -> Layers:
    """The real layers, against the live Earth Engine API.

    The counterpart of `gee_export.earth_engine`, and kept thin for the same reason: everything it
    does happens on Google's side of a credentialed connection, so it is the one part of this
    level a test cannot reach. It builds requests, it reads answers, and it decides nothing —
    whether a value is missing, what missing looks like in the file, and which row each answer
    belongs to are all settled by `attach`, which is tested.
    """
    try:
        import ee
    except ModuleNotFoundError as missing:
        raise ModuleNotFoundError(
            "Earth Engine is an optional extra of this package and is not installed. "
            'Install it with `pip install -e ".[gee]"`, then authenticate once with '
            "`earthengine authenticate`. Everything except `darkvessel export`, "
            "`darkvessel scenes` and `darkvessel context` runs without it."
        ) from missing

    ee.Initialize(project=project)
    return _EarthEngineLayers(sources)


class _EarthEngineLayers:
    """Earth Engine, expressed as the one thing this level needs from it.

    Two round trips per scene, not one per detection: the rasters are stacked and reduced in one
    call, and the EEZ is a spatial join in a second because a polygon membership is not a reducer.
    """

    def __init__(self, sources: LayerSources) -> None:
        self.sources = sources

    def sample(self, points: Sequence[tuple[float, float]]) -> list[Context]:
        import ee

        if not points:
            return []

        features = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point([longitude, latitude]), {INDEX: index})
                for index, (longitude, latitude) in enumerate(points)
            ]
        )
        measured = self._reduced(features)
        return [
            Context(
                distance_to_shore_m=measured.get(index, {}).get(DISTANCE_TO_SHORE),
                depth_m=measured.get(index, {}).get(DEPTH),
                fishing_hours=measured.get(index, {}).get(FISHING_HOURS),
            )
            for index in range(len(points))
        ]

    def _reduced(self, features: Any) -> dict[int, dict[str, float | None]]:
        """The raster layers, stacked and reduced at every point in one call.

        A band the stack does not carry — a source this run has none for — is simply absent from
        the properties that come back, and `.get` then leaves the field `None`. That is the whole
        handling of an unsampled layer: nothing fills it in on the way past.
        """
        import ee

        sources = self.sources
        # Distance in metres to the nearest land polygon. `FeatureCollection.distance` returns an
        # image that is masked beyond the search radius, which is what makes "further than this
        # was not measured" arrive as missing rather than as the radius itself.
        stack = (
            ee.FeatureCollection(sources.shore)
            .distance(sources.search_radius_m)
            .rename(DISTANCE_TO_SHORE)
        )
        stack = stack.addBands(ee.Image(sources.depth).select(sources.depth_band).rename(DEPTH))
        if sources.effort is not None:
            # Summed over the window the config names, because one day of fishing effort at one
            # point is mostly zero and says nothing. The window is a property of the run: no
            # effort product covers the year these scenes were acquired, so what this samples is
            # where effort concentrated over the years that are published. See docs/decisions.md.
            #
            # Two sums, because the collection is two-dimensional: one image per flag state per
            # day — 15 004 of them in 2016 — and one band per gear type. `.sum()` adds the
            # images, `Reducer.sum()` adds the gear. The band list is in the config because there
            # is no total band; Earth Engine said so when this was written against a guessed one:
            # "Band pattern 'WLD' did not match any bands. Available bands: [drifting_longlines,
            # fixed_gear, other_fishing, purse_seines, squid_jigger, trawlers]".
            #
            # `unmask(0)` on this band and on no other, which looks like the thing this module
            # exists to refuse and is the opposite of it. A mask means something different in
            # each of these products: an unmeasured depth is a place nobody surveyed, and an
            # unmeasured distance is beyond the search radius, so both stay missing. GFW's grid
            # covers the ocean, and a masked cell there is a cell where no fishing hours were
            # recorded — an answer, and the answer this variable is most often going to have.
            # Left masked it would arrive as NaN and be indistinguishable from the run that has
            # no effort source at all, which is exactly the confusion being avoided elsewhere.
            stack = stack.addBands(
                ee.ImageCollection(sources.effort)
                .filterDate(sources.effort_start, sources.effort_end)
                .select(list(sources.effort_bands))
                .sum()
                .reduce(ee.Reducer.sum())
                .unmask(0)
                .rename(FISHING_HOURS)
            )

        answered = stack.reduceRegions(
            collection=features, reducer=ee.Reducer.first(), scale=sources.scale_m
        ).getInfo()
        return {
            int(feature["properties"][INDEX]): {
                name: feature["properties"].get(name) for name in MEASURED
            }
            for feature in answered["features"]
        }
