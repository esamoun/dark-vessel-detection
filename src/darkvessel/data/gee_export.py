"""Sentinel-1 selection and export via Earth Engine.

Scenes are selected and exported server-side so that full GRD products never transit the local
disk. Records acquisition time per scene: the AIS fusion stage depends on it being exact.

The catalogue is a parameter of the export, not an import inside it — the same seam that lets the
pipeline run with a substitute detector. Earth Engine needs credentials and a network, so every
decision that can be made without it is made on this side of that boundary and tested there: which
scene to take, whether a request is small enough to answer, and what metadata has to survive into
the file the chain reads. `earth_engine` builds the real catalogue and is the only code here that
imports `ee`, which keeps the package installable without the `gee` extra.

Two things are carried across deliberately. The acquisition timestamp, because a scene without one
is refused by `scene.py` rather than guessed at — an hour of drift is 22 km of vessel track at the
fusion stage and nothing downstream could see it. And the georeferencing Earth Engine itself wrote,
which is taken as it stands: this module adds tags to the fetched file and never recomputes a
transform. Rebuilding one from a bounding box and a pixel size is a plausible-looking way to put
every detection in the wrong place.
"""

import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import rasterio

from darkvessel.data.area import Bounds
from darkvessel.data.scene import ACQUIRED_AT_TAG

# Enough to choose from, few enough that the metadata call stays small. A fortnight over one
# area is a handful of acquisitions; a listing longer than this means the window is too wide to
# describe a run anyway.
_MOST_CANDIDATES_WORTH_LISTING = 50

SCENE_ID_TAG = "SCENE_ID"
POLARISATIONS_TAG = "POLARISATIONS"
ORBIT_PASS_TAG = "ORBIT_PASS"

COLLECTION = "COPERNICUS/S1_GRD"

# Earth Engine's own limit on a single direct download, quoted from its refusal rather than
# guessed at: "Total request size (57353670 bytes) must be less than or equal to 50331648 bytes."
# This was 64 MB for as long as no request came near it, on the reasoning that the real cap is a
# server-side detail that would go stale in a comment — which was true, and left the guard unable
# to do the one thing it exists for. The second study area is what found it, at 57 MB.
MAX_REQUEST_BYTES = 48 * 1024 * 1024
# Nine, not eight, and the ninth byte is the reason the guard let that request through. Earth
# Engine returns S1 GRD bands as float64 — eight bytes, itself a measurement rather than a guess,
# after this was once set at four — and counts a byte of validity mask alongside each sample. The
# refusal is what says so and says it exactly: the scene that came back is 1845 x 1727 px, and
# 1845 x 1727 x 2 bands x 18 bytes is 57 353 670, which is the number in the message to the byte.
# The band count is in the caller's hands, so this is per sample rather than per pixel.
BYTES_PER_SAMPLE = 9


@dataclass(frozen=True)
class DateWindow:
    """The span searched for an acquisition. Both ends timezone-aware, like everything here."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("the search window must be timezone-aware; acquisitions are in UTC")
        if self.start >= self.end:
            raise ValueError(f"the search window ends before it starts: {self.start} to {self.end}")


@dataclass(frozen=True)
class SceneRef:
    """One acquisition as the catalogue describes it, before any pixels have moved."""

    id: str
    acquired_at: datetime
    polarisations: tuple[str, ...]
    orbit_pass: str


class Catalogue(Protocol):
    """What the export needs of Earth Engine, and nothing beyond it."""

    def search(
        self, area: Bounds, window: DateWindow, polarisations: tuple[str, ...]
    ) -> Sequence[SceneRef]:
        """Acquisitions covering `area` within `window` that carry every polarisation asked for."""
        ...

    def geotiff(
        self,
        scene: SceneRef,
        area: Bounds,
        polarisations: tuple[str, ...],
        crs: str,
        resolution_m: float,
    ) -> bytes:
        """One scene, clipped to `area` and reprojected server-side, as GeoTIFF bytes."""
        ...


def export_scene(
    *,
    catalogue: Catalogue,
    area: Bounds,
    window: DateWindow,
    polarisations: tuple[str, ...],
    crs: str,
    resolution_m: float,
    path: Path,
) -> SceneRef:
    """Fetch the first acquisition covering `area` in `window`, and write it to `path`.

    Returns what the catalogue said about the scene, so a caller can report it without reading
    the file back.
    """
    _refuse_a_request_too_large_to_answer(area, polarisations, crs, resolution_m)

    candidates = catalogue.search(area, window, polarisations)
    if not candidates:
        raise ValueError(
            f"no {COLLECTION} acquisition covers {area.as_rectangle()} between "
            f"{window.start.isoformat()} and {window.end.isoformat()} "
            f"with polarisations {', '.join(polarisations)}; widen the window or the area"
        )

    # The earliest, rather than whichever the catalogue happened to list first: a run has to be
    # repeatable, and "the scene I got that day" is not a description anyone can act on.
    scene = min(candidates, key=lambda candidate: (candidate.acquired_at, candidate.id))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(catalogue.geotiff(scene, area, polarisations, crs, resolution_m))
    _record(scene, path)
    return scene


def export_archive(
    *,
    catalogue: Catalogue,
    area: Bounds,
    window: DateWindow,
    polarisations: Sequence[str],
    crs: str,
    resolution_m: float,
    directory: Path,
) -> list[SceneRef]:
    """Fetch every acquisition covering `area` in `window`, one file each, and return them.

    `export_scene` takes the earliest acquisition of a window because a run is defined over one
    scene. An archive is the other question: the embedding level asks which detections across the
    record resemble one another, and one 17 km box holds a handful of vessels at any one instant,
    so what makes that question mean anything is many acquisitions of the same water.

    Resumable, and that is not a convenience. Thirty scenes is thirty downloads over a connection
    that will drop at least once, and a session that has to start again from the first is a
    session nobody runs to the end. A scene already on the disk is left alone — by name, which is
    the acquisition's own identity, so "already fetched" cannot mean a different scene.

    Ordered by acquisition, so the directory listing is the record in the order it was acquired
    and two runs of this command produce the same archive in the same order.
    """
    found = sorted(
        catalogue.search(area, window, tuple(polarisations)),
        key=lambda scene: (scene.acquired_at, scene.id),
    )
    if not found:
        raise ValueError(
            f"no {COLLECTION} acquisition covers {area.as_rectangle()} between "
            f"{window.start.isoformat()} and {window.end.isoformat()} "
            f"with polarisations {', '.join(polarisations)}; widen the window or the area"
        )

    _refuse_a_request_too_large_to_answer(area, tuple(polarisations), crs, resolution_m)
    directory.mkdir(parents=True, exist_ok=True)

    for scene in found:
        path = directory / f"{scene.id.rsplit('/', 1)[-1]}.tif"
        if path.exists():
            continue
        path.write_bytes(catalogue.geotiff(scene, area, tuple(polarisations), crs, resolution_m))
        _record(scene, path)

    return found


def _record(scene: SceneRef, path: Path) -> None:
    """Write into the file what the catalogue knew and the pixels do not carry.

    Opened for update rather than rewritten: the transform and CRS in the fetched file are Earth
    Engine's own and are left exactly as they arrived.
    """
    with rasterio.open(path, "r+") as dataset:
        dataset.update_tags(
            **{
                ACQUIRED_AT_TAG: scene.acquired_at.isoformat(),
                SCENE_ID_TAG: scene.id,
                POLARISATIONS_TAG: ",".join(scene.polarisations),
                ORBIT_PASS_TAG: scene.orbit_pass,
            }
        )


def earth_engine(project: str | None = None) -> Catalogue:
    """The real catalogue, against the live Earth Engine API.

    The only function in the package that imports `ee`, and the only code here that a test
    cannot reach: everything it does happens on Google's side of a credentialed connection. It
    is kept this thin for that reason — it filters, it reads metadata, it fetches bytes, and it
    decides nothing. Whatever it returns is checked by `export_scene`, which is tested.
    """
    try:
        import ee
    except ModuleNotFoundError as missing:
        # The package installs without Earth Engine on purpose — the chain runs with no network
        # — so arriving here is a normal state to be in, not a broken installation.
        raise ModuleNotFoundError(
            "Earth Engine is an optional extra of this package and is not installed. "
            'Install it with `pip install -e ".[gee]"`, then authenticate once with '
            "`earthengine authenticate`. Everything except `darkvessel export` runs without it."
        ) from missing

    ee.Initialize(project=project)
    return _EarthEngine()


class _EarthEngine:
    """Earth Engine, expressed as the four things this package needs from it."""

    def search(
        self, area: Bounds, window: DateWindow, polarisations: tuple[str, ...]
    ) -> list[SceneRef]:
        import ee

        collection = (
            ee.ImageCollection(COLLECTION)
            .filterBounds(ee.Geometry.Rectangle(area.as_rectangle()))
            .filterDate(window.start.isoformat(), window.end.isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        for polarisation in polarisations:
            collection = collection.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", polarisation)
            )

        # `select([])` drops the bands: this call is about metadata, and asking for the pixels
        # of every candidate to decide which one to fetch would defeat the point of deciding.
        listing = collection.select([]).limit(_MOST_CANDIDATES_WORTH_LISTING).getInfo()
        return [_scene_ref(feature) for feature in listing["features"]]

    def geotiff(
        self,
        scene: SceneRef,
        area: Bounds,
        polarisations: tuple[str, ...],
        crs: str,
        resolution_m: float,
    ) -> bytes:
        import ee

        region = ee.Geometry.Rectangle(area.as_rectangle())
        image = ee.Image(scene.id).select(list(polarisations)).clip(region)
        # Clipping, reprojection and resampling all happen on Google's machines; what crosses the
        # network is the window asked for and nothing else. This is the whole reason no GRD
        # product reaches the local disk.
        url = image.getDownloadURL(
            {
                "region": region,
                "scale": resolution_m,
                "crs": crs,
                "format": "GEO_TIFF",
                "filePerBand": False,
            }
        )
        with urllib.request.urlopen(url) as response:  # noqa: S310 — a URL Earth Engine signed
            return bytes(response.read())


def _scene_ref(feature: dict[str, Any]) -> SceneRef:
    """One catalogue entry, as this package describes an acquisition.

    `system:time_start` is milliseconds since the epoch, UTC. Read without a timezone it becomes
    whatever the machine running the export is set to, and `scene.py` would accept it: the
    acquisition time is then wrong by the operator's offset from UTC, and only the fusion stage
    would ever show it — as vessels that appear not to have declared themselves.
    """
    properties = feature["properties"]
    return SceneRef(
        id=feature["id"],
        acquired_at=datetime.fromtimestamp(properties["system:time_start"] / 1000, tz=UTC),
        polarisations=tuple(properties["transmitterReceiverPolarisation"]),
        orbit_pass=properties["orbitProperties_pass"],
    )


def _refuse_a_request_too_large_to_answer(
    area: Bounds,
    polarisations: tuple[str, ...],
    crs: str,
    resolution_m: float,
) -> None:
    """Refuse an area too large to come back in one response, before anything is sent.

    Earth Engine refuses an oversized request too, after the wait, and its message does not say
    which number to change: the area, the resolution, or the number of polarisations.

    All four corners are transformed, not two. A rectangle in degrees is not a rectangle in a
    projected CRS — its edges bow, and its bounding box is wider than the box between two
    opposite corners. Taking two corners understated the first study area by 6.5% and understated
    every area in the one direction that matters, which is why the guard once waved through a
    request Earth Engine then refused. On the two areas this project has exported, the four
    corners land within 0.1% of the grid Earth Engine settled on.
    """
    from pyproj import Transformer

    to_working = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [
        to_working.transform(longitude, latitude)
        for longitude in (area.west, area.east)
        for latitude in (area.south, area.north)
    ]
    eastings = [easting for easting, _ in corners]
    northings = [northing for _, northing in corners]

    pixels = ((max(eastings) - min(eastings)) / resolution_m) * (
        (max(northings) - min(northings)) / resolution_m
    )
    size = pixels * len(polarisations) * BYTES_PER_SAMPLE
    if size > MAX_REQUEST_BYTES:
        raise ValueError(
            f"{area.as_rectangle()} at {resolution_m:g} m in {len(polarisations)} polarisations is "
            f"about {size / 1e6:.0f} MB, past the {MAX_REQUEST_BYTES / 1e6:.0f} MB Earth Engine "
            "answers in a single response; shrink the area, drop a polarisation, coarsen the "
            "resolution, or export to Drive — see docs/decisions.md"
        )
