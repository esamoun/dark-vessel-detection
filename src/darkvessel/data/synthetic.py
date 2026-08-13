"""A synthetic scene and AIS slice, for exercising the chain without downloading anything.

The point of this module is that the whole pipeline can be run, and its output opened in QGIS,
by someone who has just cloned the repository: no Earth Engine credentials, no AIS archive, no
weights. It is also what the seam test runs on, so the ground truth below is stated once and
asserted against literals worked out from it by hand.

The scene is a dark sea with faint speckle and four bright targets. Two of them have a vessel
declaring itself nearby in AIS; the third does not, and must come back dark. A vessel declaring
itself far outside the scene must match nothing.

Where it sits is chosen, not left to a round number. The scene covers 2.56 km of open water in
the Kattegat, inside the area `configs/anholt.yaml` fetches a real acquisition over. The first
thing anyone does with the output is drag it onto a basemap, and a demonstration of a maritime
pipeline whose detections land in a field in Jutland answers a question nobody asked.

The fourth target stands where the tiles the shipped config cuts this scene into meet, and it is
the whole reason the scene is larger than one tile. Several tiles see it and exactly one may
report it. It declares itself, so both ways of getting the reconciliation wrong show up in the
result rather than in an error: lose it and a declared vessel disappears, report it twice and the
copy no declaration is left to explain becomes a dark vessel that was never there.
"""

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer

CRS = "EPSG:25832"
SIZE_PX = 256
PIXEL_M = 10.0
ORIGIN_X = 639_000.0
ORIGIN_Y = 6_282_000.0
TRANSFORM = Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y)

ACQUIRED_AT = datetime(2026, 3, 14, 5, 30, tzinfo=UTC)
ACQUIRED_AT_TAG = "ACQUIRED_AT"

SPECKLE_SEED = 20260314
SPECKLE_MAX = 0.2

# (row, col) of each target centre. Ground coordinates, for reference and for the test to assert
# against: (60, 80) -> 639805, 6281395 | (120, 200) -> 641005, 6280795 | (30, 220) -> 641205,
# 6281695 | (128, 128) -> 640285, 6280715. Each is `origin + (index + 0.5) * pixel size`,
# northing decreasing.
#
# The last one is placed, not chosen: at the tile size and overlap `configs/pipeline.yaml` sets,
# this scene is cut into four tiles that meet at row 128 and column 128. A target anywhere else
# would leave cross-tile reconciliation unexercised by the shipped configuration.
TARGETS = [(60, 80), (120, 200), (30, 220), (128, 128)]

# (mmsi, offset from acquisition, easting offset from a target, northing offset), where the
# target is the one at the same position in TARGETS. The first vessel reports twice: an old
# report far to the west, and a recent one alongside the target. Matching against the old one
# would call a declared vessel dark, which is the failure this arrangement exists to catch.
DECLARATIONS = [
    ("219000001", TARGETS[0], timedelta(hours=-2), -1500.0, 0.0),
    ("219000001", TARGETS[0], timedelta(minutes=-3), 40.0, 0.0),
    ("219000002", TARGETS[1], timedelta(minutes=-1), 0.0, 60.0),
    # A vessel that declared itself well outside the scene: nothing here should match it.
    ("219000003", TARGETS[2], timedelta(minutes=-5), -8000.0, -8000.0),
    # The target on the tile boundary declares itself, so that a copy of it surviving
    # reconciliation appears as a dark vessel rather than as a harmless second row.
    ("219000004", TARGETS[3], timedelta(minutes=-2), 0.0, -50.0),
]


def write_synthetic_inputs(directory: Path) -> tuple[Path, Path]:
    """Write `scene.tif` and `ais.csv` into `directory`. Returns both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    scene_path = directory / "scene.tif"
    ais_path = directory / "ais.csv"

    _write_scene(scene_path)
    _write_ais(ais_path)
    return scene_path, ais_path


def _write_scene(path: Path) -> None:
    rng = np.random.default_rng(SPECKLE_SEED)
    image = rng.uniform(0.0, SPECKLE_MAX, size=(SIZE_PX, SIZE_PX)).astype(np.float32)
    for row, col in TARGETS:
        image[row - 1 : row + 2, col - 1 : col + 2] = 1.0

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE_PX,
        width=SIZE_PX,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
    ) as dataset:
        dataset.write(image, 1)
        dataset.update_tags(**{ACQUIRED_AT_TAG: ACQUIRED_AT.isoformat()})


def _write_ais(path: Path) -> None:
    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mmsi", "timestamp", "lon", "lat"])
        for mmsi, (row, col), age, east_m, north_m in DECLARATIONS:
            x, y = TRANSFORM * (col + 0.5, row + 0.5)
            lon, lat = to_wgs84.transform(x + east_m, y + north_m)
            writer.writerow([mmsi, (ACQUIRED_AT + age).isoformat(), f"{lon:.8f}", f"{lat:.8f}"])
