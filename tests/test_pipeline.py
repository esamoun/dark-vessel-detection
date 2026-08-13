"""The single seam: the full chain, with the detector injected.

Given a synthetic scene with bright targets at known ground coordinates, a synthetic AIS slice,
and a trivial deterministic detector substituted for the trained model, the pipeline must return
the expected detections at the expected coordinates, each correctly classified as matched or
dark.

These tests exercise the three paths where an error is silent at this stage: georeferencing (a
target must land at its true ground coordinate), tiling (a target on a tile boundary must come
back exactly once) and matching (a detection must be called dark only when no declared position
stands within the stated tolerance). Ground coordinates below are worked out by hand from the
affine transform the scene is built with, so a fault in the conversion cannot agree with the
expectation.

The geometry of the tiling itself is not tested here — a scene fully covered is a property of the
layout, and a strip skipped between two targets leaves no trace in a fixture. That lives in
`test_tiling.py`; what this file asserts is what comes out of the chain.

Not yet covered, because the chain does not yet do it: AIS interpolation to the acquisition time
(matching is against the nearest report in time). It arrives with the level that introduces it,
and this file grows to meet it.

Detector quality is not tested here. A model is evaluated, not asserted.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from affine import Affine
from shapely import Point

from darkvessel.cli import main
from darkvessel.data.scene import Scene
from darkvessel.data.synthetic import SIZE_PX, TARGETS, write_synthetic_inputs
from darkvessel.data.tiling import Tiling
from darkvessel.detect.threshold import BrightPixelDetector
from darkvessel.pipeline import run

# The scene the tests are built on: 10 m pixels, north-up, in the working CRS. Every expected
# ground coordinate in this file derives from these three numbers and nothing else.
ORIGIN_X = 500_000.0
ORIGIN_Y = 6_150_000.0
PIXEL_M = 10.0

WORKING_CRS = "EPSG:25832"
ACQUIRED_AT = datetime(2026, 3, 14, 5, 30, tzinfo=UTC)
TOLERANCE_M = 200.0

# Larger than any scene in this file, so the tests that are not about tiling see one tile and
# nothing else changes underneath them.
ONE_TILE = Tiling(size_px=1024, overlap_px=64)

# Against the 64 px scenes below: tiles start at 0 and 28 on both axes and the scene splits down
# the middle, so a target at row or column 32 sits exactly on a boundary and is seen by four
# tiles at once. The overlap is comfortably wider than a 3 px target.
FOUR_TILES = Tiling(size_px=36, overlap_px=8)
SPLIT_PX = 32

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"

CONFIG = (
    # Paths are relative to the config file, so a run is portable between machines.
    "area:\n"
    f"  crs: {WORKING_CRS}\n"
    # Cuts the 256 px synthetic scene into four tiles meeting at row and column 128, where the
    # fixture stands a target: a run driven by a config file has to cross a seam, not just the
    # tests that call the pipeline directly.
    "tiling:\n"
    "  size_px: 144\n"
    "  overlap_px: 32\n"
    "run:\n"
    "  scene: scene.tif\n"
    "  ais: ais.csv\n"
    "  output: detections.gpkg\n"
    "  detector: bright-pixel\n"
    "  threshold: 0.5\n"
    "fusion:\n"
    f"  match_tolerance_m: {TOLERANCE_M:g}\n"
)


def synthetic_scene(targets: list[tuple[int, int]], size: int = 64) -> Scene:
    """A dark sea with a bright 3x3 target centred on each (row, col) given."""
    image = np.zeros((size, size), dtype=np.float32)
    for row, col in targets:
        image[row - 1 : row + 2, col - 1 : col + 2] = 1.0
    return Scene(
        image=image,
        transform=Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y),
        crs=WORKING_CRS,
        acquired_at=ACQUIRED_AT,
    )


def ais_slice(reports: list[tuple[str, datetime, float, float]]) -> gpd.GeoDataFrame:
    """Declared positions, as (mmsi, timestamp, x, y) in the working CRS."""
    return gpd.GeoDataFrame(
        {
            "mmsi": [mmsi for mmsi, _, _, _ in reports],
            "timestamp": pd.to_datetime([when for _, when, _, _ in reports], utc=True),
        },
        geometry=[Point(x, y) for _, _, x, y in reports],
        crs=WORKING_CRS,
    )


def detect(
    scene: Scene,
    ais: gpd.GeoDataFrame | None = None,
    tolerance_m: float = TOLERANCE_M,
    tiling: Tiling = ONE_TILE,
) -> gpd.GeoDataFrame:
    """Run the chain with the deterministic stand-in detector."""
    return run(
        scene=scene,
        ais=ais,
        detector=BrightPixelDetector(threshold=0.5),
        tiling=tiling,
        tolerance_m=tolerance_m,
    )


def run_from_config(directory: Path) -> gpd.GeoDataFrame:
    """Write synthetic inputs and a config into `directory`, run the command, read the output."""
    write_synthetic_inputs(directory)
    config = directory / "pipeline.yaml"
    config.write_text(CONFIG)

    assert main(["run", "--config", str(config)]) == 0
    return gpd.read_file(directory / "detections.gpkg")


def detection_at(detections: gpd.GeoDataFrame, x: float, y: float) -> pd.Series:
    """The one detection standing at the given ground coordinate."""
    here = detections[detections.geometry.distance(Point(x, y)) < 1.0]
    assert len(here) == 1, f"expected exactly one detection at ({x}, {y}), found {len(here)}"
    return here.iloc[0]


def test_detection_lands_at_its_true_ground_coordinate() -> None:
    scene = synthetic_scene(targets=[(20, 30)])

    detections = detect(scene)

    assert len(detections) == 1
    assert detections.crs == WORKING_CRS
    point = detections.geometry.iloc[0]
    # Centre of pixel (row 20, col 30): 500000 + 30.5 * 10, 6150000 - 20.5 * 10.
    assert point.x == pytest.approx(500_305.0)
    assert point.y == pytest.approx(6_149_795.0)


def test_a_target_on_a_tile_boundary_is_reported_exactly_once() -> None:
    # The target sits on the corner where all four tiles meet: every one of them sees it, three
    # of them must keep quiet. Losing it and reporting it twice are both silent faults — the
    # count stays plausible either way — so the assertion is on the count and on the place.
    scene = synthetic_scene(targets=[(SPLIT_PX, SPLIT_PX)])

    detections = detect(scene, tiling=FOUR_TILES)

    assert len(detections) == 1
    # Centre of pixel (32, 32): 500000 + 32.5 * 10, 6150000 - 32.5 * 10.
    detection_at(detections, 500_325.0, 6_149_675.0)


def test_no_part_of_a_scene_larger_than_one_tile_goes_unread() -> None:
    # One target in each of the four tiles' cores, plus the one on the boundary between them.
    scene = synthetic_scene(targets=[(10, 10), (10, 50), (50, 10), (50, 50), (32, 32)])

    detections = detect(scene, tiling=FOUR_TILES)

    assert len(detections) == 5
    for x, y in [
        (500_105.0, 6_149_895.0),
        (500_505.0, 6_149_895.0),
        (500_105.0, 6_149_495.0),
        (500_505.0, 6_149_495.0),
        (500_325.0, 6_149_675.0),
    ]:
        detection_at(detections, x, y)


def test_the_tiling_a_run_uses_does_not_change_what_it_finds() -> None:
    # Tiling is a constraint of the hardware, not a parameter of the answer. The same scene cut
    # three ways — whole, in four, and in tiles small enough that the boundary target lands in a
    # different tile again — must produce the same detections in the same order.
    scene = synthetic_scene(targets=[(10, 10), (10, 50), (50, 10), (50, 50), (32, 32)])

    whole = detect(scene, tiling=ONE_TILE)
    quartered = detect(scene, tiling=FOUR_TILES)
    finely = detect(scene, tiling=Tiling(size_px=20, overlap_px=8))

    pd.testing.assert_frame_equal(whole, quartered)
    pd.testing.assert_frame_equal(whole, finely)


def test_a_vessel_that_declared_itself_is_matched_and_one_that_did_not_is_dark() -> None:
    # Two targets: pixel (20, 30) -> (500305, 6149795), pixel (40, 10) -> (500105, 6149595).
    scene = synthetic_scene(targets=[(20, 30), (40, 10)])
    ais = ais_slice(
        [
            # Two hours stale and 1305 m west — the wrong report to match against.
            ("219000001", ACQUIRED_AT - timedelta(hours=2), 499_000.0, 6_149_795.0),
            # Three minutes old and 40 m east of the target — the right one.
            ("219000001", ACQUIRED_AT - timedelta(minutes=3), 500_345.0, 6_149_795.0),
        ]
    )

    detections = detect(scene, ais)

    assert len(detections) == 2

    declared = detection_at(detections, 500_305.0, 6_149_795.0)
    assert declared["status"] == "matched"
    assert declared["mmsi"] == "219000001"
    assert declared["match_distance_m"] == pytest.approx(40.0)
    assert declared["tolerance_m"] == pytest.approx(200.0)

    undeclared = detection_at(detections, 500_105.0, 6_149_595.0)
    assert undeclared["status"] == "dark"
    assert pd.isna(undeclared["mmsi"])
    assert undeclared["tolerance_m"] == pytest.approx(200.0)


def test_a_declared_position_beyond_the_tolerance_leaves_the_detection_dark() -> None:
    scene = synthetic_scene(targets=[(20, 30)])
    # 250 m east of the target at (500305, 6149795).
    ais = ais_slice([("219000001", ACQUIRED_AT, 500_555.0, 6_149_795.0)])

    within = detect(scene, ais, tolerance_m=300.0)
    beyond = detect(scene, ais, tolerance_m=200.0)

    assert within["status"].iloc[0] == "matched"
    assert beyond["status"].iloc[0] == "dark"


def test_one_declared_position_cannot_explain_two_detections() -> None:
    # Two targets 60 m apart: pixel (20, 30) -> (500305, 6149795), (20, 36) -> (500365, 6149795).
    scene = synthetic_scene(targets=[(20, 30), (20, 36)])
    # One vessel between them: 20 m from the first, 40 m from the second, both inside tolerance.
    ais = ais_slice([("219000001", ACQUIRED_AT, 500_325.0, 6_149_795.0)])

    detections = detect(scene, ais)

    nearer = detection_at(detections, 500_305.0, 6_149_795.0)
    farther = detection_at(detections, 500_365.0, 6_149_795.0)
    assert nearer["status"] == "matched"
    assert nearer["mmsi"] == "219000001"
    # Two hulls that close together are two vessels, and only one of them declared itself.
    assert farther["status"] == "dark"
    assert pd.isna(farther["mmsi"])


def test_a_single_command_turns_a_config_into_a_georeferenced_layer(tmp_path: Path) -> None:
    written = run_from_config(tmp_path)

    assert written.crs == WORKING_CRS
    assert len(written) == 4

    # Ground coordinates of the four synthetic targets, worked out from the transform that
    # `write_synthetic_inputs` documents.
    declared_a = detection_at(written, 500_805.0, 6_149_395.0)
    declared_b = detection_at(written, 502_005.0, 6_148_795.0)
    undeclared = detection_at(written, 502_205.0, 6_149_695.0)
    on_the_seam = detection_at(written, 501_285.0, 6_148_715.0)

    assert declared_a["status"] == "matched"
    assert declared_a["mmsi"] == "219000001"
    assert declared_b["status"] == "matched"
    assert declared_b["mmsi"] == "219000002"
    assert undeclared["status"] == "dark"
    assert undeclared["tolerance_m"] == pytest.approx(200.0)

    # The target standing where the four tiles meet. `detection_at` fails if it came back twice,
    # and it declared itself, so a duplicate would also surface as an extra dark vessel: one
    # declared position cannot explain two detections.
    assert on_the_seam["status"] == "matched"
    assert on_the_seam["mmsi"] == "219000004"
    assert (written["status"] == "dark").sum() == 1


def test_the_shipped_config_still_cuts_the_synthetic_scene_across_a_target() -> None:
    """The claim the README makes about the shipped run, pinned.

    Every test above builds its own config, so `configs/pipeline.yaml` is the one file here that
    nothing executes: widen its tiling until the synthetic scene fits in a single tile and the
    suite stays green while the shipped command quietly stops crossing a seam at all. The
    fixture's boundary target is only a boundary target with respect to a tiling, and this is
    the tiling a reader actually runs.
    """
    tiling = Tiling(**yaml.safe_load(SHIPPED_CONFIG.read_text())["tiling"])
    row, col = TARGETS[-1]

    saw_it = [
        tile
        for tile in tiling.tiles((SIZE_PX, SIZE_PX))
        if tile.rows.start <= row < tile.rows.stop and tile.cols.start <= col < tile.cols.stop
    ]

    assert len(saw_it) > 1, (
        f"the shipped tiling gives target {TARGETS[-1]} to {len(saw_it)} tile(s); with fewer "
        "than two the shipped run never reconciles anything across a tile boundary"
    )


def test_the_same_inputs_produce_the_same_output_every_time(tmp_path: Path) -> None:
    # Two full runs of the command, each generating its own inputs, rather than one pipeline
    # called twice: the synthetic scene and the written layer have to be reproducible too.
    first = run_from_config(tmp_path / "first")
    second = run_from_config(tmp_path / "second")

    pd.testing.assert_frame_equal(first, second)


def test_a_scene_outside_the_declared_working_crs_is_refused(tmp_path: Path) -> None:
    write_synthetic_inputs(tmp_path)
    config = tmp_path / "pipeline.yaml"
    config.write_text(CONFIG.replace(WORKING_CRS, "EPSG:4326"))

    # A tolerance in metres compared against degrees matches or darkens everything for the
    # wrong reason, and never crashes on its own.
    with pytest.raises(ValueError, match="working CRS"):
        main(["run", "--config", str(config)])
