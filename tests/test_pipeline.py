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
`test_tiling.py`; what this file asserts is what comes out of the chain. Placing a vessel at the
acquisition instant is the same shape of thing and lives in `test_interpolate.py`: the chain
cannot show whether a position is right, only whether the classification that followed from it
looks plausible. What this file asserts about it is that the placement reaches the matching, and
changes the verdict where it should.

Detector quality is not tested here. A model is evaluated, not asserted.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from affine import Affine
from shapely import Point

from darkvessel.cli import (
    archive_request_from,
    check_tile_size,
    embedding_request_from,
    fusion_settings_from,
    main,
    trained_request_from,
)
from darkvessel.config import load_config
from darkvessel.data.scene import Scene
from darkvessel.data.synthetic import BOUNDARY_TARGET, SIZE_PX, write_synthetic_inputs
from darkvessel.data.tiling import Tiling
from darkvessel.detect.threshold import BrightPixelDetector
from darkvessel.embed.embedder import columns, vectors_of
from darkvessel.pipeline import run

# The scene the tests are built on: 10 m pixels, north-up, in the working CRS. Every expected
# ground coordinate in this file derives from these three numbers and nothing else.
ORIGIN_X = 639_000.0
ORIGIN_Y = 6_282_000.0
PIXEL_M = 10.0

WORKING_CRS = "EPSG:25832"
ACQUIRED_AT = datetime(2026, 3, 14, 5, 30, tzinfo=UTC)
TOLERANCE_M = 200.0
# The widest bracket of AIS reports these tests allow a position to be interpolated across. The
# chain has no default for it, for the reason it has none for the tolerance: it decides what the
# answer is, so a run states it.
MAX_GAP = timedelta(minutes=10)

# Larger than any scene in this file, so the tests that are not about tiling see one tile and
# nothing else changes underneath them.
ONE_TILE = Tiling(size_px=1024, overlap_px=64)

# Against the 64 px scenes below: tiles start at 0 and 28 on both axes and the scene splits down
# the middle, so a target at row or column 32 sits exactly on a boundary and is seen by four
# tiles at once. The overlap is comfortably wider than a 3 px target.
FOUR_TILES = Tiling(size_px=36, overlap_px=8)
SPLIT_PX = 32

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHIPPED_CONFIG = CONFIGS / "pipeline.yaml"

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
    "  interpolation_max_gap_s: 600\n"
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


def ais_slice(
    reports: list[tuple[str, datetime, float, float]],
    lengths: dict[str, float] | None = None,
) -> gpd.GeoDataFrame:
    """Declared positions, as (mmsi, timestamp, x, y) in the working CRS.

    A vessel absent from `lengths` never declared a size, which is most of the archive and the
    default here for that reason.
    """
    declared = lengths or {}
    return gpd.GeoDataFrame(
        {
            "mmsi": [mmsi for mmsi, _, _, _ in reports],
            "timestamp": pd.to_datetime([when for _, when, _, _ in reports], utc=True),
            "length_m": [declared.get(mmsi, float("nan")) for mmsi, _, _, _ in reports],
        },
        geometry=[Point(x, y) for _, _, x, y in reports],
        crs=WORKING_CRS,
    )


def detect(
    scene: Scene,
    ais: gpd.GeoDataFrame | None = None,
    tolerance_m: float = TOLERANCE_M,
    tiling: Tiling = ONE_TILE,
    max_gap: timedelta = MAX_GAP,
) -> gpd.GeoDataFrame:
    """Run the chain with the deterministic stand-in detector."""
    return run(
        scene=scene,
        ais=ais,
        detector=BrightPixelDetector(threshold=0.5),
        tiling=tiling,
        tolerance_m=tolerance_m,
        max_gap=max_gap,
    )


def run_from_config(directory: Path, config_text: str = CONFIG) -> gpd.GeoDataFrame:
    """Write synthetic inputs and a config into `directory`, run the command, read the output."""
    write_synthetic_inputs(directory)
    config = directory / "pipeline.yaml"
    config.write_text(config_text)

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
    # Centre of pixel (row 20, col 30): 639000 + 30.5 * 10, 6282000 - 20.5 * 10.
    assert point.x == pytest.approx(639_305.0)
    assert point.y == pytest.approx(6_281_795.0)


def test_a_target_on_a_tile_boundary_is_reported_exactly_once() -> None:
    # The target sits on the corner where all four tiles meet: every one of them sees it, three
    # of them must keep quiet. Losing it and reporting it twice are both silent faults — the
    # count stays plausible either way — so the assertion is on the count and on the place.
    scene = synthetic_scene(targets=[(SPLIT_PX, SPLIT_PX)])

    detections = detect(scene, tiling=FOUR_TILES)

    assert len(detections) == 1
    # Centre of pixel (32, 32): 639000 + 32.5 * 10, 6282000 - 32.5 * 10.
    detection_at(detections, 639_325.0, 6_281_675.0)


def test_no_part_of_a_scene_larger_than_one_tile_goes_unread() -> None:
    # One target in each of the four tiles' cores, plus the one on the boundary between them.
    scene = synthetic_scene(targets=[(10, 10), (10, 50), (50, 10), (50, 50), (32, 32)])

    detections = detect(scene, tiling=FOUR_TILES)

    assert len(detections) == 5
    for x, y in [
        (639_105.0, 6_281_895.0),
        (639_505.0, 6_281_895.0),
        (639_105.0, 6_281_495.0),
        (639_505.0, 6_281_495.0),
        (639_325.0, 6_281_675.0),
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
    # Two targets: pixel (20, 30) -> (639305, 6281795), pixel (40, 10) -> (639105, 6281595).
    scene = synthetic_scene(targets=[(20, 30), (40, 10)])
    ais = ais_slice(
        [
            # Two hours stale and 1305 m west — the wrong report to match against.
            ("219000001", ACQUIRED_AT - timedelta(hours=2), 638_000.0, 6_281_795.0),
            # Three minutes old and 40 m east of the target — the right one.
            ("219000001", ACQUIRED_AT - timedelta(minutes=3), 639_345.0, 6_281_795.0),
        ]
    )

    detections = detect(scene, ais)

    assert len(detections) == 2

    declared = detection_at(detections, 639_305.0, 6_281_795.0)
    assert declared["status"] == "matched"
    assert declared["mmsi"] == "219000001"
    assert declared["match_distance_m"] == pytest.approx(40.0)
    assert declared["tolerance_m"] == pytest.approx(200.0)

    undeclared = detection_at(detections, 639_105.0, 6_281_595.0)
    assert undeclared["status"] == "dark"
    assert pd.isna(undeclared["mmsi"])
    assert undeclared["tolerance_m"] == pytest.approx(200.0)


def test_a_matched_detection_carries_the_size_the_vessel_declared() -> None:
    """What explained the detection, and whether the radar could plausibly have seen it.

    A 228 m tanker is twenty pixels at 10 m and a 15 m sailing boat is a pixel and a half. Which
    of the two matched is the difference between a chain that works and a chain that agreed with
    a threshold on sea clutter, and a reader with only the layer cannot tell them apart unless
    the size is in the row — the same argument that puts the tolerance there.
    """
    # Two targets: pixel (20, 30) -> (639305, 6281795), pixel (40, 10) -> (639105, 6281595).
    scene = synthetic_scene(targets=[(20, 30), (40, 10)])
    ais = ais_slice(
        [("219000001", ACQUIRED_AT, 639_345.0, 6_281_795.0)],
        lengths={"219000001": 228.0},
    )

    detections = detect(scene, ais)

    assert detection_at(detections, 639_305.0, 6_281_795.0)["length_m"] == pytest.approx(228.0)
    # Nothing explained the other one, so there is no vessel whose size it could carry.
    assert pd.isna(detection_at(detections, 639_105.0, 6_281_595.0)["length_m"])


def test_a_vessel_whose_track_reaches_the_target_by_the_acquisition_is_not_dark() -> None:
    """The failure this level exists to remove, at the seam.

    Neither of this vessel's reports stands within the tolerance of the detection: one is 900 m
    west and the other 600 m east. Matched against either as it stands, a vessel that declared
    itself twice inside five minutes is published as an undeclared one. Interpolated along the
    track, it is exactly where the radar imaged it.
    """
    # Target at pixel (20, 30) -> (639305, 6281795). The vessel covers 1500 m in five minutes,
    # three fifths of them before the acquisition: 638405 + 0.6 * 1500 = 639305.
    scene = synthetic_scene(targets=[(20, 30)])
    ais = ais_slice(
        [
            ("219000001", ACQUIRED_AT - timedelta(minutes=3), 638_405.0, 6_281_795.0),
            ("219000001", ACQUIRED_AT + timedelta(minutes=2), 639_905.0, 6_281_795.0),
        ]
    )

    detections = detect(scene, ais)

    vessel = detection_at(detections, 639_305.0, 6_281_795.0)
    assert vessel["status"] == "matched"
    assert vessel["mmsi"] == "219000001"
    assert vessel["match_distance_m"] == pytest.approx(0.0, abs=1e-6)
    # The row says the position was built rather than observed, and how close the nearest real
    # report sits to the acquisition: "matched" here rests on an interpolation, and says so.
    assert vessel["position_basis"] == "interpolated"
    assert vessel["position_age_s"] == pytest.approx(120.0)


def test_a_declared_position_beyond_the_tolerance_leaves_the_detection_dark() -> None:
    scene = synthetic_scene(targets=[(20, 30)])
    # 250 m east of the target at (639305, 6281795).
    ais = ais_slice([("219000001", ACQUIRED_AT, 639_555.0, 6_281_795.0)])

    within = detect(scene, ais, tolerance_m=300.0)
    beyond = detect(scene, ais, tolerance_m=200.0)

    assert within["status"].iloc[0] == "matched"
    assert beyond["status"].iloc[0] == "dark"


def test_a_gap_in_the_scene_is_not_a_target(tmp_path: Path) -> None:
    """A real product has holes, and a hole is not dark water.

    Earth Engine writes masked pixels as a fill value and declares it as nodata. Read without
    honouring that declaration, the fill is just a number — and on a scene in dB, where the sea
    sits near -14 and the fill is 0, it is a number brighter than any vessel in the image. The
    first real Sentinel-1 scene run through this chain produced three "targets" of 72100, 38955
    and 36428 pixels that way: not a crash, not an obviously silly result, just a plausible
    count with the largest ships in Denmark in it.
    """
    path = tmp_path / "with-gaps.tif"
    sea = np.full((64, 64), -14.0, dtype=np.float32)
    sea[20:23, 30:33] = -2.0  # one genuine bright target
    sea[40:56, 8:24] = 0.0  # a hole in the product, brighter than the target if taken at face value

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        crs=WORKING_CRS,
        transform=Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y),
        nodata=0.0,
    ) as dataset:
        dataset.write(sea, 1)
        dataset.update_tags(ACQUIRED_AT=ACQUIRED_AT.isoformat())

    detections = run(
        scene=Scene.from_geotiff(path),
        ais=None,
        detector=BrightPixelDetector(threshold=-5.0),
        tiling=ONE_TILE,
        tolerance_m=TOLERANCE_M,
        max_gap=MAX_GAP,
    )

    assert len(detections) == 1
    # Centre of pixel (21, 31): 639000 + 31.5 * 10, 6282000 - 21.5 * 10.
    detection_at(detections, 639_315.0, 6_281_785.0)


def test_a_run_with_no_declarations_to_search_calls_nothing_dark() -> None:
    # "Dark" is a claim about what was searched, and with no AIS supplied nothing was. A layer
    # that said dark anyway would open in QGIS looking exactly like a sea full of undeclared
    # vessels — the most confident wrong answer this chain is capable of producing.
    scene = synthetic_scene(targets=[(20, 30)])

    detections = detect(scene, ais=None)

    assert detections["status"].iloc[0] == "unsearched"
    assert pd.isna(detections["mmsi"].iloc[0])
    # No radius either: a tolerance next to a detection nothing was compared against reads as a
    # search that happened and came back empty.
    assert pd.isna(detections["tolerance_m"].iloc[0])


def test_a_search_that_had_nothing_to_search_says_so_rather_than_reporting_dark_vessels() -> None:
    """The first real Danish slice this project ingested was empty, and this is why that matters.

    An AIS slice that held no vessel in the scene at the acquisition is a search that ran and
    came back empty, so every detection in it is honestly dark. It is also the one case where
    the honest word reads as its opposite: a layer of a hundred dark vessels over the Kattegat
    looks exactly like a finding, and nothing in it says that no ship declared itself there at
    all. The count of what the radius was applied to travels with the verdict, like the radius.
    """
    scene = synthetic_scene(targets=[(20, 30)])

    detections = detect(scene, ais=ais_slice([]))

    assert detections["status"].iloc[0] == "dark"
    assert detections["declarations_searched"].iloc[0] == 0
    # Still a search, unlike `ais=None`: the radius was applied, to nothing.
    assert detections["tolerance_m"].iloc[0] == pytest.approx(200.0)


def test_a_run_says_how_many_declared_positions_its_verdict_rests_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """And the number it says is the number in the layer.

    The verdict counts the vessels in the slice and the layer counts the positions the matching
    was run against. They are the same quantity by construction — one placed position per MMSI —
    and this is what stops them from drifting into two different answers to one question.
    """
    written = run_from_config(tmp_path)

    verdict = capsys.readouterr().out

    # Five declarations from five MMSIs in the fixture; the vessel reporting twice is placed once.
    assert "against 5 declared positions" in verdict
    assert written["declarations_searched"].eq(5).all()
    assert "dark by default rather than by evidence" not in verdict


def test_a_scene_with_no_detections_does_not_claim_nobody_declared_themselves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No detections is not the same as no declarations, and the verdict must not confuse them.

    Reading the count back off the layer meant reading it off a row that does not exist, and
    falling back to zero — so a run over a quiet scene would announce that no vessel declared
    itself, over a slice holding five of them. That is the wrong-but-plausible claim the count
    was added to prevent, reintroduced by the way it was reported.
    """
    run_from_config(
        tmp_path,
        CONFIG.replace("threshold: 0.5", "threshold: 2.0"),  # nothing is that bright
    )

    verdict = capsys.readouterr().out

    assert "0 detections" in verdict
    assert "against 5 declared positions" in verdict
    assert "dark by default rather than by evidence" not in verdict


def test_one_declared_position_cannot_explain_two_detections() -> None:
    # Two targets 60 m apart: pixel (20, 30) -> (639305, 6281795), (20, 36) -> (639365, 6281795).
    scene = synthetic_scene(targets=[(20, 30), (20, 36)])
    # One vessel between them: 20 m from the first, 40 m from the second, both inside tolerance.
    ais = ais_slice([("219000001", ACQUIRED_AT, 639_325.0, 6_281_795.0)])

    detections = detect(scene, ais)

    nearer = detection_at(detections, 639_305.0, 6_281_795.0)
    farther = detection_at(detections, 639_365.0, 6_281_795.0)
    assert nearer["status"] == "matched"
    assert nearer["mmsi"] == "219000001"
    # Two hulls that close together are two vessels, and only one of them declared itself.
    assert farther["status"] == "dark"
    assert pd.isna(farther["mmsi"])


def test_a_single_command_turns_a_config_into_a_georeferenced_layer(tmp_path: Path) -> None:
    written = run_from_config(tmp_path)

    assert written.crs == WORKING_CRS
    assert len(written) == 5

    # Ground coordinates of the five synthetic targets, worked out from the transform that
    # `write_synthetic_inputs` documents.
    declared_a = detection_at(written, 639_805.0, 6_281_395.0)
    declared_b = detection_at(written, 641_005.0, 6_280_795.0)
    undeclared = detection_at(written, 641_205.0, 6_281_695.0)
    on_the_seam = detection_at(written, 640_285.0, 6_280_715.0)
    moving = detection_at(written, 639_605.0, 6_279_995.0)

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

    # The vessel under way: 900 m west of the target three minutes before the acquisition, 600 m
    # east of it two minutes after, and never within the tolerance of it at either. Matched
    # against a report as it stands — which is what the shipped run did before this level — it is
    # published as a dark vessel that was never there.
    assert moving["status"] == "matched"
    assert moving["mmsi"] == "219000005"
    assert moving["position_basis"] == "interpolated"

    # The one honest dark detection: the vessel that declared itself 11 km away.
    assert (written["status"] == "dark").sum() == 1

    # Every other match rests on a report taken as it stands — those vessels reported only
    # before the acquisition, so there is nothing to interpolate between and the layer says so.
    assert declared_a["position_basis"] == "reported"
    assert declared_a["position_age_s"] == pytest.approx(180.0)


def test_a_run_says_how_many_of_its_matches_rest_on_an_interpolated_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interpolated position is a construction, and the run's verdict should not hide it.

    The layer carries `position_basis` per row, which is the durable form of the claim. This is
    the same claim on the way past: someone who has just run the command and not yet opened the
    output should be told how much of the answer is built rather than observed.
    """
    run_from_config(tmp_path)

    verdict = capsys.readouterr().out
    # Four matches: the moving vessel is placed by interpolation, the other three reported only
    # before the acquisition and are matched against a report taken as it stands.
    assert "1 on a position interpolated" in verdict
    assert "3 on a report taken as it stands" in verdict


def test_the_gap_a_config_allows_decides_what_is_interpolated_across(tmp_path: Path) -> None:
    """The knob has to reach the matching, and be visible in the answer when it does.

    The moving vessel's two reports are five minutes apart. Told that a straight line may span
    no more than a minute, the run must refuse to draw one and fall back to the nearest report —
    600 m from the target, so the vessel comes back dark. That is the wrong answer about the
    world and the right answer about the evidence: with reports that far apart, nothing here
    knows where the vessel was.
    """
    written = run_from_config(
        tmp_path, CONFIG.replace("interpolation_max_gap_s: 600", "interpolation_max_gap_s: 60")
    )

    moving = detection_at(written, 639_605.0, 6_279_995.0)
    assert moving["status"] == "dark"
    assert (written["status"] == "dark").sum() == 2


@pytest.mark.parametrize(
    "shipped", sorted(CONFIGS.rglob("*.yaml")), ids=lambda path: str(path.name)
)
def test_every_shipped_config_names_the_fusion_settings_a_run_needs(shipped: Path) -> None:
    """The same gap `export_request_from` exists to close, on the settings fusion reads.

    Every test above writes its own config, so the shipped files are the ones nothing in the
    suite executes. `configs/kattegat-lane.yaml` is worse than that: running it needs Earth Engine
    credentials, so a missing key there surfaces only to someone who has already authenticated
    and waited. Both go through the command's own parsing here instead.

    Every file under `configs/` is covered, rungs of the ladder included — they are one directory
    down, and `rglob` rather than `glob` is what keeps them from dropping out of this test in
    silence. Files that are not runs have to say what they are instead.
    """
    config = load_config(shipped)
    if "run" not in config:
        assert "survey" in config or "training" in config or "ladder" in config, (
            f"{shipped.name} describes neither a run, a survey, a training nor a ladder"
        )
        return

    settings = fusion_settings_from(config)

    assert settings["tolerance_m"] > 0.0
    assert settings["max_gap"] > timedelta(0)


def test_the_shipped_config_still_cuts_the_synthetic_scene_across_a_target() -> None:
    """The claim the README makes about the shipped run, pinned.

    Every test above builds its own config, so `configs/pipeline.yaml` is the one file here that
    nothing executes: widen its tiling until the synthetic scene fits in a single tile and the
    suite stays green while the shipped command quietly stops crossing a seam at all. The
    fixture's boundary target is only a boundary target with respect to a tiling, and this is
    the tiling a reader actually runs.
    """
    tiling = Tiling(**yaml.safe_load(SHIPPED_CONFIG.read_text())["tiling"])
    row, col = BOUNDARY_TARGET

    saw_it = [
        tile
        for tile in tiling.tiles((SIZE_PX, SIZE_PX))
        if tile.rows.start <= row < tile.rows.stop and tile.cols.start <= col < tile.cols.stop
    ]

    assert len(saw_it) > 1, (
        f"the shipped tiling gives target {BOUNDARY_TARGET} to {len(saw_it)} tile(s); with fewer "
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


# The trained detector's half of a run config. Its window is a placeholder with round numbers:
# what the shipped one is, and how it was arrived at, belongs in configs/ and docs/decisions.md.
TRAINED_RUN = {
    "detector": "trained",
    "trained": {
        "checkpoint": "../models/epoch-012.pt",
        "tile_px": 800,
        "anchor_sizes": [[32], [64], [128], [256], [512]],
        "score_threshold": 0.75,
        "stretch": {"floor_db": -30.0, "ceiling_db": 10.0, "sea_db": -21.84},
    },
}


def test_a_trained_run_is_read_without_the_framework_installed(tmp_path):
    """Every key of a shipped config goes through a function a test can call, or it becomes the
    one key nothing in the suite ever parses — the argument `export_request_from` already makes.
    Here it is sharper: the framework this config names is an optional extra, so a test that had
    to import torch to check a spelling would not run in CI at all.
    """
    request = trained_request_from(TRAINED_RUN, tmp_path)

    assert request["checkpoint"] == (tmp_path / "../models/epoch-012.pt").resolve()
    assert request["tile_px"] == 800
    assert request["anchor_sizes"] == ((32,), (64,), (128,), (256,), (512,))
    assert request["score_threshold"] == 0.75
    assert request["stretch"].floor_db == -30.0
    assert request["stretch"].sea == pytest.approx(0.204, abs=1e-3)


def test_a_tiling_the_model_was_not_built_for_is_refused():
    """Torchvision would resize each tile to the size the model declares, silently, and
    resampling radar amplitude changes what the detector sees. The same refusal
    `_check_working_crs` makes about a reprojection, for the same reason."""
    with pytest.raises(ValueError, match="800"):
        check_tile_size(TRAINED_RUN, Tiling(size_px=512, overlap_px=64))


def test_the_tiling_the_model_was_built_for_is_accepted():
    check_tile_size(TRAINED_RUN, Tiling(size_px=800, overlap_px=64))


def test_the_stand_in_is_not_asked_what_tile_size_it_wants():
    """The threshold detector has no opinion about tile size, so it is not consulted."""
    check_tile_size(
        {"detector": "bright-pixel", "threshold": 0.5}, Tiling(size_px=144, overlap_px=32)
    )


def test_the_shipped_real_config_names_a_tiling_its_detector_can_run_at():
    """The one config in this package nothing else in the suite runs — every other stage of it
    needs Earth Engine credentials before it can even fail."""
    path = Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml"
    config = yaml.safe_load(path.read_text())

    check_tile_size(
        config["run"],
        Tiling(
            size_px=int(config["tiling"]["size_px"]),
            overlap_px=int(config["tiling"]["overlap_px"]),
        ),
    )


class Brightness:
    """A deterministic stand-in embedder: how bright the crop is, and how much it varies.

    The same thing `BrightPixelDetector` is to the trained detector — something that satisfies the
    contract, needs no weights and no framework, and lets the chain be exercised through the
    optional stage without any of it. It describes a crop honestly enough to be a representation
    and badly enough that nobody could mistake it for one.
    """

    crop_px = 8
    margin_px = 2

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        if len(crops) == 0:
            return np.empty((0, 2), dtype=np.float32)
        flat = crops.reshape(len(crops), -1)
        return np.stack([np.nanmean(flat, axis=1), np.nanstd(flat, axis=1)], axis=1)


def test_the_chain_runs_end_to_end_with_the_embedding_stage_disabled() -> None:
    """The claim the ticket rests on, and the reason the parameter defaults to None.

    A representation of what the chain found is a second question asked of the same detections.
    A run that never asks it must be unaffected by the stage existing at all — not merely able to
    run, but returning what it returned before the stage was written.
    """
    scene = synthetic_scene(targets=[(20, 30), (40, 10)])
    ais = ais_slice([("219000001", ACQUIRED_AT, 639_345.0, 6_281_795.0)])

    without = detect(scene, ais)

    assert vectors_of(without).shape == (2, 0)
    assert not [name for name in without.columns if name.startswith("e0")]


def test_the_embedding_stage_adds_columns_and_changes_nothing_else() -> None:
    scene = synthetic_scene(targets=[(20, 30), (40, 10)])
    ais = ais_slice([("219000001", ACQUIRED_AT, 639_345.0, 6_281_795.0)])

    without = detect(scene, ais)
    with_vectors = run(
        scene=scene,
        ais=ais,
        detector=BrightPixelDetector(threshold=0.5),
        tiling=ONE_TILE,
        tolerance_m=TOLERANCE_M,
        max_gap=MAX_GAP,
        embedder=Brightness(),
    )

    # Every column of the run without the stage, unchanged, in the same order.
    pd.testing.assert_frame_equal(with_vectors[without.columns], without)
    assert list(with_vectors.columns[-2:]) == columns(2)
    # The bright target and the dark one do not describe alike; a stage that attached the same
    # vector to every row would pass every assertion above.
    assert vectors_of(with_vectors).shape == (2, 2)


def test_a_detection_is_described_by_the_pixels_around_it_and_not_by_its_neighbour() -> None:
    """Attached by position, and the position has to be the right one.

    Two targets, one of them beside a second bright patch that no detection stands on. If the
    crops were cut in any order but the detections' own, the vectors would swap and the layer
    would still open, still carry two rows, and describe each vessel with the other's pixels.
    """
    image = np.zeros((64, 64), dtype=np.float32)
    image[19:22, 29:32] = 1.0  # a lone target
    image[39:42, 9:12] = 1.0  # a target with company
    image[39:42, 13:16] = 1.0
    scene = Scene(
        image=image,
        transform=Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y),
        crs=WORKING_CRS,
        acquired_at=ACQUIRED_AT,
    )

    described = run(
        scene=scene,
        ais=None,
        detector=BrightPixelDetector(threshold=0.5),
        tiling=ONE_TILE,
        tolerance_m=TOLERANCE_M,
        max_gap=MAX_GAP,
        embedder=Brightness(),
    )

    lonely = detection_at(described, 639_305.0, 6_281_795.0)
    crowded = detection_at(described, 639_105.0, 6_281_595.0)
    # The crowded crop holds more bright pixels than the lonely one, whichever way round the
    # rows happen to be sorted.
    assert crowded["e00"] > lonely["e00"]


def test_a_scene_with_no_detections_still_writes_the_columns(tmp_path: Path) -> None:
    """Otherwise a quiet acquisition writes a layer with a different schema from the one beside
    it, and the archive of layers stops stacking."""
    scene = synthetic_scene(targets=[])

    described = run(
        scene=scene,
        ais=None,
        detector=BrightPixelDetector(threshold=0.5),
        tiling=ONE_TILE,
        tolerance_m=TOLERANCE_M,
        max_gap=MAX_GAP,
        embedder=Brightness(),
    )

    assert len(described) == 0
    assert list(described.columns[-2:]) == columns(2)


EMBEDDING_CONFIG = CONFIGS / "embeddings.yaml"


def test_the_shipped_embedding_config_is_read_without_the_framework_installed() -> None:
    """The argument `trained_request_from` makes, on the level whose every stage needs the extra.

    Three of the four commands of this level need either Earth Engine credentials or torch, and
    the fourth needs an archive that takes an hour to build. Nothing in this suite runs any of
    them, so the shipped config is checked here, key by key, by the same functions the commands
    parse it with.
    """
    config = load_config(EMBEDDING_CONFIG)
    relative_to = EMBEDDING_CONFIG.parent

    archive = archive_request_from(config, relative_to)
    embedding = embedding_request_from(config, relative_to)

    assert archive["window"].start < archive["window"].end
    assert len(archive["boxes"]) >= 1
    for name, box in archive["boxes"].items():
        assert box.west < box.east and box.south < box.north, f"{name} is not a rectangle"
    # The archive is cut at an operating point of its own, and a lower one: a representation
    # fitted only on the objects the detector was certain about has never been shown the others.
    assert archive["score_threshold"] < float(config["run"]["trained"]["score_threshold"])
    assert embedding["enabled"] is True
    assert embedding["crop_px"] > 0 and embedding["margin_px"] > 0
    assert embedding["speckle"] is not None and embedding["speckle"].looks > 0
    assert embedding["schedule"]["batch_size"] >= 2
    assert embedding["retrieval"]["neighbours"] >= 1


def test_the_shipped_embedding_config_cuts_crops_the_scene_can_hold() -> None:
    """A crop wider than a tile is a crop the chain can never cut whole, and it would come back
    padded with holes on every detection near a tile edge rather than on the few at the scene's."""
    config = load_config(EMBEDDING_CONFIG)
    embedding = embedding_request_from(config, EMBEDDING_CONFIG.parent)

    assert embedding["crop_px"] + 2 * embedding["margin_px"] < int(config["tiling"]["size_px"])


def test_a_config_that_never_heard_of_the_embedding_stage_runs_without_one() -> None:
    """Every config in this project written before this level, and the synthetic demo since."""
    from darkvessel.cli import _embedder_from

    assert _embedder_from(load_config(SHIPPED_CONFIG), CONFIGS) is None
    assert (
        _embedder_from({"embedding": {"enabled": False, "encoder": "nowhere.pt"}}, CONFIGS) is None
    )
