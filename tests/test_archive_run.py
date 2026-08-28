"""The chain run over a whole archive of acquisitions, into one layer.

One scene carries a handful of detections. The spatial analysis this feeds asks where undeclared
traffic concentrates, and a distribution cannot be read off a handful — so the same chain, at the
same operating point, runs over every acquisition of the same box and the detections accumulate.

What is tested here is what accumulation makes newly possible to get wrong. A merged layer can
lose track of which acquisition a row came from; it can be built from an ingestion that stopped
halfway and quietly score scenes against no declarations at all; and it can average away the one
scene-level property the operating point is sensitive to. None of those fail loudly, and all
three would be read as findings about undeclared traffic.

The network is not here. `archive-ais` is what needs it, and it is tested against a fake archive
in `test_ais.py`; this file starts from slices already on disk.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely import Point

from darkvessel.cli import main
from darkvessel.data.ais import write_ais
from darkvessel.data.provenance import SCENE, SEA_LEVEL

WORKING_CRS = "EPSG:25832"
PIXEL_M = 10.0
ORIGIN_X, ORIGIN_Y = 600_000.0, 6_270_000.0
SIZE_PX = 64

# Two acquisitions of the same box, a fortnight apart. The names are Sentinel-1 shaped because
# the provenance column is the product's file stem and a reader is meant to recognise it.
FIRST = ("S1A_IW_GRDH_1SDV_20260602T054028", datetime(2026, 6, 2, 5, 40, 28, tzinfo=UTC))
SECOND = ("S1C_IW_GRDH_1SDV_20260616T053134", datetime(2026, 6, 16, 5, 31, 34, tzinfo=UTC))

CONFIG = (
    "area:\n"
    f"  crs: {WORKING_CRS}\n"
    "  bounds: {west: 11.0, south: 57.55, east: 11.3, north: 57.7}\n"
    "tiling:\n"
    "  size_px: 64\n"
    "  overlap_px: 16\n"
    "ais:\n"
    "  window_s: 900\n"
    "  margin_m: 5000\n"
    "  max_speed_kn: 60\n"
    "  out: unused.csv\n"
    "archive:\n"
    "  scenes: scenes\n"
    "  ais: slices\n"
    "  detections: archive.gpkg\n"
    "run:\n"
    "  scene: scenes/unused.tif\n"
    "  ais: null\n"
    "  output: detections.gpkg\n"
    "  detector: bright-pixel\n"
    "  threshold: -5.0\n"
    "fusion:\n"
    "  match_tolerance_m: 200\n"
    "  interpolation_max_gap_s: 600\n"
)


def write_scene(path: Path, acquired_at: datetime, sea_db: float, targets: list) -> None:
    """A flat sea at `sea_db` with a bright 3x3 target on each (row, col)."""
    image = np.full((SIZE_PX, SIZE_PX), np.float32(sea_db), dtype=np.float32)
    for row, col in targets:
        image[row - 1 : row + 2, col - 1 : col + 2] = np.float32(-2.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE_PX,
        width=SIZE_PX,
        count=1,
        dtype="float32",
        crs=WORKING_CRS,
        transform=Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y),
    ) as dataset:
        dataset.write(image, 1)
        dataset.update_tags(ACQUIRED_AT=acquired_at.isoformat())


def write_slice(path: Path, acquired_at: datetime, at: tuple[float, float]) -> None:
    """One declared vessel, standing still, reporting either side of the acquisition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y = at
    reports = gpd.GeoDataFrame(
        {
            "mmsi": ["219000001", "219000001"],
            "timestamp": [acquired_at - timedelta(minutes=2), acquired_at + timedelta(minutes=2)],
            "length_m": [120.0, 120.0],
        },
        geometry=[Point(x, y), Point(x, y)],
        crs=WORKING_CRS,
    ).to_crs("EPSG:4326")
    write_ais(reports, path)


def archive(directory: Path, seas: tuple[float, float] = (-14.0, -14.0)) -> Path:
    """Two acquisitions, each with one declared vessel and one that declared nothing."""
    config = directory / "archive.yaml"
    config.write_text(CONFIG)

    declared = (ORIGIN_X + 30 * PIXEL_M + PIXEL_M / 2, ORIGIN_Y - 20 * PIXEL_M - PIXEL_M / 2)
    for (name, acquired_at), sea_db in zip((FIRST, SECOND), seas, strict=True):
        write_scene(
            directory / "scenes" / f"{name}.tif",
            acquired_at,
            sea_db,
            targets=[(20, 30), (40, 10)],
        )
        write_slice(directory / "slices" / f"{name}.csv", acquired_at, declared)
    return config


def test_every_acquisition_of_the_archive_reaches_one_layer(tmp_path: Path) -> None:
    config = archive(tmp_path)

    assert main(["archive-run", "--config", str(config)]) == 0

    merged = gpd.read_file(tmp_path / "archive.gpkg")
    assert len(merged) == 4  # two targets on each of two acquisitions
    assert set(merged["status"]) == {"matched", "dark"}


def test_a_detection_in_the_merged_layer_says_which_acquisition_it_came_out_of(
    tmp_path: Path,
) -> None:
    """The provenance an accumulated layer cannot be read without.

    Fifty scenes stacked, a row is a coordinate and a verdict and nothing else unless it carries
    the acquisition. A dark detection nobody can trace back to a product cannot be checked
    against the image, and cannot be dropped when its scene turns out to have a problem.
    """
    config = archive(tmp_path)

    main(["archive-run", "--config", str(config)])

    merged = gpd.read_file(tmp_path / "archive.gpkg")
    assert set(merged[SCENE]) == {FIRST[0], SECOND[0]}
    assert merged.groupby(SCENE).size().tolist() == [2, 2]


def test_the_sea_each_detection_was_scored_against_survives_the_merge(tmp_path: Path) -> None:
    """The confound, still legible after fifty scenes are stacked.

    The window between decibels and amplitude is fixed and calibrated on one scene's sea, so a
    scene whose sea sits away from that is scored at an operating point nobody chose. Merged
    without this column the archive would look like one homogeneous population, and a count of
    dark detections that tracked the weather would read as a finding about undeclared traffic.
    """
    config = archive(tmp_path, seas=(-14.0, -8.0))

    main(["archive-run", "--config", str(config)])

    merged = gpd.read_file(tmp_path / "archive.gpkg")
    by_scene = merged.groupby(SCENE)[SEA_LEVEL].first()
    assert by_scene[FIRST[0]] == pytest.approx(-14.0)
    assert by_scene[SECOND[0]] == pytest.approx(-8.0)


def test_the_declarations_reported_are_counted_per_scene_not_per_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`declarations_searched` is constant down a scene's rows, so summing the column is wrong.

    Two acquisitions, one declared vessel each: the run searched two declarations. Added up off
    the four detection rows it would report four, and the number would grow with how many targets
    the detector happened to find rather than with how many vessels declared themselves.
    """
    config = archive(tmp_path)

    main(["archive-run", "--config", str(config)])

    assert "against 2 declared positions" in capsys.readouterr().out


def test_an_archive_run_on_an_interrupted_ingestion_is_refused(tmp_path: Path) -> None:
    """The failure that would look exactly like a finding.

    `run` allows `ais: null` and marks what comes out `unsearched`, because a scene before the
    ingestion level genuinely has nothing to search. Here a missing slice means the 21 GB
    download stopped early, and a scene scored against no declarations contributes detections
    that are dark by default rather than by evidence — into the one layer built to count them.
    """
    config = archive(tmp_path)
    (tmp_path / "slices" / f"{SECOND[0]}.csv").unlink()

    with pytest.raises(FileNotFoundError, match="have no declarations"):
        main(["archive-run", "--config", str(config)])


def test_an_archive_with_no_acquisitions_in_it_is_refused(tmp_path: Path) -> None:
    config = archive(tmp_path)
    for scene in (tmp_path / "scenes").glob("*.tif"):
        scene.unlink()

    with pytest.raises(FileNotFoundError, match="no acquisitions"):
        main(["archive-run", "--config", str(config)])
