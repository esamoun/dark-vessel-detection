"""Selecting and fetching a real Sentinel-1 scene, with the catalogue injected.

This is the one stage of the chain that has to talk to a network, and the network is the part
that cannot be asserted. So the catalogue arrives as a parameter — the same trick that lets the
pipeline run without a detector — and everything around it is tested here with a fake that hands
back canned scenes and canned bytes.

What is deliberately *not* tested: whether Earth Engine's filters select what we think they
select. That claim can only be made by running against Earth Engine, and it is recorded in the
README as a manual check rather than pretended at here.

What is tested is everything that stays wrong silently: the acquisition time surviving the trip
from the catalogue into the file the pipeline reads, the georeferencing arriving untouched, a
deterministic choice among candidate scenes, and a request too large to fetch being refused
before it is sent rather than half-answered.
"""

import importlib.util
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from affine import Affine
from rasterio.io import MemoryFile

from darkvessel.cli import export_request_from
from darkvessel.data.gee_export import (
    Bounds,
    Catalogue,
    DateWindow,
    SceneRef,
    earth_engine,
    export_scene,
)
from darkvessel.data.scene import Scene

WORKING_CRS = "EPSG:25832"
POLARISATIONS = ("VV", "VH")

SHIPPED_EXPORT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "anholt.yaml"

# The Anholt wind farm and the water around it: about 15 km on a side.
ANHOLT = Bounds(west=11.15, south=56.58, east=11.40, north=56.71)
WINDOW = DateWindow(
    start=datetime(2026, 3, 1, tzinfo=UTC),
    end=datetime(2026, 3, 15, tzinfo=UTC),
)

# A transform and shape for the canned bytes. Small, but real georeferencing: the point of the
# fixture is that whatever Earth Engine put in the file comes out of ours unchanged.
FETCHED_TRANSFORM = Affine(10.0, 0.0, 640_000.0, 0.0, -10.0, 6_280_000.0)
FETCHED_SHAPE = (48, 64)


def scene_ref(scene_id: str, acquired_at: datetime) -> SceneRef:
    return SceneRef(
        id=scene_id,
        acquired_at=acquired_at,
        polarisations=POLARISATIONS,
        orbit_pass="DESCENDING",
    )


def geotiff_bytes() -> bytes:
    """What the catalogue hands back: a georeferenced GeoTIFF with no acquisition time in it."""
    image = np.zeros(FETCHED_SHAPE, dtype=np.float32)
    image[20:23, 30:33] = 1.0

    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=FETCHED_SHAPE[0],
            width=FETCHED_SHAPE[1],
            count=1,
            dtype="float32",
            crs=WORKING_CRS,
            transform=FETCHED_TRANSFORM,
        ) as dataset:
            dataset.write(image, 1)
        return memory.read()


@dataclass
class FakeCatalogue(Catalogue):
    """Earth Engine, minus Earth Engine.

    It filters nothing — filtering is what the real catalogue does server-side, and a fake that
    reimplemented it would only ever agree with itself. It hands back what it was given and
    records what it was asked for.
    """

    scenes: list[SceneRef]
    fetches: list[tuple[SceneRef, Bounds]] = field(default_factory=list)
    searches: list[tuple[Bounds, DateWindow]] = field(default_factory=list)

    def search(
        self, area: Bounds, window: DateWindow, polarisations: tuple[str, ...]
    ) -> list[SceneRef]:
        self.searches.append((area, window))
        return self.scenes

    def geotiff(
        self,
        scene: SceneRef,
        area: Bounds,
        polarisations: tuple[str, ...],
        crs: str,
        resolution_m: float,
    ) -> bytes:
        self.fetches.append((scene, area))
        return geotiff_bytes()


def export(catalogue: Catalogue, path: Path, area: Bounds = ANHOLT) -> SceneRef:
    return export_scene(
        catalogue=catalogue,
        area=area,
        window=WINDOW,
        polarisations=POLARISATIONS,
        crs=WORKING_CRS,
        resolution_m=10.0,
        path=path,
    )


def test_the_exported_scene_carries_the_acquisition_time_the_catalogue_reported(
    tmp_path: Path,
) -> None:
    # The pixels come from Earth Engine and the timestamp comes from its metadata; they meet for
    # the first time in the written file. A drift of an hour here is a drift of 22 km of vessel
    # track at the fusion stage, and nothing downstream can detect it.
    acquired_at = datetime(2026, 3, 8, 5, 27, 43, tzinfo=UTC)
    catalogue = FakeCatalogue(scenes=[scene_ref("S1A_IW_GRDH_20260308", acquired_at)])
    path = tmp_path / "anholt.tif"

    exported = export(catalogue, path)

    assert exported.acquired_at == acquired_at
    assert Scene.from_geotiff(path).acquired_at == acquired_at


def test_the_exported_scene_records_which_polarisations_it_was_built_from(tmp_path: Path) -> None:
    # Which polarisations went in decides what the amplitude in the file even means. Left out of
    # the file, it survives only in whatever command someone happened to run.
    catalogue = FakeCatalogue(scenes=[scene_ref("S1A_IW_GRDH_20260308", WINDOW.start)])
    path = tmp_path / "anholt.tif"

    export(catalogue, path)

    with rasterio.open(path) as dataset:
        tags = dataset.tags()
    assert tags["POLARISATIONS"] == "VV,VH"
    assert tags["SCENE_ID"] == "S1A_IW_GRDH_20260308"
    assert tags["ORBIT_PASS"] == "DESCENDING"


def test_the_georeferencing_earth_engine_produced_is_left_exactly_as_it_arrived(
    tmp_path: Path,
) -> None:
    # The one thing this module must not do is have an opinion about where the pixels are. It
    # adds metadata to the fetched file; a transform rebuilt from a bounding box and a pixel size
    # would look entirely reasonable and put every detection somewhere else.
    catalogue = FakeCatalogue(scenes=[scene_ref("S1A_IW_GRDH_20260308", WINDOW.start)])
    path = tmp_path / "anholt.tif"

    export(catalogue, path)

    with rasterio.open(path) as dataset:
        assert dataset.transform == FETCHED_TRANSFORM
        assert dataset.crs.to_string() == WORKING_CRS
        assert dataset.shape == FETCHED_SHAPE


def test_the_earliest_acquisition_in_the_window_is_the_one_taken(tmp_path: Path) -> None:
    # Not whichever the catalogue happened to list first: a run has to be repeatable, and "the
    # scene I got that day" is not a description anyone else can act on.
    first = scene_ref("S1A_IW_GRDH_20260304", WINDOW.start + timedelta(days=3))
    later = scene_ref("S1A_IW_GRDH_20260310", WINDOW.start + timedelta(days=9))
    catalogue = FakeCatalogue(scenes=[later, first])

    exported = export(catalogue, tmp_path / "anholt.tif")

    assert exported.id == first.id
    assert [fetched.id for fetched, _ in catalogue.fetches] == [first.id]


def test_a_window_with_no_acquisition_is_an_error_naming_what_was_searched(
    tmp_path: Path,
) -> None:
    # An empty result written out as an empty file is the failure this refuses: the next stage
    # would report zero detections over the area, which reads exactly like a quiet sea.
    catalogue = FakeCatalogue(scenes=[])

    with pytest.raises(ValueError, match="no COPERNICUS/S1_GRD acquisition covers"):
        export(catalogue, tmp_path / "anholt.tif")

    assert catalogue.fetches == []


def test_an_area_too_large_to_come_back_in_one_response_is_refused_before_it_is_sent(
    tmp_path: Path,
) -> None:
    # Two degrees square at 10 m is some 900 MB in two polarisations — a whole GRD product by
    # another name. Earth Engine refuses it too, but only after the wait, and its message does
    # not say which of the three numbers to change.
    too_large = Bounds(west=11.0, south=56.0, east=13.0, north=58.0)
    catalogue = FakeCatalogue(scenes=[scene_ref("S1A_IW_GRDH_20260308", WINDOW.start)])

    with pytest.raises(ValueError, match="past the 34 MB a direct download returns"):
        export(catalogue, tmp_path / "anholt.tif", area=too_large)

    assert catalogue.searches == []
    assert catalogue.fetches == []


@pytest.mark.skipif(
    importlib.util.find_spec("ee") is not None,
    reason="the gee extra is installed here, so the missing-extra path cannot be reached",
)
def test_a_missing_gee_extra_is_answered_with_the_command_that_installs_it() -> None:
    # Not having Earth Engine installed is the normal state of this package, not a broken one:
    # the chain is meant to run with no network. Someone reaching the one command that does need
    # it should be told which extra to install, not handed an import error.
    with pytest.raises(ModuleNotFoundError, match=r'pip install -e "\.\[gee\]"'):
        earth_engine()


def test_the_shipped_export_config_describes_a_request_that_can_be_answered(
    tmp_path: Path,
) -> None:
    """`configs/anholt.yaml`, run through the command's own parsing, minus Earth Engine.

    This is the one config in the package that needs credentials, so it is the one whose faults
    would otherwise be found by a person who had already authenticated and waited. Everything up
    to the network is exercised here in under a second: the keys exist and are spelled as the
    command expects, the bounds are a real rectangle, the window carries a timezone, and the area
    is small enough to come back in one response.
    """
    config = yaml.safe_load(SHIPPED_EXPORT_CONFIG.read_text())
    catalogue = FakeCatalogue(scenes=[scene_ref("S1A_IW_GRDH_20260703", WINDOW.start)])

    request = export_request_from(config, SHIPPED_EXPORT_CONFIG.parent)
    export_scene(catalogue=catalogue, **{**request, "path": tmp_path / "anholt.tif"})

    assert len(catalogue.fetches) == 1
    # The scene the export writes is the scene the run reads: one config, one file, no third
    # place where the path is spelled out again.
    assert request["path"] == (SHIPPED_EXPORT_CONFIG.parent / config["run"]["scene"]).resolve()
