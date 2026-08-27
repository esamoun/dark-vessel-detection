"""Contextual variables attached to detections, with the catalogue injected.

The same seam as `test_export.py`, for the same reason: everything that happens on Google's side
of a credentialed connection cannot be asserted here, so the layers arrive as a parameter and a
fake hands back canned values. What is tested is everything on this side of that boundary — the
frame the sampler is asked in, the row each answer lands on, and the one property this level
exists to keep, that a value nobody could sample is missing rather than zero.

What is deliberately *not* tested: whether `NOAA/NGDC/ETOPO1` is the depth we think it is, or
whether the fishing-effort collection covers this water. Those claims can only be made by running
against Earth Engine, and they are recorded in `docs/decisions.md` as unrun rather than pretended
at here.
"""

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from pyproj import Transformer
from shapely import Point

from darkvessel.cli import context_request_from, main
from darkvessel.context.gee_layers import (
    CONTEXT,
    DEGREES,
    HIGH_SEAS,
    UNAVAILABLE,
    Context,
    LayerSources,
    attach,
    coverage,
    without_context,
)
from darkvessel.detect.geo import DETECTIONS_LAYER, write_detections
from darkvessel.fusion.match import DARK

WORKING_CRS = "EPSG:25832"

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml"

# Three positions in the northern Kattegat, in the CRS the chain works in. Nothing about them
# matters except that they are metres rather than degrees and that they are far enough apart to
# tell one row from another.
POSITIONS = ((620_000.0, 6_390_000.0), (622_500.0, 6_392_000.0), (625_000.0, 6_394_000.0))


@dataclass
class FakeLayers:
    """Canned answers, and a record of what it was asked."""

    answers: list[Context]
    asked: list[tuple[float, float]] = field(default_factory=list)

    def sample(self, points):
        self.asked = list(points)
        return self.answers


def detections_at(*positions: tuple[float, float]) -> gpd.GeoDataFrame:
    """Detections as the chain reports them, before anything contextual is asked of them."""
    return gpd.GeoDataFrame(
        {"score": [0.95] * len(positions), "status": [DARK] * len(positions)},
        geometry=[Point(x, y) for x, y in positions],
        crs=WORKING_CRS,
    )


def context_config() -> dict:
    return {
        "context": {
            "shore": {"asset": "USDOS/LSIB_SIMPLE/2017", "search_radius_m": 200000},
            "depth": {"asset": "NOAA/NGDC/ETOPO1", "band": "bedrock"},
            "eez": {"asset": None, "property": "SOVEREIGN1"},
            "effort": {
                "asset": "GFW/GFF/V1/fishing_hours",
                "bands": ["trawlers", "purse_seines"],
                "start": "2016-01-01",
                "end": "2017-01-01",
            },
            "scale_m": 1000,
        }
    }


def test_every_detection_carries_every_contextual_column() -> None:
    layers = FakeLayers([Context(1.0, -20.0, "Denmark", 3.0)] * 3)

    carried = attach(detections_at(*POSITIONS), layers)

    assert list(CONTEXT) == [name for name in CONTEXT if name in carried.columns]
    assert len(carried) == 3


def test_each_answer_lands_on_the_detection_it_was_sampled_at() -> None:
    """By position, which is the only correspondence there is — so it is checked, not trusted."""
    layers = FakeLayers(
        [
            Context(1_000.0, -18.0, "Denmark", 0.0),
            Context(2_000.0, -24.0, "Sweden", 1.5),
            Context(3_000.0, -31.0, HIGH_SEAS, 12.25),
        ]
    )

    carried = attach(detections_at(*POSITIONS), layers)

    assert list(carried["distance_to_shore_m"]) == [1_000.0, 2_000.0, 3_000.0]
    assert list(carried["depth_m"]) == [-18.0, -24.0, -31.0]
    assert list(carried["eez"]) == ["Denmark", "Sweden", HIGH_SEAS]
    assert list(carried["fishing_hours"]) == [0.0, 1.5, 12.25]


def test_a_value_of_zero_is_kept_and_a_missing_one_is_not_turned_into_zero() -> None:
    """The one property this level exists to keep.

    No fishing effort recorded at a position is a fact about that water. A layer that could not
    answer is not, and the two are the same number if either is allowed to be filled in.
    """
    layers = FakeLayers(
        [
            Context(0.0, 0.0, HIGH_SEAS, 0.0),
            Context(None, None, None, None),
        ]
    )

    carried = attach(detections_at(*POSITIONS[:2]), layers)

    assert list(carried.loc[0, list(CONTEXT)]) == [0.0, 0.0, HIGH_SEAS, 0.0]
    assert np.isnan(carried.loc[1, "distance_to_shore_m"])
    assert np.isnan(carried.loc[1, "depth_m"])
    assert np.isnan(carried.loc[1, "fishing_hours"])
    assert carried.loc[1, "eez"] == UNAVAILABLE


def test_no_eez_at_a_position_is_the_high_seas_and_not_an_unanswered_layer() -> None:
    """Two different statements, and the layer has to keep them apart."""
    layers = FakeLayers([Context(1.0, -20.0, HIGH_SEAS, 0.0), Context(1.0, -20.0, None, 0.0)])

    carried = attach(detections_at(*POSITIONS[:2]), layers)

    assert carried.loc[0, "eez"] == HIGH_SEAS
    assert carried.loc[1, "eez"] == UNAVAILABLE
    assert HIGH_SEAS != UNAVAILABLE


def test_the_sampler_is_asked_in_degrees_however_the_scene_is_projected() -> None:
    """Earth Engine's catalogue is addressed in lon/lat; the chain works in metres.

    A UTM easting handed over as a longitude is not an error anything downstream could see: it
    samples a position off the coast of Africa and returns a number.
    """
    layers = FakeLayers([Context(1.0, -20.0, "Denmark", 0.0)] * 3)

    attach(detections_at(*POSITIONS), layers)

    into_degrees = Transformer.from_crs(WORKING_CRS, DEGREES, always_xy=True)
    expected = [into_degrees.transform(x, y) for x, y in POSITIONS]
    assert layers.asked == pytest.approx(expected)
    assert all(8.0 < lon < 16.0 and 54.0 < lat < 58.0 for lon, lat in layers.asked)


def test_a_sampler_that_answers_a_different_number_of_points_is_refused() -> None:
    """A short answer attached by position puts every value on the wrong vessel and still opens."""
    layers = FakeLayers([Context(1.0, -20.0, "Denmark", 0.0)] * 2)

    with pytest.raises(ValueError, match="3 detections"):
        attach(detections_at(*POSITIONS), layers)


def test_a_layer_that_was_never_sampled_carries_the_same_columns_empty() -> None:
    """A layer whose schema depends on which stages ran is a layer that cannot be stacked."""
    carried = without_context(detections_at(*POSITIONS))

    assert [name in carried.columns for name in CONTEXT] == [True] * len(CONTEXT)
    assert carried["depth_m"].isna().all()
    assert list(carried["eez"]) == [UNAVAILABLE] * 3


def test_a_scene_that_found_nothing_still_carries_the_columns() -> None:
    layers = FakeLayers([])

    carried = attach(detections_at(), layers)

    assert len(carried) == 0
    assert all(name in carried.columns for name in CONTEXT)


def test_the_missing_values_are_still_missing_after_a_round_trip_through_the_geopackage(
    tmp_path: Path,
) -> None:
    """The criterion is about the written output, so it is checked on the written output."""
    layers = FakeLayers([Context(0.0, 0.0, HIGH_SEAS, 0.0), Context(None, None, None, None)])
    carried = attach(detections_at(*POSITIONS[:2]), layers)

    path = tmp_path / "detections.gpkg"
    write_detections(carried, path)
    read = gpd.read_file(path, layer=DETECTIONS_LAYER)

    assert read.loc[0, "fishing_hours"] == 0.0
    assert read.loc[0, "distance_to_shore_m"] == 0.0
    assert read["fishing_hours"].isna()[1]
    assert read["distance_to_shore_m"].isna()[1]
    assert read.loc[1, "eez"] == UNAVAILABLE


def test_a_write_that_fails_leaves_the_detections_it_was_enriching_where_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`darkvessel context` writes back over its own input, so a half-write costs the whole run.

    Every other caller of `write_detections` can lose its output and get it back by running the
    chain again. This one cannot: the file being replaced is the file that was read, and the way
    back from an empty path is a scene, a checkpoint and the whole pipeline.
    """
    path = tmp_path / "detections.gpkg"
    write_detections(without_context(detections_at(*POSITIONS)), path)
    before = path.read_bytes()

    def refuse(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", refuse)
    with pytest.raises(OSError, match="no space"):
        write_detections(without_context(detections_at(*POSITIONS[:1])), path)

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial*"))


def test_the_coverage_says_what_each_layer_answered_and_what_it_did_not() -> None:
    layers = FakeLayers(
        [
            Context(1_000.0, -18.0, "Denmark", 0.0),
            Context(2_000.0, None, HIGH_SEAS, None),
            Context(3_000.0, None, None, None),
        ]
    )
    carried = attach(detections_at(*POSITIONS), layers)

    lines = "\n".join(coverage(carried))

    assert "distance to shore: 3 of 3" in lines
    assert "water depth: 1 of 3" in lines
    assert "fishing effort: 1 of 3" in lines
    assert "1 in a named EEZ, 1 on the high seas, 1 unavailable" in lines


def test_a_source_named_null_is_a_layer_this_run_cannot_sample_rather_than_an_error() -> None:
    """The EEZ boundaries are not in the public catalogue; see docs/decisions.md."""
    sources = context_request_from(context_config(), Path("."))["sources"]

    assert sources.eez is None
    assert sources.effort == "GFW/GFF/V1/fishing_hours"


def test_a_fishing_effort_window_that_ends_before_it_starts_is_refused() -> None:
    config = context_config()
    config["context"]["effort"]["end"] = "2015-01-01"

    with pytest.raises(ValueError, match="window"):
        context_request_from(config, Path("."))


def test_a_fishing_effort_source_with_no_bands_to_read_is_refused() -> None:
    """The collection carries one band per gear type and no total, so the list is the variable.

    An empty list would sample nothing and report it as nothing recorded, which is the one
    reading this column must never carry by accident.
    """
    config = context_config()
    config["context"]["effort"]["bands"] = []

    with pytest.raises(ValueError, match="no bands"):
        context_request_from(config, Path("."))


def test_a_sampling_scale_of_nothing_is_refused() -> None:
    config = context_config()
    config["context"]["scale_m"] = 0

    with pytest.raises(ValueError, match="scale"):
        context_request_from(config, Path("."))


def test_the_command_refuses_a_run_whose_detections_were_never_written(tmp_path: Path) -> None:
    """Before any credential is touched, because the fix is to run the chain rather than to log in.

    It is also what checks the command is wired into `main` at all: the failure is a plain
    `unrecognized arguments` otherwise, and no test of `attach` would notice.
    """
    config = context_config() | {"run": {"output": "detections.gpkg"}}
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(FileNotFoundError, match="darkvessel run"):
        main(["context", "--config", str(config_path)])


def test_the_shipped_config_states_every_source_the_sampling_needs() -> None:
    """The config a reader runs, parsed by the code that runs it."""
    config = yaml.safe_load(SHIPPED_CONFIG.read_text())

    request = context_request_from(config, SHIPPED_CONFIG.parent)

    assert isinstance(request["sources"], LayerSources)
    assert request["sources"].depth_band
    assert request["sources"].scale_m > 0
