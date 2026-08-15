"""The conversion between what the chain exports and what the detector was fitted on.

Every window in this file is chosen here, in the test, and none of them is the shipped one. The
shipped window is a measurement — where the sea sat in the images the model was fitted on — and a
test that pinned it would turn that measurement into a target, which is the rule this project
already applies to precision and recall. What is asserted here is arithmetic and refusal: that a
window places a value where it says it does, and that one which cannot be applied honestly is
rejected rather than approximated.
"""

import numpy as np
import pytest

from darkvessel.detect.amplitude import (
    DecibelStretch,
    SeaReference,
    fit_window,
    sea_level,
    without_holes,
)
from darkvessel.detect.detector import PixelDetection

# Round numbers, so every expectation below is arithmetic a reader can do by eye: forty decibels
# wide, and the sea eight of those forty above the floor.
STRETCH = DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-22.0)


def test_the_floor_is_zero_and_the_ceiling_is_one():
    converted = STRETCH(np.array([[-30.0, 10.0]], dtype=np.float32))
    assert converted.tolist() == [[0.0, 1.0]]


def test_the_sea_lands_where_the_window_puts_it():
    assert STRETCH(np.array([[-22.0]], dtype=np.float32))[0, 0] == pytest.approx(0.2)


def test_beyond_either_end_is_clipped_rather_than_wrapped():
    # -48 dB and +27 dB are the extremes of the real scene this window was written against.
    converted = STRETCH(np.array([[-48.0, 27.0]], dtype=np.float32))
    assert converted.tolist() == [[0.0, 1.0]]


def test_a_hole_comes_back_at_the_sea_and_never_at_the_top():
    converted = STRETCH(np.array([[np.nan, -22.0]], dtype=np.float32))
    assert converted[0, 0] == pytest.approx(0.2)
    assert converted[0, 0] == converted[0, 1]


def test_nothing_comes_back_as_nan():
    """One NaN reaching the network propagates through every convolution that touches it."""
    converted = STRETCH(np.array([[np.nan, np.nan]], dtype=np.float32))
    assert np.isfinite(converted).all()


def test_the_input_is_not_modified():
    """The guard downstream reads the holes off the original, so the original keeps them."""
    image = np.array([[np.nan, -22.0]], dtype=np.float32)
    STRETCH(image)
    assert np.isnan(image[0, 0])


def test_the_result_is_float32_as_the_model_input_expects():
    # A real product comes off rasterio as float64; `as_model_input` hands whatever it is
    # straight to torch, and a float64 tile is a tensor the model will not take.
    assert STRETCH(np.array([[-22.0]], dtype=np.float64)).dtype == np.float32


def test_a_window_that_does_not_widen_is_refused():
    with pytest.raises(ValueError, match="ceiling"):
        DecibelStretch(floor_db=0.0, ceiling_db=-10.0, sea_db=-5.0)


def test_a_sea_outside_its_own_window_is_refused():
    with pytest.raises(ValueError, match="sea"):
        DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-40.0)


# A reference sea standing at 0.15 with a spread of 0.05, which is a plausible shape for an 8-bit
# product and is deliberately *not* the measured one. What the measurement said belongs in the
# config and in docs/decisions.md, where changing it is a decision rather than a failing test.
REFERENCE = SeaReference(mean=0.15, spread=0.05)

# The first real scene, measured: median and MAD-derived spread of kattegat-lane.tif. Used here
# only as a realistic pair of inputs — nothing below asserts what window they produce.
SCENE_SEA_DB = -21.84
SCENE_SPREAD_DB = 2.30


def test_the_fitted_window_puts_the_sea_where_the_reference_says():
    stretch = fit_window(sea_db=SCENE_SEA_DB, spread_db=SCENE_SPREAD_DB, reference=REFERENCE)
    converted = stretch(np.array([[SCENE_SEA_DB]], dtype=np.float32))
    assert converted[0, 0] == pytest.approx(REFERENCE.mean, abs=1e-4)


def test_one_spread_above_the_sea_is_one_reference_spread_above_it():
    """The second moment. Matching only the first would place the sea correctly and leave the
    contrast between a hull and the water it sits in scaled by an arbitrary factor."""
    stretch = fit_window(sea_db=SCENE_SEA_DB, spread_db=SCENE_SPREAD_DB, reference=REFERENCE)
    converted = stretch(np.array([[SCENE_SEA_DB + SCENE_SPREAD_DB]], dtype=np.float32))
    assert converted[0, 0] == pytest.approx(REFERENCE.mean + REFERENCE.spread, abs=1e-4)


def test_the_fitted_window_fills_holes_at_the_sea_it_was_fitted_to():
    stretch = fit_window(sea_db=SCENE_SEA_DB, spread_db=SCENE_SPREAD_DB, reference=REFERENCE)
    assert stretch.sea_db == SCENE_SEA_DB
    assert stretch.sea == pytest.approx(REFERENCE.mean, abs=1e-4)


def test_a_reference_with_no_spread_is_refused():
    with pytest.raises(ValueError, match="spread"):
        SeaReference(mean=0.15, spread=0.0)


def test_the_sea_is_measured_past_the_ships_in_it():
    """A plain standard deviation would be dragged upwards by the targets, and the window fitted
    from it would be widened by exactly the signal it exists to preserve."""
    generator = np.random.default_rng(20260815)
    image = generator.normal(-22.0, 2.0, size=(200, 200)).astype(np.float32)
    image[:5, :8] = 25.0

    median, sigma = sea_level(image)
    assert median == pytest.approx(-22.0, abs=0.15)
    assert sigma == pytest.approx(2.0, abs=0.15)


def test_holes_do_not_count_as_sea():
    image = np.full((100, 100), -22.0, dtype=np.float32)
    image[:50] = np.nan

    median, _ = sea_level(image)
    assert median == pytest.approx(-22.0)


def test_a_scene_with_no_water_left_in_it_is_refused():
    with pytest.raises(ValueError, match="nodata"):
        sea_level(np.full((4, 4), np.nan, dtype=np.float32))


def _tile_with_a_hole_at(row: int, col: int) -> np.ndarray:
    image = np.full((4, 4), -22.0, dtype=np.float32)
    image[row, col] = np.nan
    return image


def test_a_detection_centred_in_a_hole_is_dropped():
    kept = without_holes([PixelDetection(row=1.0, col=1.0, score=0.9)], _tile_with_a_hole_at(1, 1))
    assert kept == []


def test_a_detection_on_water_survives():
    detection = PixelDetection(row=2.0, col=2.0, score=0.9)
    assert without_holes([detection], _tile_with_a_hole_at(1, 1)) == [detection]


def test_the_pixel_a_fractional_centre_falls_in_is_the_one_that_is_checked():
    """1.4 addresses a point inside pixel 1, which is the hole; 1.6 is inside pixel 2, which is
    not. The half added before flooring is the one `tiling.Core.contains` adds, for the same
    reason: comparing an index against an edge without it is a silent half-pixel error."""
    image = _tile_with_a_hole_at(1, 1)
    assert without_holes([PixelDetection(row=1.4, col=1.4, score=0.9)], image) == []
    assert len(without_holes([PixelDetection(row=1.6, col=1.6, score=0.9)], image)) == 1


def test_order_and_scores_are_left_alone():
    """The guard drops; it does not reorder, rescore or deduplicate. Claiming a target once is
    `tiling.Tile.claim`'s job and it is done by ownership, not here."""
    image = _tile_with_a_hole_at(0, 0)
    detections = [
        PixelDetection(row=3.0, col=1.0, score=0.4),
        PixelDetection(row=1.0, col=2.0, score=0.9),
    ]
    assert without_holes(detections, image) == detections


def test_a_detection_outside_the_tile_is_not_treated_as_a_hole():
    """A box can be regressed past the edge of the tile it was proposed in. That is a detection
    the tiling will discard as belonging to no core, and it is not this guard's to judge."""
    detection = PixelDetection(row=-1.0, col=9.0, score=0.9)
    assert without_holes([detection], _tile_with_a_hole_at(1, 1)) == [detection]
