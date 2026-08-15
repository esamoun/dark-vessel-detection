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

from darkvessel.detect.amplitude import DecibelStretch

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
