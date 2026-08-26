"""Cutting a detection out of a scene.

Three things can go wrong here without anything crashing, and each of them produces an archive
that looks exactly like a working one. A crop centred half a pixel off puts every object slightly
off-centre, which a representation then learns to be invariant to instead of learning about the
object. A crop at the edge of a scene padded with a number rather than left as a hole puts a hard
wall of bright or dark next to a hull. And a stack whose order does not match the detections it
came from attaches every vector to the wrong row of the layer.

Nothing here needs torch: the conventions above are arithmetic, and they are the half of this
level a laptop can check in a second.
"""

import numpy as np
import pytest

from darkvessel.detect.detector import PixelDetection
from darkvessel.embed.crops import centre, crop_at, crops_for, has_measurements, stored_px


def scene(size: int = 32) -> np.ndarray:
    """A scene whose every pixel says where it is: value = row * 100 + col."""
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return (rows * 100 + cols).astype(np.float32)


def test_a_crop_is_centred_on_the_pixel_the_detection_falls_in() -> None:
    # An odd side has a middle, so the assertion can be made on the exact pixel.
    crop = crop_at(scene(), row=10.0, col=20.0, size_px=5)

    assert crop.shape == (5, 5)
    assert crop[2, 2] == pytest.approx(10 * 100 + 20)
    assert crop[0, 0] == pytest.approx(8 * 100 + 18)


def test_a_fractional_index_addresses_the_pixel_it_falls_in() -> None:
    """The half `without_holes` adds, added here too and for the same reason.

    A detection at row 10.6 is inside pixel 11, not pixel 10. Comparing the index against the
    grid without that half is a silent half-pixel error at every crop in the archive.
    """
    below = crop_at(scene(), row=10.6, col=20.0, size_px=5)
    at = crop_at(scene(), row=11.0, col=20.0, size_px=5)

    assert np.array_equal(below, at)


def test_a_crop_that_overruns_the_scene_is_padded_with_holes_rather_than_dropped() -> None:
    # A vessel imaged at the edge of a scene is exactly the object an archive should not be
    # missing, and a fill value would be a wall of contrast beside a hull.
    crop = crop_at(scene(size=32), row=0.0, col=0.0, size_px=5)

    assert np.isnan(crop[:2, :]).all()
    assert np.isnan(crop[:, :2]).all()
    assert crop[2, 2] == pytest.approx(0.0)
    assert crop[3, 3] == pytest.approx(1 * 100 + 1)


def test_crops_come_back_in_the_order_their_detections_did() -> None:
    """The vectors are attached to a layer's rows by position, and nothing re-sorts in between."""
    detections = [
        PixelDetection(row=5.0, col=5.0, score=0.1),
        PixelDetection(row=20.0, col=20.0, score=0.9),
    ]

    stack = crops_for(scene(), detections, crop_px=3)

    assert stack.shape == (2, 3, 3)
    assert stack[0][1, 1] == pytest.approx(5 * 100 + 5)
    assert stack[1][1, 1] == pytest.approx(20 * 100 + 20)


def test_a_scene_with_no_detections_still_returns_something_stackable() -> None:
    # An ordinary outcome — most acquisitions over open water hold a handful of ships and some
    # hold none — and it must not come back as a shape nothing downstream can concatenate.
    stack = crops_for(scene(), [], crop_px=8, margin_px=4)

    assert stack.shape == (0, 16, 16)


def test_the_margin_is_cut_around_the_crop_the_encoder_sees() -> None:
    assert stored_px(crop_px=64, margin_px=8) == 80

    stack = crops_for(
        scene(), [PixelDetection(row=16.0, col=16.0, score=1.0)], crop_px=4, margin_px=2
    )

    assert stack.shape == (1, 8, 8)
    # The middle four of the eight are the crop; the rest is what a view may translate into.
    assert centre(stack, crop_px=4).shape == (1, 4, 4)
    assert centre(stack, crop_px=4)[0][2, 2] == pytest.approx(16 * 100 + 16)


def test_a_crop_is_never_enlarged_to_the_size_the_encoder_wants() -> None:
    """Resampling radar amplitude is a decision, and the same refusal `check_tile_size` makes."""
    with pytest.raises(ValueError, match="resampling"):
        centre(np.zeros((1, 8, 8), dtype=np.float32), crop_px=16)


def test_a_clip_holding_no_measurement_at_all_is_recognised_as_one() -> None:
    """Earth Engine lists an acquisition whose footprint *intersects* the rectangle, not one that
    covers it, so a scene can be exported whole and hold no water. Three of the fifty over the
    Anholt box are like this, and the archive is right to skip them rather than die on the twelfth
    of ninety-six — but only if it can tell."""
    assert not has_measurements(np.full((8, 8), np.nan, dtype=np.float32))

    mostly_empty = np.full((8, 8), np.nan, dtype=np.float32)
    mostly_empty[3, 4] = -21.0

    assert has_measurements(mostly_empty)
    assert has_measurements(scene())
