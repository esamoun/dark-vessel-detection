"""What a view of a crop is allowed to change, and what it must not.

There are no labels at this level, so the augmentations *are* the supervision: what two views of
one crop have in common is the whole of what the representation is told to keep. An augmentation
that quietly destroys the object leaves a model that has learned the sea, and nothing about that
failure looks like a failure — the loss falls either way.

So the assertions here are about physics rather than about arithmetic. A symmetry moves pixels
and invents none. Speckle multiplies an intensity and leaves a hole a hole. A translation stays
inside the margin the archive stored for it, and never wraps a vessel around its own edge.
"""

import numpy as np
import pytest

from darkvessel.embed.views import NOMINAL_LOOKS, Speckle, laid_down, looks_of, rng_for, view


def crop(size: int = 8) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return (rows * 100 + cols).astype(np.float32)


def test_the_eight_symmetries_move_pixels_and_invent_none() -> None:
    original = crop()

    laid = [laid_down(original, index) for index in range(8)]

    assert len({tuple(image.ravel()) for image in laid}) == 8, "eight distinct layings expected"
    for image in laid:
        assert sorted(image.ravel().tolist()) == sorted(original.ravel().tolist())


def test_speckle_shakes_a_pixel_by_a_gamma_draw_and_leaves_a_hole_alone() -> None:
    """A pixel the product never measured does not acquire a measurement by being looked at
    again, and NaN plus anything is what says so."""
    sea = np.full((64, 64), -21.0, dtype=np.float32)
    sea[0, 0] = np.nan

    looked = Speckle(looks=NOMINAL_LOOKS)(sea, np.random.default_rng(0))

    assert np.isnan(looked[0, 0])
    # A multiplicative fluctuation of mean one: the intensity, not the decibels, averages out.
    intensity = np.power(10.0, looked[np.isfinite(looked)] / 10.0)
    assert float(np.mean(intensity)) == pytest.approx(10.0 ** (-21.0 / 10.0), rel=0.02)


def test_the_number_of_looks_is_recovered_from_a_sea_that_has_that_many() -> None:
    """The measurement `looks_of` makes, checked against a sea built to a known figure.

    Speckled at 4.4 looks, a synthetic sea must measure back at 4.4 looks — otherwise the number
    the shipped config carries is measuring something other than what it claims.
    """
    sea = np.full((512, 512), -21.0, dtype=np.float32)

    speckled = Speckle(looks=NOMINAL_LOOKS)(sea, np.random.default_rng(7))

    assert looks_of(speckled) == pytest.approx(NOMINAL_LOOKS, rel=0.05)


def test_a_sea_with_no_speckle_in_it_is_refused_rather_than_reported_as_infinite_looks() -> None:
    with pytest.raises(ValueError, match="no variation"):
        looks_of(np.full((16, 16), -21.0, dtype=np.float32))


def test_a_view_is_a_window_of_the_stored_crop_and_never_wraps_around_it() -> None:
    # Every value in a view has to be a value that was stored: a translation that rolled the crop
    # would put one side of a vessel against the other and still look like a plausible ship.
    stored = crop(size=12)

    for draw in range(20):
        taken = view(stored, crop_px=8, speckle=None, rng=rng_for("crop", draw))

        assert taken.shape == (8, 8)
        assert set(taken.ravel().tolist()) <= set(stored.ravel().tolist())


def test_two_views_of_one_crop_differ_and_two_draws_of_one_name_do_not() -> None:
    """Reproducible by name rather than by position in a stream — the convention
    `dataset.symmetry_for` states, because a resumed run has to reproduce what it was doing."""
    stored = crop(size=12)

    first = view(stored, crop_px=8, speckle=None, rng=rng_for("crop-3", 1, "view-0"))
    again = view(stored, crop_px=8, speckle=None, rng=rng_for("crop-3", 1, "view-0"))
    other = view(stored, crop_px=8, speckle=None, rng=rng_for("crop-3", 1, "view-1"))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_a_view_larger_than_the_crop_it_is_taken_from_is_refused() -> None:
    with pytest.raises(ValueError, match="resampling"):
        view(crop(size=8), crop_px=16, speckle=None, rng=rng_for("x"))
