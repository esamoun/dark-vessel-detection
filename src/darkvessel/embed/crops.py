"""Detection crops: the unit the representation is learned over.

A detection is a point. What a representation can be learned from is the neighbourhood around
that point, and this module is where a point becomes a small square of amplitude. Nothing here
decides what a crop *means* — that is `contrastive.py` — and nothing here imports torch, so the
conventions most likely to be got wrong quietly are on the side of the seam a laptop tests in a
second, beside `amplitude.py` and `dataset.py`.

Two conventions carry the module, and both are the same convention the rest of the chain already
states in one place rather than reinventing. A detection's `row` and `col` are fractional indices
addressing a pixel centre, so the pixel a detection falls in is `floor(row + 0.5)` — the half
`without_holes` and `tiling.Core.contains` each add for the same reason. And a crop that overruns
the edge of the scene is padded with NaN rather than dropped or filled: NaN is what a hole in a
product already reads as, `DecibelStretch` fills it at the sea, and a vessel imaged at the edge of
a scene is exactly the object an archive should not be quietly missing.

Crops are cut with a margin around them, and the margin is not decoration. It is what lets a view
of a crop be translated by a few pixels without either wrapping around or reaching for pixels that
were never stored — the random-crop half of a contrastive view, in the one form that stays
geometry-preserving on radar amplitude. `centre` is the other end of the same convention: what the
encoder is shown when nothing is being augmented.
"""

from collections.abc import Sequence

import numpy as np

from darkvessel.detect.detector import PixelDetection


def stored_px(crop_px: int, margin_px: int) -> int:
    """The side actually cut from the scene, for a crop the encoder sees at `crop_px`.

    One function rather than the same addition written in five places: the archive, the chain's
    embedding stage and the views all have to agree on it, and a disagreement would show up as a
    representation fitted at one scale and applied at another.
    """
    if crop_px <= 0:
        raise ValueError(f"a crop of {crop_px} px has no pixels in it")
    if margin_px < 0:
        raise ValueError(f"a margin of {margin_px} px is not a margin")
    return crop_px + 2 * margin_px


def crop_at(image: np.ndarray, row: float, col: float, size_px: int) -> np.ndarray:
    """The `size_px` square of `image` centred on the pixel (`row`, `col`) falls in.

    Where the square overruns the scene it is padded with NaN, which is what this chain already
    means by "no measurement here": `scene.py` writes a product's nodata that way, and
    `DecibelStretch` fills it at the sea before a network ever sees it. Dropping the detection
    instead would lose exactly the objects at the edge of a scene, and filling with a number would
    put a hard bright or dark wall next to a hull.

    For an even `size_px` the detection's own pixel sits one place past the middle, at index
    `size_px // 2`. Stated rather than balanced away: a crop cannot be symmetric about a pixel
    and have an even side, and choosing the convention here is what stops it being chosen
    differently by the code that draws the crops out.
    """
    if size_px <= 0:
        raise ValueError(f"a crop of {size_px} px has no pixels in it")

    top = int(np.floor(row + 0.5)) - size_px // 2
    left = int(np.floor(col + 0.5)) - size_px // 2

    crop = np.full((size_px, size_px), np.nan, dtype=np.float32)
    rows, cols = image.shape
    into_rows = slice(max(0, -top), min(size_px, rows - top))
    into_cols = slice(max(0, -left), min(size_px, cols - left))
    from_rows = slice(max(0, top), min(rows, top + size_px))
    from_cols = slice(max(0, left), min(cols, left + size_px))
    crop[into_rows, into_cols] = image[from_rows, from_cols]

    return crop


def crops_for(
    image: np.ndarray,
    detections: Sequence[PixelDetection],
    *,
    crop_px: int,
    margin_px: int = 0,
) -> np.ndarray:
    """Every detection in `detections`, as a stack of squares cut from `image`.

    In the order the detections arrive, which is the order the chain reports them in and
    therefore the order a layer's rows are in: the vectors that come back are attached to those
    rows by position, and nothing re-sorts in between.

    Returns an array of shape (n, s, s) with `s = stored_px(crop_px, margin_px)`, empty of rows
    but still square where there is nothing to crop — a scene with no detections is an ordinary
    outcome and must not come back as a shape nothing downstream can stack.
    """
    side = stored_px(crop_px, margin_px)
    if not detections:
        return np.empty((0, side, side), dtype=np.float32)

    return np.stack(
        [crop_at(image, detection.row, detection.col, side) for detection in detections]
    )


def centre(crops: np.ndarray, crop_px: int) -> np.ndarray:
    """The middle `crop_px` of each stored crop: what the encoder is shown, unaugmented.

    The counterpart of the margin `crops_for` cuts. A view drawn during training takes some other
    window of the same square; this takes the one the detection is actually at, so that what is
    stored in a layer is a representation of the object rather than of a corner near it.
    """
    stored = crops.shape[-1]
    if crop_px > stored:
        raise ValueError(
            f"the encoder wants {crop_px} px and the crops are {stored} px; a crop cannot be "
            "enlarged without resampling it, and resampling radar amplitude is a decision"
        )

    start = (stored - crop_px) // 2
    return crops[..., start : start + crop_px, start : start + crop_px]
