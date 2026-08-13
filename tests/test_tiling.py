"""The tiling geometry, on its own.

Full coverage of a scene is not observable from the pipeline seam. A region skipped between two
tiles produces no detection and no error, and a fixture can only ever show that the targets it
happens to carry came back — never that a strip between them was read at all. The property is
asserted directly here instead, over scenes chosen to break the arithmetic in different ways.
"""

import numpy as np
import pytest

from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import PixelDetection

# (scene shape, tile size, overlap), each chosen for the case it breaks:
LAYOUTS = [
    ((64, 64), 36, 8),  # stride divides the scene exactly
    ((50, 75), 16, 4),  # neither dimension divides; the two axes disagree
    ((25, 25), 32, 4),  # the scene is smaller than one tile
    ((33, 17), 16, 8),  # a single row and column left over past the last full stride
]


@pytest.mark.parametrize(("shape", "size_px", "overlap_px"), LAYOUTS)
def test_every_pixel_of_the_scene_is_read_by_some_tile(
    shape: tuple[int, int], size_px: int, overlap_px: int
) -> None:
    read = np.zeros(shape, dtype=bool)

    for tile in Tiling(size_px=size_px, overlap_px=overlap_px).tiles(shape):
        read[tile.rows, tile.cols] = True

    assert read.all(), f"{(~read).sum()} pixels of {shape} were not covered by any tile"


@pytest.mark.parametrize(("shape", "size_px", "overlap_px"), LAYOUTS)
def test_a_target_anywhere_in_the_scene_is_claimed_by_exactly_one_tile(
    shape: tuple[int, int], size_px: int, overlap_px: int
) -> None:
    """The deduplication rule, stated as the property it exists to guarantee.

    Every position in the scene is walked as each tile sees it — in tile coordinates, which is
    all a detector ever reports — and counted where the claim comes back, in scene coordinates.
    Two claims on one position is a duplicated target; none is a lost one.
    """
    claims = np.zeros(shape, dtype=int)

    for tile in Tiling(size_px=size_px, overlap_px=overlap_px).tiles(shape):
        for row, col in np.ndindex(tile.height, tile.width):
            seen = PixelDetection(row=float(row), col=float(col), score=1.0)
            claimed = tile.claim(seen)
            if claimed is not None:
                claims[round(claimed.row), round(claimed.col)] += 1

    duplicated = int((claims > 1).sum())
    lost = int((claims == 0).sum())
    assert (duplicated, lost) == (0, 0), f"{duplicated} positions claimed twice, {lost} by none"


@pytest.mark.parametrize(("shape", "size_px", "overlap_px"), LAYOUTS)
def test_a_tile_reaches_half_the_overlap_past_everything_it_claims(
    shape: tuple[int, int], size_px: int, overlap_px: int
) -> None:
    """What the overlap buys, and the condition the whole scheme rests on.

    Claiming a target is only sound if the claiming tile saw the whole of it — otherwise the
    tile reports the centroid of a target cut in half. Every core therefore stops half an
    overlap short of its tile's edge, so any target up to the overlap across is seen entire by
    whichever tile owns it. Only the edges of the scene are exempt: there is nothing past them
    to be cut off by.
    """
    reach = overlap_px / 2

    for tile in Tiling(size_px=size_px, overlap_px=overlap_px).tiles(shape):
        for core, start, length, extent in (
            (tile.core_rows, tile.row_off, tile.height, shape[0]),
            (tile.core_cols, tile.col_off, tile.width, shape[1]),
        ):
            assert core.start == 0.0 or core.start >= start + reach
            assert core.stop == float(extent) or core.stop <= start + length - reach


@pytest.mark.parametrize(
    ("size_px", "overlap_px"),
    [(0, 0), (64, -1), (64, 64), (64, 96)],
)
def test_a_tiling_that_would_never_cross_the_scene_is_refused(
    size_px: int, overlap_px: int
) -> None:
    # An overlap at or past the tile size gives a stride of zero or less. Left to run, it would
    # loop forever or step backwards; refused at construction, it names the config that is wrong.
    with pytest.raises(ValueError):
        Tiling(size_px=size_px, overlap_px=overlap_px)
