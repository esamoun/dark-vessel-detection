"""Tiling of large scenes, with overlap, and the rule that keeps a target from being counted twice.

A Sentinel-1 scene does not fit in GPU memory. Tiles overlap so that vessels near a tile edge
are not cut in half; the overlap is what makes cross-tile deduplication possible downstream.
Geometry-critical: covered by tests.

Overlapping tiles see the same target twice, so one of the two views has to be dropped. That is
done by ownership rather than by comparing detections after the fact: the scene is partitioned
into cores, one per tile, and a tile reports only what falls in its own core. Every position in
the scene lies in exactly one core, so a target is claimed exactly once — by construction, with
no distance threshold to tune and no risk of merging two hulls that genuinely lie close together.
See docs/decisions.md.

The condition this rests on: the overlap must be at least as wide as the largest target. A tile's
core stops half an overlap short of its edge, so a target centred in the core is seen whole by the
tile that owns it, and the clipped view the neighbour gets is discarded rather than reported at
its own, wrong, centroid.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from darkvessel.detect.detector import PixelDetection


@dataclass(frozen=True)
class Core:
    """The stretch of one axis of the scene that a single tile answers for.

    Bounds are pixel edges, so 0.0 is the leading edge of the scene and an integer bound falls
    between two pixels. Half-open, which is what makes cores partition rather than merely cover:
    a target landing exactly on the bound between two of them belongs to the second, and to only
    one of them.
    """

    start: float
    stop: float

    def contains(self, index: float) -> bool:
        """Whether a pixel index falls in this core.

        A pixel index addresses the centre of that pixel, half a pixel past its leading edge —
        the same convention `geo.py` undoes when it puts a detection on the ground. Comparing an
        index against an edge without that half pixel is a silent half-pixel error at every
        boundary in the scene.
        """
        return self.start <= index + 0.5 < self.stop


@dataclass(frozen=True)
class Tile:
    """A window on the scene, and the part of it this tile answers for.

    `row_off`/`col_off` place the window in the scene; `core_rows`/`core_cols` bound the region
    this tile, and no other, reports detections in.
    """

    row_off: int
    col_off: int
    height: int
    width: int
    core_rows: Core
    core_cols: Core

    @property
    def rows(self) -> slice:
        return slice(self.row_off, self.row_off + self.height)

    @property
    def cols(self) -> slice:
        return slice(self.col_off, self.col_off + self.width)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """The part of the scene this tile reads. A view, not a copy."""
        return image[self.rows, self.cols]

    def claim(self, detection: PixelDetection) -> PixelDetection | None:
        """Move a detection from this tile's coordinates to the scene's, if the tile owns it.

        `None` means a neighbouring tile owns that position and will report the target itself.
        Translation and ownership are one operation deliberately: the test has to be made in
        scene coordinates, and separating the two invites it being made in tile coordinates,
        where it is meaningless.
        """
        row = self.row_off + detection.row
        col = self.col_off + detection.col
        if not (self.core_rows.contains(row) and self.core_cols.contains(col)):
            return None
        return replace(detection, row=row, col=col)


@dataclass(frozen=True)
class Tiling:
    """How a scene is cut into overlapping tiles. Both numbers come from the run's config."""

    size_px: int
    overlap_px: int

    def __post_init__(self) -> None:
        if self.size_px < 1:
            raise ValueError(f"tile size must be at least 1 px, got {self.size_px}")
        if self.overlap_px < 0:
            raise ValueError(f"tile overlap cannot be negative, got {self.overlap_px}")
        if self.overlap_px >= self.size_px:
            raise ValueError(
                f"tile overlap of {self.overlap_px} px leaves no stride in a tile of "
                f"{self.size_px} px; the tiles would never advance across the scene"
            )

    def tiles(self, shape: tuple[int, int]) -> Iterator[Tile]:
        """Every tile covering a scene of `shape`, in row-major order."""
        rows = _axis(shape[0], self.size_px, self.overlap_px)
        cols = _axis(shape[1], self.size_px, self.overlap_px)

        for row_off, core_rows in rows:
            for col_off, core_cols in cols:
                yield Tile(
                    row_off=row_off,
                    col_off=col_off,
                    height=min(self.size_px, shape[0]),
                    width=min(self.size_px, shape[1]),
                    core_rows=core_rows,
                    core_cols=core_cols,
                )


def _axis(extent: int, size: int, overlap: int) -> list[tuple[int, Core]]:
    """Tile starts along one axis, each with the stretch of the scene it answers for."""
    offsets = _offsets(extent, size, overlap)
    # Neighbours share their overlap, and the edge between their cores is the middle of it. The
    # first and last edges are the edges of the scene: nothing lies beyond them to answer for.
    pairs = zip(offsets, offsets[1:], strict=False)
    edges = [0.0, *((previous + size + offset) / 2 for previous, offset in pairs), float(extent)]
    cores = [Core(start, stop) for start, stop in zip(edges, edges[1:], strict=False)]
    return list(zip(offsets, cores, strict=True))


def _offsets(extent: int, size: int, overlap: int) -> list[int]:
    """Where tiles start along one axis, the last one flush with the far edge of the scene.

    The scene is not a multiple of the stride, so the last tile is pulled back against the edge
    rather than allowed to hang over it or be cut short. It overlaps its predecessor by more
    than `overlap`, which costs a little duplicated work and leaves every tile the same size —
    the shape a detector is trained on.
    """
    if extent <= size:
        return [0]

    stride = size - overlap
    return [*range(0, extent - size, stride), extent - size]
