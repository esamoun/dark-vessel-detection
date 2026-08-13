"""A deterministic threshold detector, standing in for the trained model.

Bright pixels above a threshold are grouped into connected regions and each region is reported
once, at its centroid — the same output shape a real detector produces, so the chain around it
is exercised honestly. There are no weights, no GPU and no network, which is what makes the
whole pipeline testable.

This is a substitute, not a detector: it has no notion of what a vessel looks like, and on a
real scene it would return every bright scatterer in the image. It is sized for synthetic
scenes and small tiles.
"""

from collections.abc import Iterator

import numpy as np

from darkvessel.detect.detector import PixelDetection

_NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


class BrightPixelDetector:
    """Reports one detection per connected region of pixels at or above `threshold`."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, image: np.ndarray) -> list[PixelDetection]:
        mask = image >= self.threshold
        return [
            PixelDetection(
                row=float(np.mean([r for r, _ in region])),
                col=float(np.mean([c for _, c in region])),
                score=float(max(image[r, c] for r, c in region)),
            )
            for region in _connected_regions(mask)
        ]


def _connected_regions(mask: np.ndarray) -> Iterator[list[tuple[int, int]]]:
    """Yield the 8-connected regions of a boolean mask, in row-major order of first pixel.

    Seeded from the set pixels rather than by scanning the image, so cost follows the number of
    bright pixels rather than the size of the scene.
    """
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)

    for seed in np.argwhere(mask):
        seed_rc = (int(seed[0]), int(seed[1]))
        if seen[seed_rc]:
            continue

        seen[seed_rc] = True
        stack = [seed_rc]
        region = []
        while stack:
            row, col = stack.pop()
            region.append((row, col))
            for d_row, d_col in _NEIGHBOURS:
                neighbour = (row + d_row, col + d_col)
                if 0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols:
                    if mask[neighbour] and not seen[neighbour]:
                        seen[neighbour] = True
                        stack.append(neighbour)

        yield region
