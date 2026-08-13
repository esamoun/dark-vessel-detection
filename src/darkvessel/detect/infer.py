"""Full-scene inference.

Crop, detect, claim: a target seen by several overlapping tiles is reconciled in scene
coordinates, never in tile coordinates. This is the step that turns a model into a chain.

Nothing is merged here, and that is the design rather than an omission — the reconciliation is
done by ownership, in `tiling.py`, where the geometry that makes it correct lives too. Each tile
answers for one stretch of the scene and stays quiet about the rest, so a target on a boundary is
claimed once however many tiles saw it, with no radius to tune. See docs/decisions.md.

What is left here is the order of operations, and the guarantee that the answer does not depend
on how the scene happened to be cut.
"""

import numpy as np

from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector, PixelDetection


def detect_scene(
    image: np.ndarray,
    detector: Detector,
    tiling: Tiling,
) -> list[PixelDetection]:
    """Run `detector` over every tile of `image` and return its targets in scene coordinates.

    Sorted into scene row-major order, so that the same scene returns the same detections in the
    same order whatever tile size it was run at. Tiling is a constraint of the hardware; it has
    no business showing through in the answer.
    """
    detections = [
        claimed
        for tile in tiling.tiles(image.shape)
        for detection in detector(tile.crop(image))
        if (claimed := tile.claim(detection)) is not None
    ]
    return sorted(detections, key=_in_scene_order)


def _in_scene_order(detection: PixelDetection) -> tuple[float, float]:
    return detection.row, detection.col
