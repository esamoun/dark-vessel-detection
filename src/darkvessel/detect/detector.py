"""The detector contract, and the unit it deals in.

Kept in a module of its own, importing nothing from the rest of the package, so that both the
pipeline that calls a detector and the code that georeferences its output can depend on the
contract without depending on each other. The architecture that will eventually satisfy this
contract lives in `model.py`; the deterministic substitute used to exercise the chain lives in
`threshold.py`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PixelDetection:
    """One detected target in image pixel coordinates, before georeferencing.

    ``row`` and ``col`` are fractional pixel indices addressing the pixel *centre*: (0.0, 0.0)
    is the centre of the top-left pixel. Keeping the convention here, in one place, is what
    stops it being reinvented differently by the code that converts it to the ground.
    """

    row: float
    col: float
    score: float


class Detector(Protocol):
    """Anything that turns an amplitude image into targets. The injected dependency."""

    def __call__(self, image: np.ndarray) -> Sequence[PixelDetection]: ...
