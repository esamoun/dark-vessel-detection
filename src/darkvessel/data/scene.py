"""A radar scene as the pipeline sees it.

Amplitude, the affine transform that places it on the ground, the CRS that transform is
expressed in, and the moment the sensor acquired it. The acquisition time is carried here rather
than looked up later because the fusion stage is only as correct as that timestamp.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

ACQUIRED_AT_TAG = "ACQUIRED_AT"


@dataclass(frozen=True)
class Scene:
    """A single-band amplitude image with everything needed to place it on the ground."""

    image: np.ndarray
    transform: Affine
    crs: str
    acquired_at: datetime

    def __post_init__(self) -> None:
        if self.image.ndim != 2:
            raise ValueError(f"scene image must be 2-D, got shape {self.image.shape}")
        if self.acquired_at.tzinfo is None:
            raise ValueError("acquired_at must be timezone-aware; the fusion stage compares it")

    @classmethod
    def from_geotiff(cls, path: Path, band: int = 1) -> "Scene":
        """Read a scene, taking its acquisition time from the ACQUIRED_AT tag.

        A missing tag is an error rather than a default. Guessing the acquisition time would
        not fail loudly — it would quietly match detections against the wrong AIS reports and
        manufacture dark vessels.
        """
        with rasterio.open(path) as dataset:
            acquired_at = dataset.tags().get(ACQUIRED_AT_TAG)
            if acquired_at is None:
                raise ValueError(
                    f"{path} carries no {ACQUIRED_AT_TAG} tag; the acquisition time is required "
                    "to match against AIS and must not be guessed"
                )
            return cls(
                image=dataset.read(band),
                transform=dataset.transform,
                crs=dataset.crs.to_string(),
                acquired_at=datetime.fromisoformat(acquired_at),
            )
