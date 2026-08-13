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
                image=_amplitude(dataset, band),
                transform=dataset.transform,
                crs=dataset.crs.to_string(),
                acquired_at=datetime.fromisoformat(acquired_at),
            )


def _amplitude(dataset: rasterio.DatasetReader, band: int) -> np.ndarray:
    """The band's pixels, with anything the product declares as nodata turned into NaN.

    A real Sentinel-1 product has holes — pixels the producer masked — and writes them as a fill
    value with `nodata` set alongside. Read plainly, that fill is just a number, and on a scene
    in dB where the sea sits near -14 dB and the fill is 0, it is brighter than any vessel in the
    image. The first real scene run through this chain returned three "targets" of 72100, 38955
    and 36428 pixels for exactly that reason: no crash, no warning, a plausible count.

    NaN rather than a mask because every comparison against NaN is false, so a hole cannot be
    above any threshold a detector picks — including one written later that never thought about
    nodata at all.
    """
    image = dataset.read(band).astype(np.float32, copy=False)
    nodata = dataset.nodatavals[band - 1]
    if nodata is not None:
        image = np.where(image == np.float32(nodata), np.nan, image)
    return image
