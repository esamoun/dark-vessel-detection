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
ORBIT_PASS_TAG = "ORBIT_PASS"


@dataclass(frozen=True)
class Scene:
    """A single-band amplitude image with everything needed to place it on the ground."""

    image: np.ndarray
    transform: Affine
    crs: str
    acquired_at: datetime
    # Which way the satellite was travelling. Written by the export, and read here because it is
    # half of what decides where a moving vessel gets drawn — see `fusion/azimuth.py`. None for a
    # scene that has no orbit behind it, which is the synthetic one: the correction then has
    # nothing to compute from and is not applied, rather than being applied from a default.
    orbit_pass: str | None = None
    # What this acquisition is called, so a detection can be traced back to the product it came
    # out of. One run over one scene never needed it — there was only ever one answer. An archive
    # -wide run accumulates fifty scenes into one layer, and a detection nobody can point at an
    # acquisition is the same non-evidence a crop with no provenance is in `embed/archive.py`.
    # None for a scene built in memory, which is what the tests and the synthetic fixture do.
    name: str | None = None

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
                # Absent rather than guessed. A missing pass is a scene the azimuth correction
                # declines to touch, which is right: guessing it wrong reverses the direction
                # every vessel is moved in, and a reversed correction is worse than none.
                orbit_pass=dataset.tags().get(ORBIT_PASS_TAG),
                # The file's own stem. A Sentinel-1 product names itself after the moment it was
                # acquired, so this and `acquired_at` agree — but they are read from different
                # places on purpose: the tag is the product's statement about itself and the name
                # is what this archive happens to call it, and it is the second one a reader needs
                # to open the scene again.
                name=path.stem,
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
