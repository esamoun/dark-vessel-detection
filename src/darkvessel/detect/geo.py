"""Pixel to ground coordinates.

Converts detections from image pixels to projected coordinates, and writes georeferenced vector
output that opens directly in QGIS. Silent errors here produce plausible detections in the wrong
place, which is worse than a crash. Geometry-critical: covered by tests.

Detections arrive here already in scene coordinates: the tile-local step, and the reconciliation
of what overlapping tiles saw twice, are done by the time this module sees them. One transform
places the whole scene, so there is exactly one conversion and it happens here.
"""

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd

from darkvessel.data.scene import Scene
from darkvessel.detect.detector import PixelDetection

DETECTIONS_LAYER = "detections"


def to_ground(detections: Sequence[PixelDetection], scene: Scene) -> gpd.GeoDataFrame:
    """Place pixel detections on the ground, in the scene's CRS.

    A pixel index addresses the centre of that pixel, while an affine transform maps pixel
    *corners*; the half-pixel shift here is the difference between a detection reported at its
    centre and one reported half a pixel up and to the left.
    """
    corners = [(det.col + 0.5, det.row + 0.5) for det in detections]
    ground = [scene.transform * corner for corner in corners]

    return gpd.GeoDataFrame(
        {"score": [det.score for det in detections]},
        geometry=gpd.points_from_xy(
            [x for x, _ in ground],
            [y for _, y in ground],
        ),
        crs=scene.crs,
    )


def write_detections(detections: gpd.GeoDataFrame, path: Path) -> None:
    """Write detections as a GeoPackage layer, which QGIS opens without a conversion step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    detections.to_file(path, layer=DETECTIONS_LAYER, driver="GPKG")
