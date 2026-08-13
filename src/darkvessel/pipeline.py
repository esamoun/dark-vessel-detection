"""The seam: a scene in, a detector injected, classified detections out.

This is the single entry point of the chain and the only high seam in the project. The detector
arrives as a parameter rather than an import, which is what lets the whole pipeline run — and be
tested — with a deterministic substitute, no weights, no GPU and no network.
"""

import geopandas as gpd

from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import to_ground
from darkvessel.detect.infer import detect_scene
from darkvessel.fusion.match import classify


def run(
    *,
    scene: Scene,
    ais: gpd.GeoDataFrame | None,
    detector: Detector,
    tiling: Tiling,
    tolerance_m: float,
) -> gpd.GeoDataFrame:
    """Run the chain over one scene and return its detections, georeferenced.

    Args:
        scene: The radar scene, with the transform and acquisition time that place it.
        ais: Declared positions to match against, or None when none are available.
        detector: The injected detector.
        tiling: How the scene is cut up for the detector, and put back together afterwards.
        tolerance_m: How far a declared position may sit from a detection and still explain it.

    Returns:
        A GeoDataFrame in the scene's CRS, one row per detection.
    """
    detections = to_ground(detect_scene(scene.image, detector, tiling), scene)
    return classify(detections, ais, scene.acquired_at, tolerance_m)
