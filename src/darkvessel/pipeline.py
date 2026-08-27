"""The seam: a scene in, a detector injected, classified detections out.

This is the single entry point of the chain and the only high seam in the project. The detector
arrives as a parameter rather than an import, which is what lets the whole pipeline run — and be
tested — with a deterministic substitute, no weights, no GPU and no network.

The embedder arrives the same way and is optional, which is a different claim. A detector is what
the chain is for; a representation of what it found is an answer to a second question — which of
these objects resemble one another — and a run that never asks it must be unaffected by its
existence. So the parameter defaults to None, the crops are never cut when it is, and the layer
that comes out is byte for byte the layer that came out before this stage existed.
"""

from datetime import timedelta

import geopandas as gpd

from darkvessel.context.gee_layers import without_context
from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import to_ground
from darkvessel.detect.infer import detect_scene
from darkvessel.embed.crops import crops_for
from darkvessel.embed.embedder import Embedder, attach
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.match import classify
from darkvessel.fusion.register import Register, without_a_register


def run(
    *,
    scene: Scene,
    ais: gpd.GeoDataFrame | None,
    detector: Detector,
    tiling: Tiling,
    tolerance_m: float,
    max_gap: timedelta,
    geometry: Geometry | None = None,
    embedder: Embedder | None = None,
    structures: Register | None = None,
) -> gpd.GeoDataFrame:
    """Run the chain over one scene and return its detections, georeferenced.

    Args:
        scene: The radar scene, with the transform and acquisition time that place it.
        ais: Declared positions to match against, or None when none are available.
        detector: The injected detector.
        tiling: How the scene is cut up for the detector, and put back together afterwards.
        tolerance_m: How far a declared position may sit from a detection and still explain it.
        max_gap: The widest bracket of AIS reports a position may be interpolated across.
        geometry: The orbit the scene was acquired from, which decides how far a moving vessel is
            drawn from where it actually was. None applies no correction — the right answer for a
            synthetic scene, which has no satellite behind it.
        embedder: What to describe each detection with, or None to describe none. Optional
            because the representation is a second question asked of the same detections, and a
            chain that does not ask it must not need a framework to answer it.
        structures: The fixed structures this run will not call dark vessels, or None to
            exclude nothing. Optional the way the embedder is, and unlike the embedder it
            changes the verdict rather than adding to it — so a run without one writes the same
            column, empty, and says in the layer that it excluded nothing rather than saying
            nothing at all.

    Returns:
        A GeoDataFrame in the scene's CRS, one row per detection.
    """
    found = detect_scene(scene.image, detector, tiling)
    detections = classify(
        to_ground(found, scene), ais, scene.acquired_at, tolerance_m, max_gap, geometry
    )
    # After the matching, never before it. A detection a declared vessel explains is a vessel,
    # whatever else stands at that coordinate — see `Register.mark`, which keeps the distance on
    # the row anyway so that a ship moored at a mast is visible rather than merely handled.
    detections = (
        without_a_register(detections) if structures is None else structures.mark(detections)
    )
    # Empty, always, and never filled in here. The contextual variables come from Earth Engine,
    # and this function is the one place in the project guaranteed to run with no network behind
    # it — `darkvessel context` samples them afterwards, into the layer this one writes. What the
    # chain owes them is the columns: a layer whose schema depends on whether a credentialed
    # stage was run is a layer that cannot be stacked with the one beside it.
    detections = without_context(detections)
    if embedder is None:
        return detections

    # Cut from the pixels the detector reported, not from the ground coordinates it was placed
    # at: a crop is a window on the image, and going back through the transform to find it again
    # would be a second conversion where the project has exactly one.
    crops = crops_for(scene.image, found, crop_px=embedder.crop_px, margin_px=embedder.margin_px)
    return attach(detections, embedder(crops))
