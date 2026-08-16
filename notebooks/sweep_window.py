"""Choosing the one number the measurement could not fix: how wide the window is.

Where the sea sits was measured — LS-SSDD's offshore held-out tiles put it at 0.2000, and that
end of the mapping is settled. How wide the window is could not be, and the reason is physical
rather than procedural: matching the *spread* of the two seas would set the width from how grainy
each product is, and LS-SSDD's sea has a relative spread near 0.8 against this chain's 0.27 in
the same units. That ratio is a difference in how many looks were averaged, not in how the bytes
were made. Anchoring on ship brightness instead swings the other way — with hulls at p95 it asks
for a window of a hundred decibels. Neither statistic can separate the stretch from the sensor.

So the ceiling is one free parameter, and in this project a free parameter is settled by
measurement against something known rather than by eye. What is known here is the AIS: twelve
vessels declared themselves inside this scene at the instant it was acquired. A window is scored
by how many of those twelve the trained detector recovers, at the tolerance the fusion stage
already applies to it.

This is tuning on the scene the result is then reported on, and it is not pretending otherwise.
Twelve vessels on one acquisition is thin, the choice will be recorded as thin in
docs/decisions.md, and the numbers that carry weight remain the held-out LS-SSDD table.

    python3 notebooks/sweep_window.py
"""

from datetime import timedelta
from pathlib import Path

import torch

from darkvessel.data.ais import load_ais
from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.amplitude import DecibelStretch
from darkvessel.detect.geo import to_ground
from darkvessel.detect.infer import detect_scene
from darkvessel.detect.trained import TrainedDetector
from darkvessel.fusion.match import MATCHED, classify

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "data" / "real" / "kattegat-lane.tif"
AIS = REPO / "data" / "real" / "kattegat-lane-ais.csv"
CHECKPOINT = REPO / "models" / "epoch-012.pt"

# Measured on 2,234 offshore held-out tiles of LS-SSDD, outside the annotated boxes, tile by tile
# rather than pooled. This is the end of the mapping that is not being swept.
LS_SSDD_SEA = 0.2000

# The scene's own sea, from `amplitude.sea_level` on the product.
SCENE_SEA_DB = -21.84

# What the chain already uses everywhere else. Not swept: changing the tolerance alongside the
# window would make it impossible to say which of the two moved a number.
TOLERANCE_M = 200.0
MAX_GAP = timedelta(seconds=600)

TILING = Tiling(size_px=800, overlap_px=64)
ANCHORS = ((32,), (64,), (128,), (256,), (512,))

SPANS = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0)
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)


def window_of(span: float) -> DecibelStretch:
    """The window of this width whose sea lands where LS-SSDD's sea was."""
    floor = SCENE_SEA_DB - LS_SSDD_SEA * span
    return DecibelStretch(floor_db=floor, ceiling_db=floor + span, sea_db=SCENE_SEA_DB)


def main() -> None:
    scene = Scene.from_geotiff(SCENE)
    ais = load_ais(AIS, crs=scene.crs)
    declared = int(ais["mmsi"].nunique())

    print(f"{SCENE.name}: {scene.image.shape[0]} x {scene.image.shape[1]} px, {scene.acquired_at}")
    print(f"{declared} vessels declared themselves inside it at that instant\n")
    print(
        f"{'span':>5}  {'window (dB)':>18}  {'score':>5}  {'found':>5}  {'matched':>7}  {'dark':>4}"
    )

    for span in SPANS:
        stretch = window_of(span)
        detector = TrainedDetector(
            checkpoint=CHECKPOINT,
            stretch=stretch,
            # Everything, so one pass over the scene answers for every operating point below.
            # The score of a detection does not depend on which threshold it is later compared
            # against, and neither claiming a tile nor discarding a hole looks at it.
            score_threshold=0.0,
            tile_px=TILING.size_px,
            anchor_sizes=ANCHORS,
            device=torch.device("cpu"),
        )
        pixels = detect_scene(scene.image, detector, TILING)

        for threshold in THRESHOLDS:
            kept = [detection for detection in pixels if detection.score >= threshold]
            out = classify(to_ground(kept, scene), ais, scene.acquired_at, TOLERANCE_M, MAX_GAP)
            matched = int(out[out["status"] == MATCHED]["mmsi"].nunique()) if len(out) else 0
            window = f"{stretch.floor_db:6.2f} .. {stretch.ceiling_db:6.2f}"

            print(
                f"{span:5.0f}  {window:>18}  {threshold:5.2f}  {len(kept):5d}  "
                f"{matched:3d}/{declared:<3d}  {len(kept) - matched:4d}"
            )
        print()


if __name__ == "__main__":
    main()
