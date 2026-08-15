"""The trained model behind the contract the stand-in satisfies.

Nothing here asserts how well it detects. A model is evaluated, not asserted — the rule
`test_pipeline.py` states, and the reason `test_training_run.py` pins no precision either. What
is asserted is that it satisfies the protocol, that it refuses a checkpoint built for a different
detector, and that a hole can never come back as a target.

The weights these tests load are untrained. Nothing below depends on the model being any good,
and a test that fetched 160 MB of COCO weights is not a test anyone runs.

Skipped where torch is not installed, which includes CI: the chain's acceptance condition is that
it installs and runs without a framework.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from darkvessel.detect.amplitude import DecibelStretch  # noqa: E402
from darkvessel.detect.detector import PixelDetection  # noqa: E402
from darkvessel.detect.model import detector_model  # noqa: E402
from darkvessel.detect.trained import TrainedDetector  # noqa: E402

# Small enough that a ResNet-50 FPN runs on a laptop CPU inside a test, and the same size
# test_training_run.py trains at.
TILE_PX = 64
ANCHORS = ((8,), (16,), (32,), (64,), (128,))

# Round numbers, not the shipped window. What the shipped one is belongs in the config.
STRETCH = DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-22.0)

BUILT = {
    "tile_px": TILE_PX,
    "anchor_sizes": ANCHORS,
    "seed": 7,
    "pretrained": False,
    "trainable_backbone_layers": 3,
}


def write_checkpoint(directory: Path, built: dict | None = BUILT) -> Path:
    model = detector_model(tile_px=TILE_PX, seed=7, anchor_sizes=ANCHORS, pretrained=False)
    state: dict = {"epoch": 1, "model": model.state_dict()}
    if built is not None:
        state["built"] = built

    path = directory / "epoch-001.pt"
    torch.save(state, path)
    return path


def sea() -> np.ndarray:
    """A tile of open water in decibels, with this project's own scene's statistics."""
    return (
        np.random.default_rng(20260815).normal(-21.84, 2.30, (TILE_PX, TILE_PX)).astype(np.float32)
    )


def detector_at(path: Path, score_threshold: float = 0.75) -> TrainedDetector:
    return TrainedDetector(
        checkpoint=path,
        stretch=STRETCH,
        score_threshold=score_threshold,
        tile_px=TILE_PX,
        anchor_sizes=ANCHORS,
        device=torch.device("cpu"),
    )


def test_it_satisfies_the_detector_contract(tmp_path):
    found = detector_at(write_checkpoint(tmp_path))(sea())

    assert isinstance(found, list)
    assert all(isinstance(detection, PixelDetection) for detection in found)


def test_a_tile_holding_nodata_still_answers(tmp_path):
    """One NaN reaching the network propagates through every convolution that touches it and
    empties the tile. The stretch fills the holes before the model ever sees them."""
    image = sea()
    image[:8, :8] = np.nan

    found = detector_at(write_checkpoint(tmp_path), score_threshold=0.0)(image)
    assert all(np.isfinite(detection.score) for detection in found)


def test_nothing_is_reported_from_inside_a_hole(tmp_path):
    image = np.full((TILE_PX, TILE_PX), np.nan, dtype=np.float32)

    assert detector_at(write_checkpoint(tmp_path), score_threshold=0.0)(image) == []


def test_a_checkpoint_built_for_other_anchors_is_refused(tmp_path):
    stock = {**BUILT, "anchor_sizes": ((32,), (64,), (128,), (256,), (512,))}

    with pytest.raises(ValueError, match="anchor"):
        detector_at(write_checkpoint(tmp_path, built=stock))


def test_a_checkpoint_built_for_another_tile_size_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tile_px"):
        detector_at(write_checkpoint(tmp_path, built={**BUILT, "tile_px": 800}))


def test_a_checkpoint_from_before_the_build_block_still_loads(tmp_path):
    """epoch-012.pt predates train.py recording it, and the run config restates the values."""
    assert detector_at(write_checkpoint(tmp_path, built=None)) is not None


def test_the_score_threshold_is_what_decides_what_is_reported(tmp_path):
    """A detector has a precision at a confidence, not a precision. Everything at 0.0, nothing
    above 1.0 — asserted as an ordering rather than as a count, which would be a claim about
    how well an untrained model detects."""
    detector = write_checkpoint(tmp_path)
    everything = detector_at(detector, score_threshold=0.0)(sea())
    nothing = detector_at(detector, score_threshold=1.01)(sea())

    assert len(nothing) == 0
    assert len(everything) >= len(nothing)
