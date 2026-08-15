"""The trained detector, behind the contract the stand-in satisfies.

This is what the seam was built for. The pipeline takes a detector as a parameter and never
learns which one it got, so swapping the model in is one branch in the command that builds it and
nothing else: no stage of the chain changes, and the deterministic substitute goes on working
beside it.

Three things stand between a checkpoint and a scene, and all three are answered here or in
`amplitude.py` rather than anywhere the pipeline can see:

* the chain deals in calibrated decibels and the model was fitted on amplitude in 0..1;
* a product's nodata arrives as NaN, which a threshold ignores and a network does not;
* a checkpoint does not record its own anchors, so it can be loaded into a detector looking for
  ships of the wrong size without a word.

torch is imported at module level, which is safe because nothing imports this module unless a
config asks for it — `cli._detector_from` imports it inside its branch, the way `cli._train`
imports torch inside its command. `darkvessel run` with the stand-in still needs no framework, no
GPU and no network, which is the chain's acceptance condition.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkvessel.detect.amplitude import DecibelStretch, without_holes
from darkvessel.detect.detector import PixelDetection
from darkvessel.detect.model import as_model_input, detections_from, detector_model


class TrainedDetector:
    """A checkpoint, run over one tile at a time, reporting what the chain's contract states."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        stretch: DecibelStretch,
        score_threshold: float,
        tile_px: int,
        anchor_sizes: tuple[tuple[int, ...], ...],
        device: torch.device | None = None,
    ) -> None:
        """Load the weights and put the model into the state it answers from.

        Args:
            checkpoint: What a training run wrote. Only the weights and the build block are read;
                the optimiser state, where there is one, is not needed to run a model.
            stretch: How decibels become the amplitude this model was fitted on.
            score_threshold: The confidence below which a detection is not reported. A detector
                has a precision *at* a confidence, and this is where that is chosen; the training
                run reports the whole table so the choice can be made against numbers.
            tile_px: The side of the tiles this model runs on. It has to be the side the chain
                cuts, or torchvision resizes between the two — `cli.check_tile_size` refuses that
                before anything here is loaded.
            anchor_sizes: One tuple per pyramid level. Checked against the checkpoint's own
                record, because anchors leave no trace in a state dict.
            device: Where to run. The GPU if there is one.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _check_built(state.get("built"), tile_px=tile_px, anchor_sizes=anchor_sizes)

        # `pretrained=False` because every weight is about to be overwritten by the load below,
        # and because fetching COCO weights would put a network on the path of a command that
        # must not need one. The seed is irrelevant for the same reason: the head it initialises
        # does not survive.
        model = detector_model(tile_px=tile_px, seed=0, anchor_sizes=anchor_sizes, pretrained=False)
        model.load_state_dict(state["model"])

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.stretch = stretch
        self.score_threshold = score_threshold

    def __call__(self, image: np.ndarray) -> list[PixelDetection]:
        """One tile of decibels, as targets in that tile's own pixel coordinates.

        The guard runs against `image` as it arrived, holes still NaN, rather than against the
        stretched copy which no longer has any. That is why the stretch returns a copy instead of
        filling in place.
        """
        with torch.no_grad():
            tile = as_model_input(self.stretch(image)).to(self.device)
            output = self.model([tile])[0]

        found = [
            detection
            for detection in detections_from(output)
            if detection.score >= self.score_threshold
        ]
        return without_holes(found, image)


def _check_built(
    built: dict[str, Any] | None,
    *,
    tile_px: int,
    anchor_sizes: tuple[tuple[int, ...], ...],
) -> None:
    """Refuse a checkpoint built for a detector other than the one being constructed.

    Silence is allowed for exactly one reason: the first trained checkpoint predates `train.py`
    writing this block, and its build parameters are restated in the run config with the training
    config named beside them. Every checkpoint written since carries its own, and a disagreement
    is an error rather than a warning — a model looking for the wrong size of ship does not fail,
    it returns detections, in plausible places, with scores.
    """
    if built is None:
        return

    if int(built["tile_px"]) != tile_px:
        raise ValueError(
            f"the checkpoint was built with tile_px {built['tile_px']} and this run asks for "
            f"{tile_px}; the model would resize every tile between the two"
        )

    recorded, asked = _as_sizes(built["anchor_sizes"]), _as_sizes(anchor_sizes)
    if recorded != asked:
        raise ValueError(
            f"the checkpoint was built with anchor_sizes {recorded} and this run asks for "
            f"{asked}; anchors are not weights, so this would load cleanly and then look for "
            "ships of the wrong size without saying so"
        )


def _as_sizes(sizes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """One shape for anchor sizes, so a list out of YAML and a tuple out of a checkpoint compare."""
    return tuple(tuple(int(size) for size in level) for level in sizes)
