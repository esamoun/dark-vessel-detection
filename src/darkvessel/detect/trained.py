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
        stem: str = "repeat",
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
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
            stem: The input stage the checkpoint was trained with. Checked against its record
                too, and defaulting to the repeat because every checkpoint written before stems
                existed was trained on three repeated channels.
            rpn_fg_iou_thresh: The IoU at which an anchor counted as a positive example of a ship
                while these weights were being fitted, and `rpn_bg_iou_thresh` the one below
                which it counted as a negative. Both default to torchvision's, because every
                checkpoint written before 2026-08-29 was trained under them.
            rpn_bg_iou_thresh: See above.
            device: Where to run. The GPU if there is one.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _check_built(
            state.get("built"),
            tile_px=tile_px,
            anchor_sizes=anchor_sizes,
            stem=stem,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
        )

        # `pretrained=False` because every weight is about to be overwritten by the load below,
        # and because fetching COCO weights would put a network on the path of a command that
        # must not need one. The seed is irrelevant for the same reason: the head it initialises
        # does not survive.
        model = detector_model(
            tile_px=tile_px,
            seed=0,
            anchor_sizes=anchor_sizes,
            stem=stem,
            pretrained=False,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
        )
        model.load_state_dict(state["model"])

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.stretch = stretch
        self.score_threshold = score_threshold
        self.stem = stem

    def __call__(self, image: np.ndarray) -> list[PixelDetection]:
        """One tile of decibels, as targets in that tile's own pixel coordinates.

        The guard runs against `image` as it arrived, holes still NaN, rather than against the
        stretched copy which no longer has any. That is why the stretch returns a copy instead of
        filling in place.
        """
        with torch.no_grad():
            tile = as_model_input(self.stretch(image), self.stem).to(self.device)
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
    stem: str = "repeat",
    rpn_fg_iou_thresh: float = 0.7,
    rpn_bg_iou_thresh: float = 0.3,
) -> None:
    """Refuse a checkpoint built for a detector other than the one being constructed.

    A whole missing block is allowed for exactly one reason: the first trained checkpoint predates
    `train.py` writing this block, and its build parameters are restated in the run config with
    the training config named beside them. Every checkpoint written since carries its own, and a
    disagreement is an error rather than a warning — a model looking for the wrong size of ship
    does not fail, it returns detections, in plausible places, with scores. Individual keys added
    after a checkpoint was written read as the value that checkpoint was trained under, which is
    torchvision's default in each case; that is stated at each of them rather than in general.

    The RPN's two IoU thresholds are checked on a different footing from the three above them, and
    it is worth saying which. `tile_px`, `anchor_sizes` and `stem` change what the loaded model
    *does* — the first resizes every tile, the second looks for ships of another size, the third
    takes another number of channels. The thresholds change none of that: `RegionProposalNetwork`
    consults its matcher only while training, so a checkpoint fitted at 0.7 and one fitted at 0.25
    are, at inference, the same model run the same way. What they are not is the same weights. The
    refusal here is therefore about provenance rather than behaviour: it is what stops a run config
    from naming one training regime while loading the checkpoint of another, which no number
    downstream of it would ever contradict.
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

    # Absent from every checkpoint written before 2026-08-17, all of which were trained on three
    # repeated channels — so silence means "repeat" rather than "unknown".
    recorded_stem = built.get("stem", "repeat")
    if recorded_stem != stem:
        raise ValueError(
            f"the checkpoint was built with the {recorded_stem!r} stem and this run asks for "
            f"{stem!r}; the two take a different number of channels"
        )

    # Absent from every checkpoint written before 2026-08-29, all of which were trained under
    # torchvision's own thresholds — so silence means 0.7 and 0.3 rather than "unknown". Read
    # through `.get` and compared as floats, because a threshold reaches this from a YAML file on
    # one side and from a JSON round-trip on the other.
    for key, asked, default in (
        ("rpn_fg_iou_thresh", rpn_fg_iou_thresh, 0.7),
        ("rpn_bg_iou_thresh", rpn_bg_iou_thresh, 0.3),
    ):
        recorded_threshold = float(built.get(key, default))
        if recorded_threshold != float(asked):
            raise ValueError(
                f"the checkpoint was built with {key} {recorded_threshold} and this run asks for "
                f"{float(asked)}; the threshold is inert at inference, so these are the same "
                "model fitted under two different regimes and nothing downstream would say so"
            )


def _as_sizes(sizes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """One shape for anchor sizes, so a list out of YAML and a tuple out of a checkpoint compare."""
    return tuple(tuple(int(size) for size in level) for level in sizes)
