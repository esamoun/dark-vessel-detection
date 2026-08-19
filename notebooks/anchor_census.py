"""How many anchors ever match a ship, and how many the sampler can therefore find.

Two rungs of the ladder in issue #11 turn on numbers nobody has measured. The first is the anchor
sizing: the stock set starts at 32 px, a 320 m vessel at 10 m, and the argument for moving it is
currently an argument from arithmetic rather than from a count. The second is the imbalance, and
it is the one that is easy to get wrong. Faster R-CNN already subsamples — 256 anchors at a
ceiling of 50% positive — so the 1000:1 imbalance of a dense detector does not exist here. What
may exist is subtler: on a tile holding three ships of four pixels there may be nowhere near 128
positive anchors to be had, in which case `rpn_positive_fraction` is not the lever at all. It is a
ceiling, not a target, and the sampler takes min(available, requested). The lever would then be
`rpn_batch_size_per_image`, moved *down*, so the few positives are not drowned.

The prediction, written before this was run: a realised positive fraction near 1%, not 50%. If the
census contradicts it, that is recorded as a contradiction — see README.md on the recall
prediction this project got wrong before.

Costs no GPU quota. It reads boxes, not images, and runs on a CPU session in minutes.

    python3 notebooks/anchor_census.py
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

# A private module of torchvision's, and named here rather than reimplemented: the point of this
# census is what torchvision's own matcher does with these boxes, not what a copy of it would.
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

from darkvessel.detect.dataset import Layout, catalogue, split_by_scene
from darkvessel.detect.model import ANCHOR_SIZES, detector_model

ROOT = Path("/kaggle/input/ls-ssdd-v10/LS-SSDD-v1.0-OPEN")
LAYOUT = Layout(images="JPEGImages", annotations="Annotations", image_suffix=".jpg")
TILE_PX = 800

# What rung 2 proposes, against what the baseline ships.
CANDIDATES = {
    "stock (the baseline)": ANCHOR_SIZES,
    "small (rung 2)": ((4,), (8,), (16,), (32,), (64,)),
}

# Torchvision's own RPN defaults (see `detector_model`'s `rpn_batch_size_per_image` and
# `rpn_positive_fraction`), named here rather than left as bare literals so the realised-fraction
# line below can show the reader where 128 comes from instead of asserting it.
RPN_BATCH_SIZE_PER_IMAGE = 256
RPN_POSITIVE_FRACTION = 0.5
RPN_POSITIVE_CAP = RPN_BATCH_SIZE_PER_IMAGE * RPN_POSITIVE_FRACTION

# Matcher's own high threshold. Named once and reused for the "rescued only by low-quality
# matching" count below, so the two cannot drift apart if this is ever retuned.
HIGH_IOU_THRESHOLD = 0.7
LOW_IOU_THRESHOLD = 0.3


def anchors_for(sizes: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, list[int]]:
    """Every anchor one tile offers, and how many of them belong to each pyramid level.

    The feature-map shapes come from running the backbone once on a dummy tile rather than from
    the stride arithmetic, because the arithmetic is a claim about torchvision's FPN and this is
    the measurement of it.
    """
    model = detector_model(tile_px=TILE_PX, seed=0, anchor_sizes=sizes, pretrained=False).eval()

    blank = torch.zeros(1, 3, TILE_PX, TILE_PX)
    with torch.no_grad():
        features = list(model.backbone(blank).values())

    images = ImageList(blank, [(TILE_PX, TILE_PX)])
    per_level = [
        level.shape[-2] * level.shape[-1] * len(model.rpn.anchor_generator.aspect_ratios[0])
        for level in features
    ]
    return model.rpn.anchor_generator(images, features)[0], per_level


@dataclass(frozen=True)
class CensusResult:
    """What one census run found, kept as data rather than only as the lines it prints.

    Printing alone is what let the mean-then-cap defect ship silently in the first place: the
    only check that ever ran against it was a human reading a terminal. Everything `_report`
    prints below is read from here, not recomputed, so a test can pin the same numbers a person
    reads.
    """

    ship_bearing_tiles: int
    positives_per_tile: list[int]
    rescued: int
    by_level: dict[int, int]
    realised_fraction: float


def _level_of(index: int, boundaries: torch.Tensor) -> int:
    """Which pyramid level an anchor index falls in, given each level's cumulative anchor count.

    `boundaries[i]` is the index one past the last anchor of level `i`. `<=` is load-bearing, not
    `<`: an index equal to a boundary is the *first* anchor of the next level, not the last of the
    one the boundary closes, because `per_level` counts anchors and `cumsum` turns those counts
    into the index one past where each level ends.
    """
    return int((boundaries <= index).sum())


def _measure(anchors: torch.Tensor, per_level: list[int], refs: list) -> CensusResult:
    """The counting core of `census`, taking anchors directly rather than a size spec.

    Split out from `census` so a caller can hand it a small hand-built anchor tensor instead of
    paying for a ResNet-50 forward pass to get one — which is what a test wants when what it is
    checking is this function's arithmetic rather than torchvision's anchor geometry, already
    checked by hand against the installed library once, elsewhere.
    """
    # Torchvision's own thresholds, and its own guarantee that every box gets at least one anchor
    # however poor the overlap. That guarantee is why "zero positives" never happens and why the
    # count, not the presence, is the thing worth measuring.
    matcher = Matcher(HIGH_IOU_THRESHOLD, LOW_IOU_THRESHOLD, allow_low_quality_matches=True)

    boundaries = torch.tensor(per_level).cumsum(0)
    positives_per_tile = []
    by_level: Counter[int] = Counter()
    rescued = 0

    for ref in refs:
        boxes = torch.tensor([box.to_xyxy() for box in ref.boxes], dtype=torch.float32)
        if not len(boxes):
            continue

        quality = box_iou(boxes, anchors)
        matched = matcher(quality)
        positive = (matched >= 0).nonzero().flatten()
        positives_per_tile.append(len(positive))

        for index in positive.tolist():
            by_level[_level_of(index, boundaries)] += 1

        # Boxes whose best anchor never reached the high threshold and were matched only by the
        # low-quality rule.
        rescued += int((quality.max(dim=1).values < HIGH_IOU_THRESHOLD).sum())

    tiles = len(positives_per_tile)
    # The sampler draws once per image and caps *that image's* positives at RPN_POSITIVE_CAP
    # before the batch is filled with background — it never sees the other tiles, so it cannot
    # average across them first. Capping each tile and then averaging is the model of what the
    # sampler actually does; capping the mean is a different quantity, and because x -> min(x,
    # cap) is concave, Jensen's inequality makes it a strictly larger one whenever any tile
    # exceeds the cap. That direction is exactly wrong for a census meant to expose a sampler
    # running out of positives, so the cap is applied per tile, not to the mean.
    capped_mean = sum(min(x, RPN_POSITIVE_CAP) for x in positives_per_tile) / max(tiles, 1)

    result = CensusResult(
        ship_bearing_tiles=tiles,
        positives_per_tile=positives_per_tile,
        rescued=rescued,
        by_level=dict(sorted(by_level.items())),
        realised_fraction=capped_mean / RPN_BATCH_SIZE_PER_IMAGE,
    )
    return result


def census(sizes: tuple[tuple[int, ...], ...], refs: list) -> CensusResult:
    anchors, per_level = anchors_for(sizes)
    result = _measure(anchors, per_level, refs)
    _report(result)
    return result


def _report(result: CensusResult) -> None:
    total = sum(result.positives_per_tile)
    tiles = result.ship_bearing_tiles
    print(f"  ship-bearing tiles: {tiles}")
    print(
        f"  positive anchors per tile: mean {total / max(tiles, 1):.1f}, "
        f"min {min(result.positives_per_tile)}, max {max(result.positives_per_tile)}"
    )
    print(f"  boxes matched only by allow_low_quality_matches: {result.rescued}")
    print(f"  by pyramid level: {result.by_level}")
    print(
        f"  realised positive fraction against a batch of {RPN_BATCH_SIZE_PER_IMAGE}: "
        f"{result.realised_fraction:.3f}"
    )


def main() -> None:
    refs = catalogue(ROOT, LAYOUT)
    training, _ = split_by_scene(refs)
    ship_bearing = [ref for ref in training if ref.boxes]

    sizes = sorted(
        max(box.to_xyxy()[2] - box.to_xyxy()[0], box.to_xyxy()[3] - box.to_xyxy()[1])
        for ref in ship_bearing
        for box in ref.boxes
    )
    print(f"{len(sizes)} ships over {len(ship_bearing)} training tiles")
    print(
        f"longest side in pixels: p05 {sizes[len(sizes) // 20]:.1f}, "
        f"median {sizes[len(sizes) // 2]:.1f}, p95 {sizes[-len(sizes) // 20]:.1f}"
    )

    for label, candidate in CANDIDATES.items():
        print(f"\n{label}: {candidate}")
        census(candidate, ship_bearing)


if __name__ == "__main__":
    main()
