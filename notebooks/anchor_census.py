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


def census(sizes: tuple[tuple[int, ...], ...], refs: list) -> None:
    anchors, per_level = anchors_for(sizes)
    # Torchvision's own thresholds, and its own guarantee that every box gets at least one anchor
    # however poor the overlap. That guarantee is why "zero positives" never happens and why the
    # count, not the presence, is the thing worth measuring.
    matcher = Matcher(0.7, 0.3, allow_low_quality_matches=True)

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
            by_level[int((boundaries <= index).sum())] += 1

        # Boxes whose best anchor never reached 0.7 and were matched only by the low-quality rule.
        rescued += int((quality.max(dim=1).values < 0.7).sum())

    total = sum(positives_per_tile)
    tiles = len(positives_per_tile)
    print(f"  ship-bearing tiles: {tiles}")
    print(
        f"  positive anchors per tile: mean {total / max(tiles, 1):.1f}, "
        f"min {min(positives_per_tile)}, max {max(positives_per_tile)}"
    )
    print(f"  boxes matched only by allow_low_quality_matches: {rescued}")
    print(f"  by pyramid level: {dict(sorted(by_level.items()))}")
    # 256 anchors are sampled at a ceiling of 50% positive, so 128 are asked for.
    print(
        f"  realised positive fraction against a batch of 256: "
        f"{min(total / max(tiles, 1), 128.0) / 256:.3f}"
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
