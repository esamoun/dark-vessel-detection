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

**The threshold sweep, added 2026-08-29 for issue #24.** Both RPN rungs the ladder ran were
rejected, and the census's own explanation of why survived them: almost no ship reaches 0.7 IoU
against any anchor, so the stock set works through `allow_low_quality_matches` rather than through
fitting the targets. That makes the foreground threshold itself the untested variable, and it is
the sixth rung. What the sweep below adds is the number that rung has to be set from: for each
candidate threshold, how many ships have a genuine match rather than a rescued one, and what the
realised positive fraction becomes. `best_iou` is the whole of it — every ship's best overlap with
any anchor, from which the rescue share at *any* threshold is a count rather than another run.

The sweep moves the foreground threshold only. The background one does not enter any number here:
`Matcher` uses it to separate an ignored anchor from a negative one, and this census counts
neither. It still has to move in the run, because `Matcher` refuses a background threshold above
the foreground one — which is what puts the floor of a one-key change at 0.3.

Costs no GPU quota. It reads boxes, not images, and runs on a CPU session in minutes.

    python3 notebooks/anchor_census.py
"""

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import torch

# A private module of torchvision's, and named here rather than reimplemented: the point of this
# census is what torchvision's own matcher does with these boxes, not what a copy of it would.
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

from darkvessel.detect.dataset import Layout, catalogue, split_by_scene
from darkvessel.detect.model import ANCHOR_SIZES, detector_model

# The Kaggle mirror (petrarodriguez/ls-ssdd-v1-0), not LS-SSDD's own layout: images already
# split into train and test directories, annotations all in one. Both image directories are
# named here for symmetry with configs/train.yaml, not because the census needs the held-out
# half: `split_by_scene` below keeps only the training side and the held-out one is discarded,
# so its tiles are globbed and parsed for nothing. Naming it costs one parse and keeps this
# script's Layout the same one the training run actually uses, rather than a second Layout that
# could quietly drift from it — see docs/decisions.md, 2026-08-20.
ROOT = Path("/kaggle/input/datasets/petrarodriguez/ls-ssdd-v1-0")
LAYOUT = Layout(
    images=[
        "JPEGImages_sub_train/JPEGImages_sub_train",
        "JPEGImages_sub_test/JPEGImages_sub_test",
    ],
    annotations="Annotations_sub/Annotations_sub",
    image_suffix=".jpg",
)
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

# The foreground thresholds the sweep reports, for issue #24's sixth rung. 0.7 is torchvision's
# own and the one every run so far was fitted under, so it is the row the others are read against
# rather than a candidate. 0.3 is the floor a one-key change can reach, `Matcher` refusing a
# background threshold above the foreground one. 0.25 is the value the census's own worked example
# names — a 16 px ship inside a 32 px anchor overlaps it at `256/1024`, identically for every
# anchor containing it — and everything below is there because the median ship's *area* is smaller
# than 16 x 16 and nobody has counted how much smaller.
SWEEP_THRESHOLDS = (0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05)

# Where the best-IoU distribution is read at. Chosen to answer "what threshold would give most
# ships a genuine match", which a mean cannot: the distribution is what the sweep is for.
PERCENTILES = (5, 25, 50, 75, 95)


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
    # What the matcher was built with. Recorded rather than assumed, because a sweep produces one
    # of these per threshold and a table of five rows that does not say which row is which is a
    # table of one number written five times.
    fg_iou_thresh: float = HIGH_IOU_THRESHOLD
    # Every ship's best overlap with any anchor in the set, one entry per box across every tile.
    # `rescued` is a count off this list — the boxes below `fg_iou_thresh` — so the two cannot
    # disagree, and the rescue share at a threshold nobody swept is a count rather than a rerun.
    best_iou: list[float] = field(default_factory=list)


def _level_of(index: int, boundaries: torch.Tensor) -> int:
    """Which pyramid level an anchor index falls in, given each level's cumulative anchor count.

    `boundaries[i]` is the index one past the last anchor of level `i`. `<=` is load-bearing, not
    `<`: an index equal to a boundary is the *first* anchor of the next level, not the last of the
    one the boundary closes, because `per_level` counts anchors and `cumsum` turns those counts
    into the index one past where each level ends.
    """
    return int((boundaries <= index).sum())


def _measure(
    anchors: torch.Tensor,
    per_level: list[int],
    refs: list,
    fg_iou_thresh: float = HIGH_IOU_THRESHOLD,
) -> CensusResult:
    """The counting core of `census`, taking anchors directly rather than a size spec.

    Split out from `census` so a caller can hand it a small hand-built anchor tensor instead of
    paying for a ResNet-50 forward pass to get one — which is what a test wants when what it is
    checking is this function's arithmetic rather than torchvision's anchor geometry, already
    checked by hand against the installed library once, elsewhere.

    One threshold, which is the shape every caller before issue #24 wanted. `_measure_at` is the
    same measurement at several at once, over one pass of the boxes.
    """
    return _measure_at(anchors, per_level, refs, (fg_iou_thresh,))[0]


def _measure_at(
    anchors: torch.Tensor,
    per_level: list[int],
    refs: list,
    thresholds: tuple[float, ...],
) -> list[CensusResult]:
    """The census at each of several foreground thresholds, from one pass over the boxes.

    One pass rather than one per threshold because the expensive line is `box_iou`, which is a
    box-by-anchor matrix — two hundred thousand anchors against every ship of every tile — and it
    does not depend on the threshold at all. Only the matcher does, and a matcher applied to a
    matrix already in hand costs nothing. Nine thresholds this way is one census, not nine.

    Only the *foreground* threshold varies. The background one decides whether a non-matching
    anchor is ignored or trained on as background, and this census counts neither, so it would
    move every row of the table by exactly zero. It still has to move in the run — `Matcher`
    refuses a background threshold above the foreground one — and that is said in the sweep's
    report rather than left for a reader to rediscover against a `torch._assert`.
    """
    # Torchvision's own guarantee that every box gets at least one anchor however poor the
    # overlap. That guarantee is why "zero positives" never happens and why the count, not the
    # presence, is the thing worth measuring. The low threshold is torchvision's throughout: it
    # changes nothing this function reports, and varying a number that changes nothing would put
    # a second variable in a table that has one.
    matchers = [
        Matcher(threshold, min(LOW_IOU_THRESHOLD, threshold), allow_low_quality_matches=True)
        for threshold in thresholds
    ]

    boundaries = torch.tensor(per_level).cumsum(0)
    positives_per_tile: list[list[int]] = [[] for _ in thresholds]
    by_level: list[Counter[int]] = [Counter() for _ in thresholds]
    best_iou: list[float] = []

    for ref in refs:
        boxes = torch.tensor([box.to_xyxy() for box in ref.boxes], dtype=torch.float32)
        if not len(boxes):
            continue

        quality = box_iou(boxes, anchors)
        # One entry per ship, kept rather than immediately turned into a count, because the
        # rescue share at every threshold — swept or not — is a count off this list.
        best_iou.extend(quality.max(dim=1).values.tolist())

        for slot, matcher in enumerate(matchers):
            positive = (matcher(quality) >= 0).nonzero().flatten()
            positives_per_tile[slot].append(len(positive))
            for index in positive.tolist():
                by_level[slot][_level_of(index, boundaries)] += 1

    return [
        _result_at(threshold, positives_per_tile[slot], by_level[slot], best_iou)
        for slot, threshold in enumerate(thresholds)
    ]


def _result_at(
    fg_iou_thresh: float,
    positives_per_tile: list[int],
    by_level: Counter[int],
    best_iou: list[float],
) -> CensusResult:
    """One row of the census, from the counts that produced it."""
    tiles = len(positives_per_tile)
    # The sampler draws once per image and caps *that image's* positives at RPN_POSITIVE_CAP
    # before the batch is filled with background — it never sees the other tiles, so it cannot
    # average across them first. Capping each tile and then averaging is the model of what the
    # sampler actually does; capping the mean is a different quantity, and because x -> min(x,
    # cap) is concave, Jensen's inequality makes it a strictly larger one whenever any tile
    # exceeds the cap. That direction is exactly wrong for a census meant to expose a sampler
    # running out of positives, so the cap is applied per tile, not to the mean.
    capped_mean = sum(min(x, RPN_POSITIVE_CAP) for x in positives_per_tile) / max(tiles, 1)

    return CensusResult(
        ship_bearing_tiles=tiles,
        positives_per_tile=positives_per_tile,
        # Boxes whose best anchor never reached the threshold and were matched only by the
        # low-quality rule. Counted off `best_iou` rather than measured a second way, so the
        # number in this field and the distribution beside it cannot describe different things.
        rescued=sum(1 for best in best_iou if best < fg_iou_thresh),
        by_level=dict(sorted(by_level.items())),
        realised_fraction=capped_mean / RPN_BATCH_SIZE_PER_IMAGE,
        fg_iou_thresh=fg_iou_thresh,
        best_iou=best_iou,
    )


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


def sweep(
    sizes: tuple[tuple[int, ...], ...],
    refs: list,
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
) -> list[CensusResult]:
    """The census at each candidate foreground threshold, and the table issue #24 sets its rung
    from.

    Returned as well as printed, for the reason `census` is: the only check that ever ran against
    the realised-fraction line before was a human reading a terminal, and that is how a defect in
    it shipped.
    """
    anchors, per_level = anchors_for(sizes)
    results = _measure_at(anchors, per_level, refs, thresholds)
    _sweep_report(results)
    return results


def percentiles(values: list[float], at: tuple[int, ...] = PERCENTILES) -> dict[int, float]:
    """The distribution, read at a few points, by nearest rank on the sorted values.

    Nearest rank rather than an interpolating quantile because every number this reports is a
    value some ship actually has against some anchor, and a threshold set from an interpolation
    between two ships would be a threshold no ship sits at.

    `ceil` rather than `round`, which is what nearest rank is defined as and is not a nicety here:
    Python rounds halves to even, so the median of an even number of ships — 3637 ships is odd,
    but no census is guaranteed to be — would land one rank low, and the number this prints is
    read as "the threshold at which half the ships have a genuine match".
    """
    if not values:
        return {}

    ordered = sorted(values)
    return {
        point: ordered[min(len(ordered) - 1, max(0, ceil(point / 100 * len(ordered)) - 1))]
        for point in at
    }


def _sweep_report(results: list[CensusResult]) -> None:
    """The sweep as the table it goes into `docs/decisions.md` as."""
    best_iou = results[0].best_iou
    read_at = percentiles(best_iou)
    print(f"  best IoU any anchor offers a ship, over {len(best_iou)} ships:")
    print("    " + ", ".join(f"p{point} {value:.3f}" for point, value in read_at.items()))

    print("\n| fg IoU | positives per tile | rescue-only | share | realised fraction |")
    print("| --- | --- | --- | --- | --- |")
    for result in results:
        total = sum(result.positives_per_tile)
        tiles = max(result.ship_bearing_tiles, 1)
        ships = max(len(result.best_iou), 1)
        print(
            f"| {result.fg_iou_thresh:.2f} | mean {total / tiles:.1f}, "
            f"max {max(result.positives_per_tile, default=0)} "
            f"| {result.rescued} / {ships} | {result.rescued / ships:.3f} "
            f"| {result.realised_fraction:.3f} |"
        )

    # Said here rather than in a comment, because the reader of this table is choosing a value to
    # put in a rung config and `Matcher` is what will refuse them at 0.25 with the background
    # threshold left alone.
    print(
        f"\nEvery row above holds the background threshold at torchvision's {LOW_IOU_THRESHOLD}, "
        "which changes none of these counts. A run at a foreground threshold below it must lower "
        "it too: Matcher refuses a background threshold above the foreground one."
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

    # Issue #24's sixth rung, and only against the anchors the project actually ships: the small
    # set was rejected on 2026-08-23 and is not what any future rung stands on, so sweeping a
    # threshold over it would produce a table describing a configuration nothing will run.
    print(f"\nforeground threshold sweep, stock anchors: {ANCHOR_SIZES}")
    sweep(ANCHOR_SIZES, ship_bearing)


if __name__ == "__main__":
    main()
