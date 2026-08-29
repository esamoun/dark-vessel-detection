"""The arithmetic `notebooks/anchor_census.py` sets two rungs of the ladder from.

That script shipped once with nothing checking it, on the ruling that it was a measurement for a
decision log rather than behaviour anything depended on. A review then found an arithmetic defect
in the one line the whole census exists to produce — the realised positive fraction capped the
*mean across tiles* rather than each tile's own count, biasing the number toward the 50% ceiling
it was written to show the sampler was nowhere near. Nothing caught that, because nothing was
looking. This file is what looks now: each property pinned against real torchvision
matching on hand-built boxes, no dataset required.

The threshold sweep of 2026-08-29 is under the same rule and for a sharper reason: issue #24's
sixth rung is *set* from the number it prints, before a GPU evening is spent on it, exactly the
way rung 4's `rpn_batch_size_per_image: 32` was set from the census of 2026-08-19. A sweep that
counted wrong would not fail, it would name a threshold.

`notebooks/` is not an importable package — it has no `__init__.py` and is not on the install
path — so the module is loaded by file location rather than `import`ed by name. This is the same
constraint `sweep_window.py` lives under; unlike that script, this one now has something to
verify it against.

Skipped where torch is not installed, which includes CI: the chain's acceptance condition is that
it installs and runs without a framework, and this script — like `test_model_stem.py` and
`test_training_run.py` before it — is on the far side of that line.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from darkvessel.detect.dataset import Box, TileRef  # noqa: E402

_MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "anchor_census.py"
_SPEC = importlib.util.spec_from_file_location("anchor_census", _MODULE_PATH)
anchor_census = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(anchor_census)


def a_tile(name: str, *boxes: Box) -> TileRef:
    """A `TileRef` with no image on disk, because the census never reads one — only `.boxes`."""
    return TileRef(name=name, scene=1, image_path=Path("/dev/null"), boxes=tuple(boxes))


def test_realised_fraction_caps_per_tile_not_the_mean():
    """The RPN sampler runs once per image and caps *that image's* positives before it ever sees
    another tile — capping the mean across tiles instead is a different, larger number, and it is
    the number this census originally reported.

    Four tiles carry one small box each (fewer than `RPN_POSITIVE_CAP` positives apiece under the
    stock anchors); a fifth is packed with sixteen, densely enough that its own positive count
    clears the cap on its own. Capping-then-averaging and averaging-then-capping disagree exactly
    when some tile is individually above the cap and others are not — which is this fixture. Both
    numbers are pinned so a reader can see what the bug was, not just that it is gone: reverting
    the fix reproduces 0.500, the number this census used to print.
    """
    sparse_tiles = [
        a_tile(
            f"sparse_{i}",
            Box(min_row=400, min_col=400, max_row=400 + 4 + i, max_col=400 + 4 + i),
        )
        for i in range(4)
    ]
    dense_boxes = [
        Box(min_row=r, min_col=c, max_row=r + 6, max_col=c + 6)
        for r in range(100, 100 + 4 * 20, 20)
        for c in range(100, 100 + 4 * 20, 20)
    ]
    dense_tile = a_tile("dense", *dense_boxes)

    result = anchor_census.census(anchor_census.ANCHOR_SIZES, [*sparse_tiles, dense_tile])

    assert result.positives_per_tile == [80, 80, 80, 80, 800]

    # Correct: cap each tile, then average. (80+80+80+80+128) / 5 / 256.
    assert result.realised_fraction == pytest.approx(0.35)

    # What the pre-fix formula gave: cap the mean, not each tile. min(mean([...]), 128) / 256.
    total = sum(result.positives_per_tile)
    old_formula = (
        min(total / len(result.positives_per_tile), anchor_census.RPN_POSITIVE_CAP)
        / anchor_census.RPN_BATCH_SIZE_PER_IMAGE
    )
    assert old_formula == pytest.approx(0.5)
    assert result.realised_fraction != pytest.approx(old_formula)


def test_real_match_and_rescue_only_are_different_counts():
    """`allow_low_quality_matches` guarantees every box an anchor however poor the overlap, so
    "how many positives exist" and "how many boxes only exist because of that guarantee" are two
    different questions. Conflating them would make the census claim ships are findable that are
    in fact only nominally matched — the opposite of what it exists to report.

    One box is an exact copy of a real generated anchor (IoU 1.0, unambiguously a genuine match).
    The other is a single pixel, far too small to reach `HIGH_IOU_THRESHOLD` against anything the
    small (rung 2) anchor set offers. Both end up positive — the guarantee sees to that — but only
    one should ever be counted as rescued.
    """
    small = anchor_census.CANDIDATES["small (rung 2)"]
    anchors, _ = anchor_census.anchors_for(small)

    # A real anchor, copied exactly: IoU with itself is 1.0, so this is a genuine match under any
    # threshold this project would plausibly use.
    x1, y1, x2, y2 = anchors[anchors.shape[0] // 2].tolist()
    real_match = Box(min_row=y1, min_col=x1, max_row=y2, max_col=x2)

    # One pixel, positioned off the anchor grid: nowhere near 0.7 IoU with even the smallest
    # (4 px) anchor on offer.
    rescue_only = Box(min_row=777.5, min_col=777.5, max_row=778.5, max_col=778.5)

    result = anchor_census.census(
        small, [a_tile("real_match", real_match), a_tile("rescue_only", rescue_only)]
    )

    assert result.positives_per_tile == [1, 1]  # both boxes are positive
    assert result.rescued == 1  # but only one of them only because of the rescue rule


def test_level_of_boundaries():
    """`_level_of` is the arithmetic that turns a flat anchor index into a pyramid level, and the
    boundary is the one place it can be silently off by one: `cumsum` gives the index one past
    each level's last anchor, so an index *equal* to a boundary belongs to the level after it, not
    the one the boundary closes. `<=` is what that requires; `<` looks almost the same and is
    wrong at every boundary.

    Levels of size 3, 2 and 4 give boundaries at 3, 5 and 9. Every index either side of a boundary
    is checked, not just one per level, because an off-by-one only shows up at the seam.
    """
    boundaries = torch.tensor([3, 5, 9])

    assert [anchor_census._level_of(i, boundaries) for i in range(9)] == [
        0, 0, 0,  # level 0: indices 0..2
        1, 1,  # level 1: indices 3..4
        2, 2, 2, 2,  # level 2: indices 5..8
    ]  # fmt: skip


def test_high_iou_threshold_is_shared():
    """`Matcher`'s own high threshold and the separate `quality.max(...) < HIGH_IOU_THRESHOLD`
    line that counts rescues have to agree on where the line is, or the rescued count stops
    describing what the matcher actually rescued.

    An earlier version of this test built a single anchor and two boxes. With
    `allow_low_quality_matches=True`, a single anchor is *always* the best (and only) match
    available to whichever box it is compared against, so it is forced positive by the rescue
    guarantee regardless of what threshold the matcher was built with — `positives_per_tile` was
    `[1, 1]` whether the matcher's high threshold was 0.7 or a drifted 0.5. That version passed
    against a matcher hardcoded to 0.5, which is exactly the defect it was meant to catch. It was
    a test of the rescue-count arithmetic wearing the name of a drift test.

    This version builds one box and *two* anchors, sized from `HIGH_IOU_THRESHOLD` and
    `LOW_IOU_THRESHOLD` rather than from literals, so it still holds if either is retuned:
    `above` overlaps the box at an IoU past `HIGH_IOU_THRESHOLD`, `between` at an IoU inside the
    band the two thresholds bound. Under a matcher built at `HIGH_IOU_THRESHOLD`, only `above` is
    a real match — `between` falls in `BETWEEN_THRESHOLDS` and is not the box's best anchor, so
    the low-quality rescue does not reach it either. Under a matcher whose high threshold has
    drifted down past `between`'s IoU (0.5, this file's diagnosed defect, is such a value), both
    anchors clear the bar and both are positive. So `positives_per_tile` is `[1]` at the real
    threshold and `[2]` under that drift: this is the assertion that actually depends on what the
    matcher was constructed with, and the one that catches the defect this file was fixed for.

    The rescued count is kept too, but it pins a different, matcher-independent fact: with the
    box's best IoU (`above`, 0.75) already past `HIGH_IOU_THRESHOLD`, the box has a genuine match
    and nothing about it needed the low-quality rule. `rescued` is computed straight from
    `quality.max(dim=1)` and `HIGH_IOU_THRESHOLD` — it never sees the matcher object — so this
    assertion would hold even under the drifted threshold and does not, by itself, catch it.
    """
    margin = 0.05
    high = anchor_census.HIGH_IOU_THRESHOLD
    low = anchor_census.LOW_IOU_THRESHOLD
    assert low < high - margin, "the band is too narrow for this fixture's margin"

    box = Box(min_row=0, min_col=0, max_row=10, max_col=10)
    # Anchors share the box's top-left corner and its width, so IoU is simply the anchor's height
    # as a fraction of the box's (the anchor is fully contained in the box): height 10 * iou.
    above = [0.0, 0.0, 10.0, 10 * (high + margin)]
    between = [0.0, 0.0, 10.0, 10 * (high - margin)]
    anchors = torch.tensor([above, between])

    result = anchor_census._measure(anchors, [2], [a_tile("one_box", box)])

    # Depends on the matcher's high threshold: [1] at HIGH_IOU_THRESHOLD, [2] if it has drifted
    # down past `between`'s IoU.
    assert result.positives_per_tile == [1]
    # Matcher-independent: the box's best anchor already clears HIGH_IOU_THRESHOLD on its own, so
    # nothing here was rescued.
    assert result.rescued == 0


# One box against two anchors, chosen so every count below can be read off two IoUs by hand.
# The first anchor is twice the box's area and shares its corner, so the overlap is `100/200`; the
# second is four times it, so `100/400`. Both are far too small a fixture to need a dataset and
# far too explicit to be checked by a number this file also computes.
_ANCHORS = [[0.0, 0.0, 10.0, 20.0], [0.0, 0.0, 10.0, 40.0]]


def _two_tiles() -> list:
    """A tile whose ship has a genuine match down to 0.5, and one whose ship never beats 0.25.

    Between them they hold the whole question the sweep exists to answer: at 0.7 both are rescued,
    at 0.4 one of them is not, and at 0.2 neither is.
    """
    return [
        a_tile("best_half", Box(min_row=0, min_col=0, max_row=10, max_col=10)),
        a_tile("best_quarter", Box(min_row=0, min_col=0, max_row=5, max_col=10)),
    ]


def test_lowering_the_threshold_turns_rescued_ships_into_matched_ones():
    """The sixth rung's whole mechanism, on two ships whose overlaps are known exactly.

    `allow_low_quality_matches` guarantees every box its best anchor whatever the threshold, so
    "how many ships are positive" does not move at all here — it is 1 and 1 at every row. What
    moves is *why*: at 0.7 both ships are positive only because of the guarantee, and by 0.2
    neither is. That distinction is the one the census of 2026-08-19 said the ladder had never
    tested, and it is the only thing the rung changes.
    """
    anchors = torch.tensor(_ANCHORS)

    rescued_at = {
        result.fg_iou_thresh: result.rescued
        for result in anchor_census._measure_at(anchors, [2], _two_tiles(), (0.7, 0.4, 0.2))
    }

    assert rescued_at == {0.7: 2, 0.4: 1, 0.2: 0}


def test_a_lower_threshold_admits_more_anchors_per_ship_not_more_ships():
    """The other half of the same measurement, and the one that reaches the sampler.

    A ship rescued at 0.7 contributes only the anchors tied at its own maximum — one, here. Drop
    the threshold under an anchor that was previously merely close and that anchor becomes a
    positive example in its own right, so the RPN's batch has two to draw on instead of one. That
    is the quantity `realised_fraction` reports, and it is why the rung is not simply relabelling
    the same anchors.
    """
    anchors = torch.tensor(_ANCHORS)

    positives_at = {
        result.fg_iou_thresh: result.positives_per_tile
        for result in anchor_census._measure_at(anchors, [2], _two_tiles(), (0.7, 0.4, 0.2))
    }

    assert positives_at == {0.7: [1, 1], 0.4: [1, 1], 0.2: [2, 1]}


def test_the_rescue_count_is_the_distribution_read_at_the_threshold():
    """`rescued` and `best_iou` are two statements about the same thing, and the sweep's value
    depends on their agreeing: the table names a threshold, and the reader picks a different one
    off the percentiles beside it. Counted from the one list rather than measured twice, so they
    cannot drift; asserted here so that a future rewrite measuring them separately fails.
    """
    anchors = torch.tensor(_ANCHORS)

    results = anchor_census._measure_at(anchors, [2], _two_tiles(), (0.7, 0.4, 0.2))

    assert results[0].best_iou == [0.5, 0.25]
    for result in results:
        assert result.rescued == sum(1 for best in result.best_iou if best < result.fg_iou_thresh)
        # The distribution is a property of the anchors and the ships, not of the threshold, so
        # every row of one sweep carries the same one.
        assert result.best_iou == results[0].best_iou


def test_a_ship_sitting_exactly_on_the_threshold_is_counted_the_way_the_matcher_counts_it():
    """The rescue count decides in float32, because that is the space `Matcher` decides in.

    `box_iou` returns float32, and torch demotes the Python threshold it is compared against to
    float32 too — so the line the matcher actually draws at "0.7" is `float32(0.7)`, which is
    0.69999998807907104. Reading the same overlaps out as Python floats and comparing them against
    the literal 0.7 draws the line at 0.69999999999999996 instead, and a ship whose best overlap
    lands exactly on the boundary falls on the other side of it.

    This is not hypothetical and it is not a rounding nicety. The sweep of 2026-08-30 reported
    3525 rescued boxes under rung 2's anchors where the census of 2026-08-19 published 3524 — one
    ship, off by exactly this, in a number two entries of `docs/decisions.md` quote. A count that
    disagrees with the matcher by one ship is a count that describes something near what the
    matcher did rather than what it did.

    Built as one box against one anchor overlapping it at exactly `float32(0.7)`: the box is the
    anchor's own top slice, so the IoU is the height ratio, and the anchor's height is chosen so
    that ratio is the float32 neighbour of 0.7 rather than 0.7 itself.
    """
    on_the_line = torch.tensor([0.7], dtype=torch.float32).item()
    # Anchor and box share a corner and a width, so IoU is the box's height over the anchor's.
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    box = Box(min_row=0, min_col=0, max_row=10 * on_the_line, max_col=10)

    result = anchor_census._measure(anchors, [1], [a_tile("on_the_line", box)])

    assert result.best_iou == [pytest.approx(on_the_line)]
    # Positive by the matcher's own reckoning, since float32(0.7) is not below float32(0.7) --
    # so nothing here was rescued, and the count has to say the same.
    assert result.positives_per_tile == [1]
    assert result.rescued == 0


def test_one_pass_over_the_boxes_says_what_one_pass_per_threshold_says():
    """`_measure_at` computes `box_iou` once and applies every matcher to the matrix, because the
    matrix is two hundred thousand anchors wide and does not depend on the threshold. That is an
    optimisation, and an optimisation on the line the census exists to produce is exactly the kind
    of thing that shipped a wrong realised fraction the first time.
    """
    anchors = torch.tensor(_ANCHORS)
    thresholds = (0.7, 0.4, 0.2)

    swept = anchor_census._measure_at(anchors, [2], _two_tiles(), thresholds)
    one_at_a_time = [
        anchor_census._measure(anchors, [2], _two_tiles(), fg_iou_thresh=threshold)
        for threshold in thresholds
    ]

    assert swept == one_at_a_time


def test_the_sweep_reaches_thresholds_below_torchvisions_background_default():
    """`Matcher` refuses a background threshold above the foreground one, and the sweep's whole
    point is the region below 0.3 — where the census puts the median ship, at `256/1024`. A sweep
    that raised there would report nothing at all about the only rows anyone is reading it for.
    """
    anchors = torch.tensor(_ANCHORS)

    results = anchor_census._measure_at(anchors, [2], _two_tiles(), anchor_census.SWEEP_THRESHOLDS)

    assert [result.fg_iou_thresh for result in results] == list(anchor_census.SWEEP_THRESHOLDS)
    assert min(result.fg_iou_thresh for result in results) < anchor_census.LOW_IOU_THRESHOLD


def test_a_percentile_is_a_value_some_ship_actually_has():
    """Nearest rank, not an interpolating quantile. The number this prints is read as "a threshold
    at which this many ships have a genuine match", so a value interpolated between two ships is a
    threshold no ship sits at and the count beside it would be off by however many ships share the
    boundary.
    """
    values = [0.1, 0.2, 0.3, 0.4]

    read_at = anchor_census.percentiles(values, at=(25, 50, 75, 100))

    assert read_at == {25: 0.1, 50: 0.2, 75: 0.3, 100: 0.4}
    assert anchor_census.percentiles([]) == {}

    # Ten values put the 25th percentile at rank 2.5, which is where `round` and `ceil` part
    # company — Python rounds halves to even, so `round` gives rank 2 and the printed number is
    # one ship low. Nearest rank is the ceiling, and this fixture is the only place the two
    # differ, so it is the only place a revert would show.
    ten = [round(0.1 * n, 1) for n in range(1, 11)]
    assert anchor_census.percentiles(ten, at=(25,)) == {25: 0.3}
