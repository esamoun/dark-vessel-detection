"""The IoU at which an anchor becomes a ship, and the three places that number has to arrive.

The census of 2026-08-19 found that ninety percent of the 3637 training ships never reach
torchvision's 0.7 foreground threshold against any anchor, in either set the ladder tried; they
are positive only because `allow_low_quality_matches` guarantees every box its best one. Two RPN
rungs then failed inside the region that describes — R2 moved the anchors, R4 moved the sampler
batch — and neither moved the threshold the region is defined by. Issue #24 is the rung that does,
and this file is what holds its parameter in place.

Three properties, and each of them is a silent failure rather than a loud one if it goes:

* the number reaching torchvision's matcher rather than being accepted and dropped, which would
  train a rung at 0.7 and report it under another number;
* the pair being refused when the background threshold sits above the foreground one, which
  torchvision refuses too, as a `torch._assert` naming neither key nor the config they came from;
* a checkpoint refusing a run that names a regime other than the one it was fitted under. That
  one is provenance and not behaviour — the threshold is inert at inference — which is exactly
  why nothing downstream would ever contradict it.

Skipped where torch is not installed, which includes CI, for the reason `test_model_stem.py`
gives: the chain installs and runs without a framework.
"""

import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from darkvessel.detect.model import detector_model  # noqa: E402

TILE_PX = 64

# Torchvision's own, and the value every checkpoint written before 2026-08-29 was trained under.
STOCK_FG = 0.7
STOCK_BG = 0.3


def a_model(**kwargs):
    """Untrained and small, because what is under test is where a number lands, not what the
    weights do with it."""
    return detector_model(tile_px=TILE_PX, seed=1, pretrained=False, **kwargs)


def test_the_thresholds_reach_the_matcher_that_labels_the_anchors() -> None:
    """`detector_model` takes them as arguments; `RegionProposalNetwork` decides what an anchor is
    with them. Between the two is one keyword pair in the `FasterRCNN` call, and dropping it is
    not an error — torchvision falls back to 0.7 and 0.3 and trains perfectly well. The rung would
    then run the baseline's thresholds under the sixth rung's name, and the ladder would post a
    gain of about zero for a change that never happened.

    Read off `rpn.proposal_matcher`, which is the object that actually stratifies the anchors,
    rather than off any attribute this module set itself.
    """
    matcher = a_model(rpn_fg_iou_thresh=0.25, rpn_bg_iou_thresh=0.1).rpn.proposal_matcher

    assert matcher.high_threshold == 0.25
    assert matcher.low_threshold == 0.1
    # The guarantee the whole census turns on, and the thing a lowered threshold is measured
    # against. Torchvision's own default here is False; the detection builder sets it True, and a
    # rung that lost it would be measuring a different mechanism under the same name.
    assert matcher.allow_low_quality_matches


def test_the_stock_build_still_matches_the_anchors_the_way_every_run_so_far_did() -> None:
    """The five rungs already published were fitted at 0.7 and 0.3. A default drifting here would
    silently redefine what R0 to R4 were, and their journals hold no copy of it to disagree with.
    """
    matcher = a_model().rpn.proposal_matcher

    assert (matcher.high_threshold, matcher.low_threshold) == (STOCK_FG, STOCK_BG)


def test_a_background_threshold_above_the_foreground_one_is_refused_by_name() -> None:
    """`Matcher` refuses this itself, as a `torch._assert` reading `low_threshold should be <=
    high_threshold` — which names neither of this project's config keys nor the file they were
    read from. The refusal is made here instead, because the config that trips it is a rung config
    edited for a GPU session, and the machine that would otherwise report it is rented by the hour.

    It is also the constraint that makes the foreground threshold unusable on its own below 0.3:
    the census puts the median ship at `256/1024 = 0.25` against its containing anchor, so
    reaching it means moving both keys.
    """
    with pytest.raises(ValueError, match="rpn_bg_iou_thresh"):
        a_model(rpn_fg_iou_thresh=0.25)


def test_the_two_may_meet_at_one_value() -> None:
    """`bg == fg` is legal and means something definite — no band of ignored anchors, every one of
    them either a positive or a negative. It is a configuration a rung may want, so the guard
    above refuses only the strict inversion.
    """
    matcher = a_model(rpn_fg_iou_thresh=0.25, rpn_bg_iou_thresh=0.25).rpn.proposal_matcher

    assert (matcher.high_threshold, matcher.low_threshold) == (0.25, 0.25)


def test_a_checkpoint_is_refused_by_a_run_naming_a_threshold_it_was_not_fitted_under() -> None:
    """Unlike the anchors and the stem, this refusal is not about what the loaded model does: the
    RPN consults its matcher only while training, so these weights would behave identically either
    way. It is about which weights they are. A run config naming 0.7 while loading the sixth rung's
    checkpoint would produce a precision and a recall that are real, reported under a training
    regime that did not produce them, and no number in the chain would ever disagree.
    """
    from darkvessel.detect.trained import _check_built

    built = {
        "tile_px": TILE_PX,
        "anchor_sizes": ((32,),),
        "stem": "repeat",
        "rpn_fg_iou_thresh": 0.25,
        "rpn_bg_iou_thresh": 0.25,
    }

    with pytest.raises(ValueError, match="rpn_fg_iou_thresh"):
        _check_built(built, tile_px=TILE_PX, anchor_sizes=((32,),), stem="repeat")

    with pytest.raises(ValueError, match="rpn_bg_iou_thresh"):
        _check_built(
            built,
            tile_px=TILE_PX,
            anchor_sizes=((32,),),
            stem="repeat",
            rpn_fg_iou_thresh=0.25,
        )

    # And accepted when the run names what the checkpoint records.
    _check_built(
        built,
        tile_px=TILE_PX,
        anchor_sizes=((32,),),
        stem="repeat",
        rpn_fg_iou_thresh=0.25,
        rpn_bg_iou_thresh=0.25,
    )


def test_a_checkpoint_written_before_the_thresholds_existed_is_read_as_torchvisions_own() -> None:
    """`models/epoch-012.pt` and every rung of the ladder were written before this key existed,
    and all of them were trained at 0.7 and 0.3. Silence means those, the way silence about a stem
    means the repeat — read through `.get` so that a key deleted from the build block fails as
    this assertion rather than as a `KeyError`.
    """
    from darkvessel.detect.trained import _check_built

    _check_built(
        {"tile_px": TILE_PX, "anchor_sizes": ((32,),), "stem": "repeat"},
        tile_px=TILE_PX,
        anchor_sizes=((32,),),
        stem="repeat",
    )


def test_a_run_config_that_says_nothing_about_them_gets_the_stock_pair() -> None:
    """The chain's own run configs name a checkpoint trained at 0.7, and none of them states it.
    `trained_request_from` has to supply it, or every one of them would be refused by the check
    above the moment the key reached the build block.
    """
    from pathlib import Path

    from darkvessel.cli import trained_request_from

    request = trained_request_from(
        {
            "trained": {
                "checkpoint": "models/epoch-012.pt",
                "tile_px": 800,
                "anchor_sizes": [[32], [64], [128], [256], [512]],
                "score_threshold": 0.75,
                "stretch": {"floor_db": -30.0, "ceiling_db": 5.0, "sea_db": -22.0},
            }
        },
        Path("."),
    )

    assert request["rpn_fg_iou_thresh"] == STOCK_FG
    assert request["rpn_bg_iou_thresh"] == STOCK_BG
