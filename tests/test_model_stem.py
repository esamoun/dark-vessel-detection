"""The input stage, and the one property that makes rung 3 of the ladder a measurement.

LS-SSDD is VV and the scene this chain exports is VV, so the dual-polarisation stem issue #11
asks for has no data on either side — see docs/failures.md. What is delivered instead is an
honest single-channel stem, and what is asserted here is that it starts life as exactly the model
the three-channel repeat starts life as. Without that, the rung would be comparing two different
initialisations and reporting the difference as an adaptation.

Skipped where torch is not installed, which includes CI.
"""

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from darkvessel.detect.model import as_model_input, detector_model  # noqa: E402

TILE_PX = 64


def a_model(stem: str):
    """Untrained, because a test that fetched 160 MB of COCO weights is not a test anyone runs.
    The fold is exact whatever the weights are."""
    return detector_model(
        tile_px=TILE_PX, seed=1, pretrained=False, trainable_backbone_layers=5, stem=stem
    ).eval()


# `conv1` pads with three rings of zeros, so its output is contaminated within three positions of
# the edge. Three at the output's stride is a safe over-estimate of the two it strictly needs.
MARGIN_PX = 3


def stem_output(stem: str, image: np.ndarray):
    """What `conv1` produces, through the model's own transform.

    Through the transform rather than around it because the normalisation is half of what the fold
    absorbs, and a test that skipped it would pass with the bias missing.
    """
    model = a_model(stem)
    with torch.no_grad():
        images, _ = model.transform([as_model_input(image, stem)], None)
        return model.backbone.body.conv1(images.tensors)


def a_parameter_map(model) -> dict:
    """Every parameter and buffer of a model, by name."""
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


def test_only_the_stem_differs_between_the_two_models_at_initialisation() -> None:
    """The property rung 3 actually rests on: the two models are not two draws.

    `detector_model` seeds the global generator and then builds, and constructing a `Conv2d` draws
    from it — so folding the stem before the fresh two-class head is built would give the two
    models different heads, and the rung would be measuring an initialisation rather than a stem.
    Everything outside `conv1` is required to be identical, which is what pins that ordering.
    """
    repeat, single = a_parameter_map(a_model("repeat")), a_parameter_map(a_model("single"))

    outside = [name for name in repeat if not name.startswith("backbone.body.conv1")]
    # The comparison would be worth nothing if the filter had matched everything.
    assert outside

    for name in outside:
        assert torch.equal(repeat[name], single[name]), name


def test_the_folded_stem_computes_the_three_channel_stem_away_from_the_tile_edge() -> None:
    """What the fold delivers, and the boundary that stops it being everything.

    `conv1` pads with three rings of zeros. Under the repeat stem the tile is normalised before the
    convolution, so those zeros stand for raw amplitude `m_c` — a different value in each channel.
    Under the folded stem the transform is the identity, so they stand for a raw zero. No single
    padding value reconciles the two: it would have to satisfy `v · A_k = B_k` for every output
    channel at once, and those ratios differ per channel.

    So what the fold buys is the interior, and the interior is what rung 3 needs: the kernels carry
    the same values and the same normalisation, and only the convention for what lies outside the
    tile differs. It is asserted on `conv1`'s own output rather than on a feature map because the
    FPN's top-down path carries C5 — whose receptive field is the whole tile — into every level,
    so no interior of a feature map agrees.
    """
    image = np.random.default_rng(0).random((TILE_PX, TILE_PX)).astype(np.float32)
    interior = (..., slice(MARGIN_PX, -MARGIN_PX), slice(MARGIN_PX, -MARGIN_PX))

    assert torch.allclose(
        stem_output("repeat", image)[interior], stem_output("single", image)[interior], atol=1e-5
    )


def test_the_single_channel_stem_takes_one_channel() -> None:
    model = a_model("single")

    assert model.backbone.body.conv1.in_channels == 1
    assert model.transform.image_mean == [0.0]
    assert model.transform.image_std == [1.0]


def test_the_bias_the_fold_needs_is_there() -> None:
    """`bn1` is frozen at these settings, so it applies fixed statistics instead of recentring the
    batch. Drop the bias and the two paths differ by a constant through the whole backbone."""
    assert a_model("single").backbone.body.conv1.bias is not None


def test_one_tile_reaches_the_model_with_the_channels_its_stem_expects() -> None:
    image = np.zeros((TILE_PX, TILE_PX), dtype=np.float32)

    assert as_model_input(image, stem="repeat").shape == (3, TILE_PX, TILE_PX)
    assert as_model_input(image, stem="single").shape == (1, TILE_PX, TILE_PX)


def test_a_stem_this_project_does_not_have_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="dual"):
        detector_model(tile_px=TILE_PX, seed=1, pretrained=False, stem="dual")


def test_a_checkpoint_is_refused_by_a_run_asking_for_the_other_stem() -> None:
    """Anchors leave no trace in a state dict and neither does a stem — except that a folded
    `conv1` has a different shape, so this one would at least fail loudly. It is checked anyway,
    beside the anchors, so the refusal reads as one rule rather than two accidents."""
    from darkvessel.detect.trained import _check_built

    built = {"tile_px": TILE_PX, "anchor_sizes": ((32,),), "stem": "single"}

    with pytest.raises(ValueError, match="stem"):
        _check_built(built, tile_px=TILE_PX, anchor_sizes=((32,),), stem="repeat")


def test_a_checkpoint_written_before_stems_existed_is_read_as_the_repeat_it_was() -> None:
    """`models/epoch-012.pt` and everything before 2026-08-17 has no stem in its build block, and
    every one of them was trained on three repeated channels."""
    from darkvessel.detect.trained import _check_built

    _check_built(
        {"tile_px": TILE_PX, "anchor_sizes": ((32,),)},
        tile_px=TILE_PX,
        anchor_sizes=((32,),),
        stem="repeat",
    )
