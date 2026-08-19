"""Detector architecture.

Two adaptations drive the design: vessels are a few pixels wide at 10 m resolution, so the
high-resolution levels of the feature pyramid carry the signal; and pretrained backbones expect
three colour channels where the input is single- or dual-polarisation radar amplitude.

Neither is made by default. This is a stock Faster R-CNN with a two-class head, and each
adaptation arrives as an argument that is off unless a config asks for it — because an adaptation
belongs to the run that measures it, against the configuration before it, on the same held-out
split. Turning one on here would delete the baseline that comparison needs.

So the anchors below are torchvision's own, and they are almost certainly wrong for this data:
the smallest is 32 px, which at 10 m is a vessel 320 m long, longer than all but the largest
container ships and therefore larger than nearly every ship in the training set. That is a
prediction about the first run's recall rather than a defect to fix here, and `anchor_sizes` is
a config key so that changing it is one line and a second run.

What cannot be deferred is the channel count, because without an answer to it nothing runs at
all: ImageNet backbones take three channels and Sentinel-1 VV is one. `STEMS` below names the two
answers this module has. The default repeats the amplitude across all three, which is the null
adaptation — it keeps the pretrained first-layer filters meaningful, since a grey image is a
thing they have seen. The other takes the one channel it is given, with the pretrained stem folded
down onto it: every weight outside that first convolution is the repeat's own, and inside the tile
the folded stem computes what the repeat computes. The two part company only over what lies
outside the tile — see `_fold_stem` — so a run that asks for the single stem measures what
training does with one bank of kernels rather than a different initialisation. Neither is the
dual-polarisation stem the ticket asked for, and that one is not coming: there is no
dual-polarisation data to fit it on and none to run it over.

This module imports torch, along with `train.py` and `trained.py`. Everything that can be got
wrong quietly — the split, the subset, the augmentations, the counting, the resume — is on the
other side of that line, in modules a laptop can test in a second.
"""

import numpy as np
import torch
from torch import Tensor
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from darkvessel.detect.dataset import Box
from darkvessel.detect.detector import PixelDetection

# torchvision reserves 0 for the background, so there are two classes: not-a-ship, and a ship.
SHIP = 1
CLASSES = 2

# One tuple per level of the feature pyramid, in pixels — torchvision's own, kept as the baseline
# the small-target work is measured against. At 10 m they are hulls of 320 m and upwards, which
# is the wrong range for this data and is meant to be.
ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))

# A ship is longer than it is wide and it can lie in any direction, so the ratios are the stock
# ones. Nothing about radar argues for changing them.
ASPECT_RATIOS = (0.5, 1.0, 2.0)

# How many channels each input stage takes. "repeat" is the baseline the ladder starts from: one
# polarisation copied three times, which is the minimum a three-channel ImageNet backbone accepts
# and is not an adaptation to anything. "single" is the adaptation — one channel of radar
# amplitude, with the pretrained stem folded down onto it.
#
# There is no "dual". LS-SSDD is VV and the scene this chain exports is VV, so a dual-polarisation
# stem has no data to be fitted on and none to be run on. See docs/failures.md.
STEMS = {"repeat": 3, "single": 1}

# What the pretrained backbone was normalised with. Kept as ImageNet's rather than recomputed
# over LS-SSDD, because the weights being adapted were fitted under these and a detector trained
# for a handful of epochs has not got the budget to move the backbone far from them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def detector_model(
    *,
    tile_px: int,
    seed: int,
    anchor_sizes: tuple[tuple[int, ...], ...] = ANCHOR_SIZES,
    stem: str = "repeat",
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    rpn_batch_size_per_image: int = 256,
    rpn_positive_fraction: float = 0.5,
    box_batch_size_per_image: int = 512,
    box_positive_fraction: float = 0.25,
) -> FasterRCNN:
    """A Faster R-CNN sized for ships of a few pixels on Sentinel-1.

    Args:
        tile_px: The side of the tiles this model is trained and run on. Fixed rather than
            inferred, so that the transform inside the model resamples nothing: rescaling radar
            amplitude changes what the detector sees, and that is a decision about a run rather
            than a convenience — the same argument `cli.py` makes about reprojecting a scene.
        seed: The run's seed. Names the weights as well as the data, which it did not until a
            run of the same config twice produced two different models.
        anchor_sizes: One tuple per pyramid level. Configurable because this is the number most
            likely to want moving once there are real numbers to move it against.
        stem: `"repeat"` or `"single"`. The single-channel stem is built from the repeat's own
            weights and reproduces what the repeat produces inside the tile, differing only at the
            padded edge, so the rung that introduces it measures what training does with it rather
            than a different starting point.
        pretrained: Start from COCO weights. A free tier gives too few epochs to train a
            ResNet-50 from scratch; set it false only where the session has no network to fetch
            them with.
        trainable_backbone_layers: How much of the backbone is unfrozen, from the top. Three of
            five is torchvision's default and is what the budget here affords.
        rpn_batch_size_per_image: How many anchors the region proposal network computes its loss
            over. Torchvision's default is 256. Lowering it is the lever on this data: with a
            handful of ships to a tile there are nowhere near 128 positive anchors to be had, so
            the remaining slots fill with background whatever `rpn_positive_fraction` asks for.
        rpn_positive_fraction: A **ceiling** on how much of that batch may be positive, not a
            target — the sampler takes `min(available, requested)`. Stated here because the
            distinction is what rung 4 of the ladder turns on.
        box_batch_size_per_image: The same, for the second-stage head.
        box_positive_fraction: The same, for the second-stage head.
    """
    if stem not in STEMS:
        raise ValueError(f"unknown stem {stem!r}; this project has {sorted(STEMS)} and no dual")

    # Applied here, before anything is constructed, because the head below is initialised from
    # scratch — two classes where COCO had 91 — and it draws from torch's global generator. Left
    # unseeded, two sessions of the same config start from two different models and report
    # different numbers, for a reason nothing in the config records. Found by running the same
    # configuration twice: see docs/failures.md.
    torch.manual_seed(seed)

    channels = STEMS[stem]
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        # Freezing is a claim about weights that were fitted on something else. Without them
        # there is nothing to preserve, and torchvision says so rather than silently obeying.
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        rpn_anchor_generator=AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=(ASPECT_RATIOS,) * len(anchor_sizes),
        ),
        # The single-channel stem absorbs the normalisation into its own weights, so the transform
        # in front of it has nothing left to do.
        image_mean=IMAGENET_MEAN if channels == 3 else [0.0],
        image_std=IMAGENET_STD if channels == 3 else [1.0],
        min_size=tile_px,
        max_size=tile_px,
        rpn_batch_size_per_image=rpn_batch_size_per_image,
        rpn_positive_fraction=rpn_positive_fraction,
        box_batch_size_per_image=box_batch_size_per_image,
        box_positive_fraction=box_positive_fraction,
    )

    # The COCO head predicts 91 classes. Ships are one of them, and reusing that column would be
    # a defensible shortcut on optical imagery; on radar amplitude the features underneath it are
    # different enough that it is not worth the confusion of explaining. Fresh head, two classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, CLASSES
    )

    # After the head, and that ordering is load-bearing. Building a `Conv2d` draws from the global
    # generator, so folding the stem before the head above would give the two stems different
    # heads — and the rung that introduces the stem would be measuring an initialisation.
    if channels == 1:
        _fold_stem(model)

    return model


def _fold_stem(model: FasterRCNN) -> None:
    """Replace the three-channel `conv1` with the one-channel convolution that computes the same
    thing inside the tile.

    The repeat path is `y = Σ_c W_c · (x − m_c) / s_c`, where `m` and `s` are ImageNet's per-channel
    statistics and `x` is one polarisation of amplitude copied three times. Summing the kernels
    over the channel axis, each divided by its own standard deviation, gives a one-channel weight
    that reproduces the first term; the second is a constant per output channel, which is a bias.

    Inside the tile, and not at its edge: `conv1` pads with three rings of zeros, which under the
    repeat stem are zeros in normalised space and stand for raw amplitude `m_c`, a different value
    per channel, and under this one are a raw zero. Nothing reconciles those two conventions — a
    single padding value would have to satisfy `v · A_k = B_k` for every output channel at once,
    and the ratios differ per channel. So the two agree beyond three positions of the border and
    differ within it, which is a convention about what lies outside a tile rather than a difference
    in what the model has been given to start from.

    The bias would be redundant in a stock ResNet, where the batch norm behind `conv1` recentres
    whatever arrives. It is not redundant here: at the trainable-layer counts this project uses,
    `bn1` is a `FrozenBatchNorm2d` applying fixed statistics, so a constant offset propagates
    through the entire backbone instead of being absorbed.

    The trainability is carried across rather than left to default. At three trainable layers
    torchvision unfreezes `layer4`, `layer3` and `layer2` and nothing else, so the stem this
    replaces is frozen — while a fresh `Conv2d` arrives trainable, and `train.py` builds its
    optimiser from whatever has `requires_grad`. Left alone, the single-stem run would train 3,200
    parameters the repeat run never touches, which is the confound this stem exists to remove
    reappearing on the trainability axis. The bias follows the weight for the same reason: the
    baseline has no such parameter at all.
    """
    conv1 = model.backbone.body.conv1
    weight = conv1.weight.data
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    folded = torch.nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=True,
    )
    folded.weight.data = (weight / std).sum(dim=1, keepdim=True)
    folded.bias.data = -(weight * mean / std).sum(dim=(1, 2, 3))
    folded.weight.requires_grad_(conv1.weight.requires_grad)
    folded.bias.requires_grad_(conv1.weight.requires_grad)

    model.backbone.body.conv1 = folded


def as_model_input(image: np.ndarray, stem: str = "repeat") -> Tensor:
    """One tile of amplitude in 0..1, as the image this model's stem expects.

    Repeated rather than averaged or padded with zeros, under the repeat stem: a grey image is
    something the pretrained filters have seen, and two channels of zeros is not. Under the single
    stem there is nothing to repeat — the fold has already put the three kernels into one.
    """
    tile = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
    return tile.repeat(STEMS[stem], 1, 1)


def detections_from(output: dict[str, Tensor]) -> list[PixelDetection]:
    """A model's boxes, as the points the rest of the chain deals in.

    Through `Box.from_xyxy` and `Box.centre` rather than by unpacking the corners here, so that
    the axis swap and the half-pixel between an edge coordinate and a pixel index are each
    applied in the one place that owns them.

    Beside `as_model_input` because it is the other half of the same boundary: one turns a tile
    into what torchvision takes, the other turns what torchvision returns back into what this
    project's contract states. It sat in `train.py` while scoring was the only caller; inference
    is the second, and a second copy of that half-pixel is exactly the defect `Box` exists to
    prevent.
    """
    return [
        PixelDetection(row=row, col=col, score=float(score))
        for box, score in zip(
            output["boxes"].cpu().tolist(), output["scores"].cpu().tolist(), strict=True
        )
        for row, col in [Box.from_xyxy(box).centre()]
    ]
