"""Detector architecture.

Two adaptations drive the design: vessels are a few pixels wide at 10 m resolution, so the
high-resolution levels of the feature pyramid carry the signal; and pretrained backbones expect
three colour channels where the input is single- or dual-polarisation radar amplitude.

Neither is made here. This is a stock Faster R-CNN with a two-class head, and the adaptations
belong to the ticket that measures them — each against the configuration before it, on the same
held-out split. Shipping them now with no measurement behind them would delete the baseline that
comparison needs.

So the anchors below are torchvision's own, and they are almost certainly wrong for this data:
the smallest is 32 px, which at 10 m is a vessel 320 m long, longer than all but the largest
container ships and therefore larger than nearly every ship in the training set. That is a
prediction about the first run's recall rather than a defect to fix here, and `anchor_sizes` is
a config key so that changing it is one line and a second run.

What *is* done here is the channel count, because without it nothing runs at all: ImageNet
backbones take three channels and Sentinel-1 VV is one, so the amplitude is repeated across all
three. Repetition is the null adaptation — it keeps the pretrained first-layer filters
meaningful, since a grey image is a thing they have seen — and it is not what the ticket means
by an input stage adapted to radar polarisation, which is a dual-polarisation stem trained as
one. That is still to come.

This module is the only one in `detect/` that imports torch, along with `train.py`. Everything
that can be got wrong quietly — the split, the subset, the augmentations, the counting, the
resume — is on the other side of that line, in modules a laptop can test in a second.
"""

import numpy as np
import torch
from torch import Tensor
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

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
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
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
        pretrained: Start from COCO weights. A free tier gives too few epochs to train a
            ResNet-50 from scratch; set it false only where the session has no network to fetch
            them with.
        trainable_backbone_layers: How much of the backbone is unfrozen, from the top. Three of
            five is torchvision's default and is what the budget here affords.
    """
    # Applied here, before anything is constructed, because the head below is initialised from
    # scratch — two classes where COCO had 91 — and it draws from torch's global generator. Left
    # unseeded, two sessions of the same config start from two different models and report
    # different numbers, for a reason nothing in the config records. Found by running the same
    # configuration twice: see docs/failures.md.
    torch.manual_seed(seed)

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
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        min_size=tile_px,
        max_size=tile_px,
    )

    # The COCO head predicts 91 classes. Ships are one of them, and reusing that column would be
    # a defensible shortcut on optical imagery; on radar amplitude the features underneath it are
    # different enough that it is not worth the confusion of explaining. Fresh head, two classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, CLASSES
    )
    return model


def as_model_input(image: np.ndarray) -> Tensor:
    """One tile of amplitude in 0..1, as the three-channel image the backbone expects.

    Repeated rather than averaged or padded with zeros: a grey image is something the pretrained
    filters have seen, and two channels of zeros is not.
    """
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).repeat(3, 1, 1)
