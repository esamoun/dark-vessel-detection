"""Detector architecture.

Two adaptations drive the design: vessels are a few pixels wide at 10 m resolution, so the
high-resolution levels of the feature pyramid carry the signal; and pretrained backbones expect
three colour channels where the input is single- or dual-polarisation radar amplitude.

Both are adaptations of a stock Faster R-CNN rather than a design from nothing, and that is the
point. What is specific to this project is the chain around the detector and the honesty of the
numbers coming out of it; the architecture is the part where the literature is already right,
and a hand-rolled one would be worse and would take the evenings that the rest of the chain
needs. What had to be changed is small, and each change is a number that can be checked.

The anchors are the first. A stock detector's smallest anchor is 32 px, which at 10 m is a
vessel 320 m long — longer than all but the largest container ships, and therefore larger than
every ship in the training set. Left alone, the region proposal network is looking for objects
none of which are present. The sizes below start at 4 px, a 40 m coastal fishing boat, and
double up the pyramid.

The second is the channel count. ImageNet backbones take three channels and Sentinel-1 VV is
one, so the amplitude is repeated across all three. That keeps the pretrained first-layer
filters meaningful — they are being shown a grey image, which is a thing they have seen — where
averaging them into a single-channel convolution throws away most of what was learnt. It costs
two-thirds of the first layer's arithmetic on data that carries no extra information, and that
is the cheapest part of the network.

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
BACKGROUND, SHIP = 0, 1
CLASSES = 2

# One tuple per level of the feature pyramid, in pixels. At 10 m these are hulls of 40, 80, 160,
# 320 and 640 m — the range from a coastal fishing boat to the largest thing that floats.
ANCHOR_SIZES = ((4,), (8,), (16,), (32,), (64,))

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
        anchor_sizes: One tuple per pyramid level. Configurable because this is the number most
            likely to want moving once there are real numbers to move it against.
        pretrained: Start from COCO weights. A free tier gives too few epochs to train a
            ResNet-50 from scratch; set it false only where the session has no network to fetch
            them with.
        trainable_backbone_layers: How much of the backbone is unfrozen, from the top. Three of
            five is torchvision's default and is what the budget here affords.
    """
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
