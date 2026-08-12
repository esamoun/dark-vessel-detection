"""Dataset and augmentations for SAR detection.

Only geometry-preserving augmentations are physically valid on radar amplitude: flips and
rotations yes, colour and contrast jitter no. Speckle perturbation is the radar-native option.
"""
