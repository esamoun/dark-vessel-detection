"""Detector architecture.

Two adaptations drive the design: vessels are a few pixels wide at 10 m resolution, so the
high-resolution levels of the feature pyramid carry the signal; and pretrained backbones expect
three colour channels where the input is single- or dual-polarisation radar amplitude.
"""
