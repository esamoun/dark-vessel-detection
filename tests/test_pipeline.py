"""The single seam: the full chain, with the detector injected.

Given a synthetic scene with bright targets at known ground coordinates, a synthetic AIS track,
and a trivial deterministic detector substituted for the trained model, the pipeline must return
the expected detections at the expected coordinates, each correctly classified as matched or
dark.

One test exercises the three paths where an error is silent: tiling (a target on a tile boundary
must be returned once, not zero or twice), georeferencing (a target must land at its true ground
coordinate), and matching (a moving vessel must match its interpolated position, not its stale
last report).

Detector quality is not tested here. A model is evaluated, not asserted.
"""
