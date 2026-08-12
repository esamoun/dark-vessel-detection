"""Pixel to ground coordinates.

Converts detections from tile-local pixels to scene pixels to projected coordinates, and writes
georeferenced vector output that opens directly in QGIS. Silent errors here produce plausible
detections in the wrong place, which is worse than a crash. Geometry-critical: covered by tests.
"""
