"""Tiling of large scenes, with overlap.

A Sentinel-1 scene does not fit in GPU memory. Tiles overlap so that vessels near a tile edge
are not cut in half; the overlap is what makes cross-tile deduplication possible downstream.
Geometry-critical: covered by tests.
"""
