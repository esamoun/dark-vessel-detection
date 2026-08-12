"""Full-scene inference.

Tile, detect, then merge: detections duplicated across overlapping tiles are reconciled in
scene coordinates, not tile coordinates. This is the step that turns a model into a chain.
"""
