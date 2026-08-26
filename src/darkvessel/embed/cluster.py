"""Clustering over the embedding space.

Separating vessels from offshore wind turbines, which are bright point scatterers that a detector
trained on ships will happily return. That is issue #14 and Level 3, and it is not implemented
here yet: the study area moved onto the shipping lane and off the Anholt wind farm — see
docs/decisions.md, 2026-08-14 — so the archive this would cluster currently holds no turbines to
separate, and a clustering fitted on data containing none of the class it exists to find would be
a figure rather than a finding.

Nearest-neighbour retrieval, which used to be described here as this module's second use, is in
`retrieval.py`. It needs no clustering and no labels, which is why it ships first.
"""
