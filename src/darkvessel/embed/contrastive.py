"""Contrastive training on detection crops, without labels.

Learns a representation in which visually similar objects sit close together, so that classes
never annotated - fixed structures, small craft, large hulls - become separable after the fact.
"""
