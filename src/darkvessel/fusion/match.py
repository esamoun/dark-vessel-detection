"""Spatio-temporal matching, and the dark vessel decision.

Detections are matched to interpolated AIS positions within a tolerance derived from position
uncertainty and geolocation error. What remains unmatched is reported as a candidate, with its
tolerance stated - a claim about evidence, not a verdict. Geometry-critical: covered by tests.
"""
