"""Interpolation of AIS tracks to the exact acquisition time.

A vessel moves between its last AIS report and the moment the radar imaged it. Comparing a
detection to a stale position manufactures dark vessels that do not exist. Positions are
interpolated along the track to the acquisition timestamp before any matching occurs.
"""
