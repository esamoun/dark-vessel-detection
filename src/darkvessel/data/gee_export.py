"""Sentinel-1 selection and export via Earth Engine.

Scenes are selected and exported server-side so that full GRD products never transit the local
disk. Records acquisition time per scene: the AIS fusion stage depends on it being exact.
"""
