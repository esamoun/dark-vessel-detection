"""Command-line entry point.

Every stage of the pipeline is reachable from here, so that a run is reproducible from a
config file rather than from a sequence of notebook cells executed in the right order.

The detector is built here, from the configuration, and handed to the pipeline as an argument.
Choosing it is a property of the run; the pipeline itself never knows which one it got.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS

from darkvessel.data.ais import load_ais
from darkvessel.data.scene import Scene
from darkvessel.data.synthetic import write_synthetic_inputs
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import write_detections
from darkvessel.detect.threshold import BrightPixelDetector
from darkvessel.fusion.match import DARK, MATCHED
from darkvessel.pipeline import run as run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darkvessel", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_command = commands.add_parser("run", help="run the pipeline described by a config file")
    run_command.add_argument("--config", type=Path, required=True)

    synthesise = commands.add_parser(
        "synthesise", help="write a synthetic scene and AIS slice to run the pipeline on"
    )
    synthesise.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "synthesise":
        return _synthesise(args.out)
    return _run(args.config)


def _synthesise(directory: Path) -> int:
    scene_path, ais_path = write_synthetic_inputs(directory)
    print(f"wrote {scene_path}\nwrote {ais_path}")
    return 0


def _run(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text())
    run_config = config["run"]
    # Paths are read relative to the config file, so a config is portable and a run is defined
    # entirely by that one file plus the data it points at.
    relative_to = config_path.parent

    scene = Scene.from_geotiff((relative_to / run_config["scene"]).resolve())
    _check_working_crs(scene.crs, config["area"]["crs"])
    ais = load_ais((relative_to / run_config["ais"]).resolve(), crs=scene.crs)
    tolerance_m = float(config["fusion"]["match_tolerance_m"])

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=_detector_from(run_config),
        tolerance_m=tolerance_m,
    )

    output = (relative_to / run_config["output"]).resolve()
    write_detections(detections, output)

    counts = detections["status"].value_counts()
    print(
        f"{len(detections)} detections in {scene.crs} -> {output}\n"
        f"  {counts.get(MATCHED, 0)} matched, {counts.get(DARK, 0)} dark "
        f"at a tolerance of {tolerance_m:g} m"
    )
    return 0


def _check_working_crs(scene_crs: str, working_crs: str) -> None:
    """Refuse a scene that is not in the CRS the run declares.

    The tolerance is a distance in metres and the matching compares it against coordinate
    distances. Hand this a scene in degrees and nothing crashes: every detection is simply
    matched or called dark for the wrong reason. Reprojecting instead of refusing would be
    worse still — resampling radar amplitude is a decision, not a convenience.
    """
    if CRS.from_user_input(scene_crs) != CRS.from_user_input(working_crs):
        raise ValueError(
            f"scene is in {scene_crs} but the run declares a working CRS of {working_crs}; "
            "reproject the scene, or correct area.crs in the config"
        )


def _detector_from(run_config: dict[str, Any]) -> Detector:
    """Build the detector named by the config. This is the injection point."""
    name = run_config["detector"]
    if name == "bright-pixel":
        return BrightPixelDetector(threshold=float(run_config["threshold"]))
    raise ValueError(f"unknown detector {name!r}; known detectors: 'bright-pixel'")
