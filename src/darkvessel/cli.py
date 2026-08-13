"""Command-line entry point.

Every stage of the pipeline is reachable from here, so that a run is reproducible from a
config file rather than from a sequence of notebook cells executed in the right order.

The detector is built here, from the configuration, and handed to the pipeline as an argument.
Choosing it is a property of the run; the pipeline itself never knows which one it got.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS

from darkvessel.data.ais import load_ais
from darkvessel.data.gee_export import Bounds, DateWindow, earth_engine, export_scene
from darkvessel.data.scene import Scene
from darkvessel.data.synthetic import write_synthetic_inputs
from darkvessel.data.tiling import Tiling
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

    export = commands.add_parser(
        "export", help="fetch a real Sentinel-1 scene from Earth Engine (needs credentials)"
    )
    export.add_argument("--config", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "synthesise":
        return _synthesise(args.out)
    if args.command == "export":
        return _export(args.config)
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
    # `ais: null` is a run with nothing to match against — a real scene before the level that
    # ingests real declarations. It is spelled out in the config rather than allowed by omission,
    # and what comes out is marked `unsearched`, never `dark`.
    declared = run_config["ais"]
    ais = None if declared is None else load_ais((relative_to / declared).resolve(), crs=scene.crs)
    tolerance_m = float(config["fusion"]["match_tolerance_m"])

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=_detector_from(run_config),
        tiling=_tiling_from(config["tiling"]),
        tolerance_m=tolerance_m,
    )

    output = (relative_to / run_config["output"]).resolve()
    write_detections(detections, output)

    counts = detections["status"].value_counts()
    verdict = (
        f"  {counts.get(MATCHED, 0)} matched, {counts.get(DARK, 0)} dark "
        f"at a tolerance of {tolerance_m:g} m"
        if ais is not None
        else "  no AIS supplied: nothing was searched, so no detection here is a dark vessel"
    )
    print(f"{len(detections)} detections in {scene.crs} -> {output}\n{verdict}")
    return 0


def export_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """Everything `export_scene` needs, read out of a config file.

    Separate from the command that runs it so that a config can be checked without credentials.
    Every other config fault in this package surfaces in a test run; without this split, a
    mistyped key in the one config that needs Earth Engine would surface only to someone who had
    already authenticated and waited.
    """
    imagery = config["imagery"]
    return {
        "area": Bounds(**config["area"]["bounds"]),
        "window": DateWindow(**_window_from(imagery["window"])),
        "polarisations": tuple(imagery["polarisations"]),
        "crs": config["area"]["crs"],
        "resolution_m": float(imagery["resolution_m"]),
        "path": (relative_to / config["export"]["out"]).resolve(),
    }


def _export(config_path: Path) -> int:
    """Fetch the scene the config names. The one command in this package that needs a network."""
    config = yaml.safe_load(config_path.read_text())
    request = export_request_from(config, config_path.parent)

    scene = export_scene(
        catalogue=earth_engine(project=config.get("earthengine", {}).get("project")),
        **request,
    )

    print(
        f"wrote {request['path']}\n"
        f"  {scene.id}\n"
        f"  acquired {scene.acquired_at.isoformat()}, {scene.orbit_pass.lower()} pass, "
        f"polarisations {', '.join(scene.polarisations)}"
    )
    return 0


def _window_from(window_config: dict[str, Any]) -> dict[str, datetime]:
    """Read the search window as written in the config.

    Zones are not supplied here if the config omitted them: a timestamp without one is read as
    local time, which shifts the window by an hour or two and can drop the acquisition at either
    end of it. `DateWindow` refuses such a window rather than have this guess at a zone.
    """
    return {end: datetime.fromisoformat(str(window_config[end])) for end in ("start", "end")}


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


def _tiling_from(tiling_config: dict[str, Any]) -> Tiling:
    """Build the tiling named by the config.

    How a scene is cut up is a property of the run — the detector it is cut for, and the memory
    that detector runs in — so it is chosen here rather than fixed in the pipeline. What the two
    numbers have to satisfy, and why, is in `tiling.py`.
    """
    return Tiling(
        size_px=int(tiling_config["size_px"]),
        overlap_px=int(tiling_config["overlap_px"]),
    )


def _detector_from(run_config: dict[str, Any]) -> Detector:
    """Build the detector named by the config. This is the injection point."""
    name = run_config["detector"]
    if name == "bright-pixel":
        return BrightPixelDetector(threshold=float(run_config["threshold"]))
    raise ValueError(f"unknown detector {name!r}; known detectors: 'bright-pixel'")
