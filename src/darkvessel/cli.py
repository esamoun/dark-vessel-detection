"""Command-line entry point.

Every stage of the pipeline is reachable from here, so that a run is reproducible from a
config file rather than from a sequence of notebook cells executed in the right order.

The detector is built here, from the configuration, and handed to the pipeline as an argument.
Choosing it is a property of the run; the pipeline itself never knows which one it got.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from pyproj import CRS

from darkvessel.data.ais import load_ais, slice_for, write_ais
from darkvessel.data.area import Bounds
from darkvessel.data.dma import danish_maritime_authority
from darkvessel.data.gee_export import DateWindow, earth_engine, export_scene
from darkvessel.data.scene import Scene
from darkvessel.data.synthetic import write_synthetic_inputs
from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import write_detections
from darkvessel.detect.threshold import BrightPixelDetector
from darkvessel.fusion.interpolate import INTERPOLATED, REPORTED
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

    ais = commands.add_parser(
        "ais", help="fetch the declared positions for a scene's acquisition (needs a network)"
    )
    ais.add_argument("--config", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "synthesise":
        return _synthesise(args.out)
    if args.command == "export":
        return _export(args.config)
    if args.command == "ais":
        return _ais(args.config)
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
    fusion = fusion_settings_from(config)

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=_detector_from(run_config),
        tiling=_tiling_from(config["tiling"]),
        **fusion,
    )
    output = (relative_to / run_config["output"]).resolve()
    write_detections(detections, output)

    print(f"{len(detections)} detections in {scene.crs} -> {output}")
    verdict = _verdict(
        detections,
        searched=ais is not None,
        tolerance_m=fusion["tolerance_m"],
        # One placed position per vessel, which is what `positions_at` returns and what the
        # matching was run against — not the number of reports the slice happens to hold.
        declarations=0 if ais is None else int(ais["mmsi"].nunique()),
    )
    for line in verdict:
        print(f"  {line}")
    return 0


def _verdict(
    detections: gpd.GeoDataFrame,
    *,
    searched: bool,
    tolerance_m: float,
    declarations: int,
) -> list[str]:
    """What the run found, said out loud.

    The second line is the same claim `position_basis` makes in the layer, made on the way past:
    a match against a position built at the acquisition instant and one against a report taken
    from another moment are different claims, and someone who has run the command but not yet
    opened the output should not have to assume they are the same.

    `declarations` is passed in rather than read back off the layer. A scene with no detections
    has no row to read it from, and defaulting to zero there would print the warning below over a
    slice of hundreds — the exact wrong-but-plausible claim the count exists to prevent. It is
    the same number `classify` writes into every row, and a test holds the two together.
    """
    if not searched:
        return ["no AIS supplied: nothing was searched, so no detection here is a dark vessel"]

    counts = detections["status"].value_counts()
    bases = detections["position_basis"].value_counts()
    verdict = [
        f"{counts.get(MATCHED, 0)} matched, {counts.get(DARK, 0)} dark "
        f"at a tolerance of {tolerance_m:g} m, against {declarations} declared positions",
        f"of those matches, {bases.get(INTERPOLATED, 0)} on a position interpolated to the "
        f"acquisition and {bases.get(REPORTED, 0)} on a report taken as it stands",
    ]
    if not declarations and len(detections):
        # Technically dark, and it reads as its opposite. The archive was searched and held no
        # vessel in this scene at this instant, which is not the same discovery as a scene full
        # of ships that switched their transponders off.
        verdict.append(
            "no vessel declared itself in this scene at this instant, so every detection is "
            "dark by default rather than by evidence: check the ingestion covered the area"
        )
    return verdict


def fusion_settings_from(config: dict[str, Any]) -> dict[str, Any]:
    """What the matching stage takes from a config file, as `run` takes it.

    Separate from the command for the same reason as `export_request_from`: every test in this
    package writes its own config, so the shipped ones are the files nothing in the suite runs,
    and `configs/anholt.yaml` needs Earth Engine credentials before it can fail at all. Both go
    through this function in a test instead.
    """
    fusion = config["fusion"]
    return {
        "tolerance_m": float(fusion["match_tolerance_m"]),
        "max_gap": timedelta(seconds=float(fusion["interpolation_max_gap_s"])),
    }


def ais_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What `slice_for` takes from a config file, and where its answer is written.

    Separate from the command that runs it for the reason `export_request_from` is: this is the
    other stage that needs a network, and a mistyped key here would otherwise surface to someone
    who had already waited for most of a gigabyte of Danish AIS.

    The acquisition is not read from here. It comes off the scene the run points at, so a slice
    and the scene it is matched against cannot describe two different moments — a config that
    named the moment separately would let them.
    """
    ais = config["ais"]
    return {
        "area": Bounds(**config["area"]["bounds"]),
        "window": timedelta(seconds=float(ais["window_s"])),
        "margin_m": float(ais["margin_m"]),
        "max_speed_kn": float(ais["max_speed_kn"]),
        "path": (relative_to / ais["out"]).resolve(),
    }


def _ais(config_path: Path) -> int:
    """Fetch, filter and clean the declarations for the scene this config runs on.

    Split from `run` the way `export` is, and for the same reason: the network happens once and
    the chain runs from the result as often as it likes. What this writes is the file `run.ais`
    names, so a run stays one command against one config file.
    """
    config = yaml.safe_load(config_path.read_text())
    request = ais_request_from(config, config_path.parent)
    path = request.pop("path")

    # The whole scene is opened to read one tag off it, which is a few megabytes of pixels
    # nobody here wants. Worth it: `Scene.from_geotiff` refuses a product with no acquisition
    # time rather than guessing one, and a second way of reading that tag is a second place for
    # the refusal to go missing. An hour of drift here is 22 km of vessel track.
    scene = (config_path.parent / config["run"]["scene"]).resolve()
    acquired_at = Scene.from_geotiff(scene).acquired_at

    print(f"declared positions around {acquired_at.isoformat()}, from {scene.name}")
    reports, cleaning = slice_for(
        archive=danish_maritime_authority(), acquired_at=acquired_at, **request
    )
    write_ais(reports, path)

    for line in cleaning.lines():
        print(f"  {line}")
    print(f"wrote {path}")
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
