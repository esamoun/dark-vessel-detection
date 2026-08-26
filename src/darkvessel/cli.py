"""Command-line entry point.

Every stage of the pipeline is reachable from here, so that a run is reproducible from a
config file rather than from a sequence of notebook cells executed in the right order.

The detector is built here, from the configuration, and handed to the pipeline as an argument.
Choosing it is a property of the run; the pipeline itself never knows which one it got.
"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from darkvessel.config import load_config
from darkvessel.data.ais import load_ais, slice_for, write_ais
from darkvessel.data.area import Bounds
from darkvessel.data.dma import danish_maritime_authority
from darkvessel.data.gee_export import DateWindow, earth_engine, export_archive, export_scene
from darkvessel.data.scene import Scene
from darkvessel.data.survey import survey as survey_traffic
from darkvessel.data.synthetic import write_synthetic_inputs
from darkvessel.data.tiling import Tiling
from darkvessel.detect.amplitude import DecibelStretch
from darkvessel.detect.checkpoints import Checkpoints, Journal
from darkvessel.detect.curve import curve, svg
from darkvessel.detect.curve import table as curve_table
from darkvessel.detect.dataset import Layout, Subset, catalogue, split_by_scene
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import to_ground, write_detections
from darkvessel.detect.infer import detect_scene
from darkvessel.detect.ladder import WINDOW, Rung, judge
from darkvessel.detect.ladder import table as ladder_table
from darkvessel.detect.metrics import Reporting
from darkvessel.detect.threshold import BrightPixelDetector
from darkvessel.embed.archive import Archive
from darkvessel.embed.crops import crops_for, has_measurements
from darkvessel.embed.embedder import Embedder
from darkvessel.embed.retrieval import (
    agreement,
    contact_sheet,
    extent,
    queries_over,
    retrieve,
    same_object,
)
from darkvessel.embed.retrieval import table as retrieval_table
from darkvessel.embed.views import Speckle, looks_of
from darkvessel.fusion.azimuth import Geometry
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

    survey_command = commands.add_parser(
        "survey", help="measure where the traffic is, to choose a study area (needs a network)"
    )
    survey_command.add_argument("--config", type=Path, required=True)

    train_command = commands.add_parser(
        "train", help="train the detector on labelled SAR (needs a GPU and the detector extra)"
    )
    train_command.add_argument("--config", type=Path, required=True)

    compare_command = commands.add_parser(
        "compare", help="read the rungs of a ladder of training runs against one another"
    )
    compare_command.add_argument("--config", type=Path, required=True)

    evaluate_command = commands.add_parser(
        "evaluate", help="the precision-recall curve of one run, banded by its last epochs"
    )
    # Either a ladder to read a rung out of, or a journal named outright. The second exists
    # because the run whose weights the chain loads is not always a rung of a ladder: R1 was
    # executed twice, and the execution that produced the shipped checkpoint is a journal in
    # docs/runs/ that no ladder judges.
    source = evaluate_command.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--metrics", type=Path)
    # Defaults to the rung the ladder kept, which is the one the chain loads. Naming a rejected
    # rung is allowed and is how the curves are put side by side. Ignored with --metrics.
    evaluate_command.add_argument("--rung", default=None)
    evaluate_command.add_argument("--svg", type=Path, default=None)

    scenes_command = commands.add_parser(
        "scenes", help="fetch every acquisition of the archive's window (needs credentials)"
    )
    scenes_command.add_argument("--config", type=Path, required=True)

    crops_command = commands.add_parser(
        "crops", help="cut the detections out of every archived scene (needs the detector extra)"
    )
    crops_command.add_argument("--config", type=Path, required=True)

    embed_command = commands.add_parser(
        "embed", help="fit a representation to the crops, without labels (needs the extra)"
    )
    embed_command.add_argument("--config", type=Path, required=True)

    retrieve_command = commands.add_parser(
        "retrieve", help="nearest neighbours over the archive, and the check that they hold"
    )
    retrieve_command.add_argument("--config", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "synthesise":
        return _synthesise(args.out)
    if args.command == "export":
        return _export(args.config)
    if args.command == "ais":
        return _ais(args.config)
    if args.command == "survey":
        return _survey(args.config)
    if args.command == "train":
        return _train(args.config)
    if args.command == "compare":
        return _compare(args.config)
    if args.command == "evaluate":
        return _evaluate(args.config, args.metrics, args.rung, args.svg)
    if args.command == "scenes":
        return _scenes(args.config)
    if args.command == "crops":
        return _crops(args.config)
    if args.command == "embed":
        return _embed(args.config)
    if args.command == "retrieve":
        return _retrieve(args.config)
    return _run(args.config)


def _synthesise(directory: Path) -> int:
    scene_path, ais_path = write_synthetic_inputs(directory)
    print(f"wrote {scene_path}\nwrote {ais_path}")
    return 0


def _run(config_path: Path) -> int:
    config = load_config(config_path)
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

    # Built and checked before the detector, so a tiling the model cannot run at is refused
    # before three hundred megabytes of weights are read off the disk.
    tiling = _tiling_from(config["tiling"])
    check_tile_size(run_config, tiling)

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=_detector_from(run_config, relative_to),
        tiling=tiling,
        geometry=geometry_from(config, scene.orbit_pass),
        embedder=_embedder_from(config, relative_to),
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


def geometry_from(config: dict[str, Any], orbit_pass: str | None) -> Geometry | None:
    """The orbit geometry the azimuth correction needs, or None where it cannot be had.

    The pass comes off the scene, never out of the config: it is a property of the acquisition,
    and a config that could name a different one would let a run correct every vessel in the
    wrong direction. The incidence angle comes out of the config, because the products this chain
    exports do not carry it yet — see `fusion/azimuth.py` and docs/decisions.md.

    A scene with no pass tag gets no correction. That is the synthetic scene, which has no
    satellite behind it, and any real product exported before the tag existed.
    """
    if orbit_pass is None:
        return None

    settings = config["fusion"].get("azimuth")
    if settings is None or not settings.get("correct", True):
        return None

    return Geometry(orbit_pass=orbit_pass, incidence_deg=float(settings["incidence_deg"]))


def fusion_settings_from(config: dict[str, Any]) -> dict[str, Any]:
    """What the matching stage takes from a config file, as `run` takes it.

    Separate from the command for the same reason as `export_request_from`: every test in this
    package writes its own config, so the shipped ones are the files nothing in the suite runs,
    and `configs/kattegat-lane.yaml` needs Earth Engine credentials before it can fail at all.
    Both go through this function in a test instead.
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
    config = load_config(config_path)
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


def survey_request_from(config: dict[str, Any]) -> dict[str, Any]:
    """What `survey` takes from a config file.

    Separate from the command for the reason `ais_request_from` is: this is the third stage that
    needs a network, and a mistyped key would otherwise surface to someone who had already waited
    for a day of Danish AIS.

    Nothing is written, so unlike the others there is no path to resolve. A survey answers a
    question rather than producing an input: what it decides is the rectangle someone then writes
    into an area config by hand, having read the argument for it.

    `report` is how many of the ranked rectangles the command prints, and it comes back here
    rather than being read in the command — like the path `_ais` pops off its own request. Every
    key of a shipped config goes through a function a test can call, or it becomes the one key
    nothing in the suite ever parses.
    """
    settings = config["survey"]
    return {
        "day": settings["day"],
        "region": Bounds(**settings["region"]),
        "box": (float(settings["box"]["lon_deg"]), float(settings["box"]["lat_deg"])),
        "stride": float(settings["stride_deg"]),
        "window": timedelta(seconds=float(settings["window_s"])),
        "min_length_m": float(settings["min_length_m"]),
        "under_way_kn": float(settings["under_way_kn"]),
        "report": int(settings["report"]),
    }


def _survey(config_path: Path) -> int:
    """Measure where the traffic is, and rank every rectangle the study area could be."""
    config = load_config(config_path)
    request = survey_request_from(config)
    report = request.pop("report")

    print(
        f"vessels of {request['min_length_m']:g} m or more, under way, in "
        f"{request['box'][0]:g} x {request['box'][1]:g} degree rectangles over "
        f"{request['day'].isoformat()}"
    )
    for candidate in survey_traffic(archive=danish_maritime_authority(), **request)[:report]:
        print(f"  {candidate.line()}")
    return 0


def training_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """Everything a training run takes from a config file.

    Separate from the command for the reason `export_request_from` is, with the constraint drawn
    tighter: this is the one stage that needs a GPU, and the machine that has one is rented by
    the hour and is not this one. A mistyped key would otherwise surface after the dataset had
    been attached, the wheels installed and the first epoch begun.

    Nothing here imports torch, and that is what lets a test parse the shipped config on a laptop
    with no framework installed. The command below imports it, once it has something to run.

    One seed, not several. The subset the run trains on, the way each tile is laid down, the
    order they arrive in, the weights the fresh detection head starts from and what the sampler
    inside an epoch draws are all derived from it, so a run is named by that number — and two
    runs of this file are the same experiment, which they were not until a Kaggle rebuild ran
    the same config twice and got two different models. See docs/failures.md.
    """
    data, out = config["data"], config["out"]
    seed = int(config["training"]["seed"])

    return {
        "root": (relative_to / data["root"]).resolve(),
        "layout": Layout(
            images=data["images"],
            annotations=data["annotations"],
            image_suffix=data["image_suffix"],
            # Empty means "measure it from the boxes", which is what a full dataset allows.
            first_index=None if data["first_index"] is None else int(data["first_index"]),
        ),
        "tile_px": int(data["tile_px"]),
        "subset": Subset(
            empty_per_ship_tile=float(config["subset"]["empty_per_ship_tile"]), seed=seed
        ),
        "model": {
            "seed": seed,
            "pretrained": bool(config["model"]["pretrained"]),
            "trainable_backbone_layers": int(config["model"]["trainable_backbone_layers"]),
            "anchor_sizes": tuple(tuple(level) for level in config["model"]["anchor_sizes"]),
            "stem": str(config["model"]["stem"]),
            "rpn_batch_size_per_image": int(config["model"]["rpn_batch_size_per_image"]),
            "rpn_positive_fraction": float(config["model"]["rpn_positive_fraction"]),
            "box_batch_size_per_image": int(config["model"]["box_batch_size_per_image"]),
            "box_positive_fraction": float(config["model"]["box_positive_fraction"]),
        },
        "schedule": {
            "epochs": int(config["schedule"]["epochs"]),
            "batch_size": int(config["schedule"]["batch_size"]),
            "learning_rate": float(config["schedule"]["learning_rate"]),
            "momentum": float(config["schedule"]["momentum"]),
            "weight_decay": float(config["schedule"]["weight_decay"]),
            "lr_schedule": str(config["schedule"]["lr_schedule"]),
            "workers": int(config["schedule"]["workers"]),
            "seed": seed,
        },
        "reporting": Reporting(
            tolerance_m=float(config["reporting"]["tolerance_m"]),
            resolution_m=float(data["resolution_m"]),
            thresholds=tuple(float(score) for score in config["reporting"]["thresholds"]),
        ),
        "checkpoints": Checkpoints(
            (relative_to / out["checkpoints"]).resolve(), keep=int(out["keep"])
        ),
        "journal": Journal((relative_to / out["metrics"]).resolve()),
        "device": config["training"].get("device"),
    }


def _train(config_path: Path) -> int:
    """Train the detector. The one command in this package that needs a GPU to be worth running.

    torch is imported here rather than at the top of the module because the chain's acceptance
    condition is that it installs and runs with no weights, no GPU and no network — see
    docs/decisions.md. `darkvessel run` must not pull two gigabytes of CUDA wheels to threshold
    bright pixels, and it does not.
    """
    import torch

    from darkvessel.detect.model import detector_model
    from darkvessel.detect.train import Schedule, train

    config = load_config(config_path)
    request = training_request_from(config, config_path.parent)

    refs = catalogue(request["root"], request["layout"])
    training, held_out = split_by_scene(refs)
    kept = request["subset"].of(training)

    print(f"{len(refs)} labelled tiles under {request['root']}")
    print(f"  training: {request['subset'].line(kept, out_of=len(training))}")
    print(
        f"  held out, scored entire: {len(held_out)} tiles carrying "
        f"{sum(len(ref.boxes) for ref in held_out)} ships"
    )

    device = torch.device(request["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    train(
        model=detector_model(tile_px=request["tile_px"], **request["model"]),
        training=kept,
        held_out=held_out,
        checkpoints=request["checkpoints"],
        journal=request["journal"],
        schedule=Schedule(**request["schedule"]),
        reporting=request["reporting"],
        device=device,
        built={"tile_px": request["tile_px"], **request["model"]},
        stem=request["model"]["stem"],
    )
    print(f"metrics in {request['journal'].path}")
    return 0


def ladder_request_from(config: dict[str, Any], relative_to: Path) -> list[dict[str, Any]]:
    """The rungs a ladder config names, with their metrics files resolved.

    Separate from the command for the reason `training_request_from` is: this file names five
    paths, and the last of them does not exist until five sessions on a rented GPU have finished.
    A mistyped key surfacing then would be the most expensive way to find it.
    """
    return [
        {
            "label": str(rung["label"]),
            "changed": str(rung["changed"]),
            "metrics": (relative_to / rung["metrics"]).resolve(),
        }
        for rung in config["ladder"]["rungs"]
    ]


def _read_ladder(config: dict[str, Any], relative_to: Path) -> list[Rung]:
    """The rungs a ladder config names, as far as they have actually been run.

    A rung whose metrics file is not there yet has not been run, which is the ordinary state of
    this file for most of a ticket. Reading stops there rather than skipping it: a ladder read
    across a gap would measure a change against the wrong configuration and would not look any
    different.

    A rung whose file exists, names its run and has scored no epoch is the same kind of pending —
    a session killed between `describe` writing the run block and the first epoch landing leaves
    exactly that file, and `best_f1(epochs[-1])` on an empty list is an `IndexError` rather than
    a verdict.
    """
    rungs: list[Rung] = []
    for requested in ladder_request_from(config, relative_to):
        if not requested["metrics"].exists():
            print(f"{requested['label']}: not run yet ({requested['metrics']})")
            break

        journal = Journal(requested["metrics"])
        entries = journal.entries()
        if not entries:
            print(f"{requested['label']}: no epoch scored yet ({requested['metrics']})")
            break

        rungs.append(
            Rung(
                label=requested["label"],
                changed=requested["changed"],
                run=journal.run(),
                epochs=entries,
            )
        )

    return rungs


def _compare(config_path: Path) -> int:
    """Read the ladder and say which rungs stand."""
    config = load_config(config_path)
    window = int(config["ladder"].get("window", WINDOW))

    rungs = _read_ladder(config, config_path.parent)
    if not rungs:
        print("no rung of this ladder has been run yet")
        return 0

    print(ladder_table(judge(rungs, window=window)))
    return 0


def _evaluate(
    config_path: Path | None, metrics_path: Path | None, label: str | None, svg_path: Path | None
) -> int:
    """Draw one run's precision-recall curve, banded by the epochs around it.

    Given a ladder, defaults to the rung it kept rather than to the last rung run, because that is
    the change that stands — a report of the last rung would describe, on this ladder, a change
    that was rejected. Given a journal, draws that journal, which is how a run nobody judged gets
    reported.
    """
    if metrics_path is not None:
        journal = Journal(metrics_path)
        entries = journal.entries()
        if not entries:
            print(f"no epoch scored yet ({metrics_path})")
            return 0
        return _draw(
            Rung(label=metrics_path.stem, changed="", run=journal.run(), epochs=entries),
            WINDOW,
            svg_path,
        )

    assert config_path is not None  # argparse requires one of the two
    config = load_config(config_path)
    window = int(config["ladder"].get("window", WINDOW))

    rungs = _read_ladder(config, config_path.parent)
    if not rungs:
        print("no rung of this ladder has been run yet")
        return 0

    by_label = {rung.label: rung for rung in rungs}
    if label is None:
        kept = [verdict for verdict in judge(rungs, window=window) if verdict.kept]
        chosen = by_label[kept[-1].label]
    elif label in by_label:
        chosen = by_label[label]
    else:
        print(f"no rung called {label!r} has been run; this ladder has {sorted(by_label)}")
        return 1

    return _draw(chosen, window, svg_path)


def _draw(chosen: Rung, window: int, svg_path: Path | None) -> int:
    """One run's curve, printed and optionally drawn."""
    points = curve(chosen.epochs, window=window)
    print(f"{chosen.label}{f' — {chosen.changed}' if chosen.changed else ''}")
    print(f"epoch {chosen.epochs[-1]['epoch']} of {len(chosen.epochs)}, banded over {window}")
    print(curve_table(points))

    if svg_path is not None:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(svg(points) + "\n")
        print(f"wrote {svg_path}")

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
    config = load_config(config_path)
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


def trained_request_from(run_config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What the trained detector takes from a config file.

    Separate from the command that builds it for the reason `training_request_from` is, with the
    constraint drawn the other way round: that one is the stage needing a GPU, this is the stage
    needing the framework to be installed at all. Read here, nothing about a run's spelling
    depends on torch, so a laptop with no detector extra still checks every key of it.

    `tile_px` and `anchor_sizes` are restated in the config rather than read off the checkpoint,
    because the first trained checkpoint predates `train.py` recording them. Where a checkpoint
    does carry its build block, `TrainedDetector` refuses a disagreement rather than preferring
    one side of it.
    """
    trained = run_config["trained"]
    stretch = trained["stretch"]

    return {
        "checkpoint": (relative_to / trained["checkpoint"]).resolve(),
        "tile_px": int(trained["tile_px"]),
        "anchor_sizes": tuple(
            tuple(int(size) for size in level) for level in trained["anchor_sizes"]
        ),
        # Optional where the rest of this block is required, because every run config written
        # before the single-channel stem existed names a checkpoint trained on three repeated
        # channels, and reading silence as anything else would break all of them.
        "stem": str(trained.get("stem", "repeat")),
        "score_threshold": float(trained["score_threshold"]),
        "stretch": DecibelStretch(
            floor_db=float(stretch["floor_db"]),
            ceiling_db=float(stretch["ceiling_db"]),
            sea_db=float(stretch["sea_db"]),
        ),
    }


def check_tile_size(run_config: dict[str, Any], tiling: Tiling) -> None:
    """Refuse a run whose tiles are not the size its detector was built for.

    The same shape of refusal `_check_working_crs` makes, and for the same reason. Torchvision
    resizes each tile to the size the model declares, silently, and resampling radar amplitude is
    a decision rather than a convenience: it changes what the detector sees, and the precision
    and recall reported for this model were measured at one scale and not the other.

    The stand-in has no opinion about tile size, so it is not asked.
    """
    if run_config["detector"] != "trained":
        return

    tile_px = int(run_config["trained"]["tile_px"])
    if tiling.size_px != tile_px:
        raise ValueError(
            f"the chain cuts {tiling.size_px} px tiles and the detector was built for {tile_px} "
            "px; the model would resize between the two, which changes what it sees. Set "
            "tiling.size_px to the model's, or run a model built for this tiling"
        )


def _detector_from(
    run_config: dict[str, Any], relative_to: Path, score_threshold: float | None = None
) -> Detector:
    """Build the detector named by the config. This is the injection point.

    `score_threshold` overrides the operating point the run declares, and there is exactly one
    caller that uses it. A run's threshold is chosen for precision, because every unmatched
    detection it publishes is a claim someone may be sent out on; an archive of crops publishes
    nothing and makes no claims, and a representation fitted only on the objects the detector was
    already certain about has never been shown the ones it was not. Passing it here rather than
    editing the run block is what keeps the two operating points in one config file, each stated
    where it belongs.
    """
    name = run_config["detector"]
    if name == "bright-pixel":
        threshold = run_config["threshold"] if score_threshold is None else score_threshold
        return BrightPixelDetector(threshold=float(threshold))
    if name == "trained":
        # Imported here rather than at the top of the module, the way `_train` imports torch: the
        # chain's acceptance condition is that it installs and runs with no framework, and a run
        # with the stand-in must not pull two gigabytes of CUDA wheels to threshold bright pixels.
        from darkvessel.detect.trained import TrainedDetector

        request = trained_request_from(run_config, relative_to)
        if score_threshold is not None:
            request["score_threshold"] = score_threshold
        return TrainedDetector(**request)
    raise ValueError(f"unknown detector {name!r}; known detectors: 'bright-pixel', 'trained'")


def archive_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What the archive of acquisitions is made of, read out of a config file.

    Separate from the commands that run it for the reason `export_request_from` is, doubled: the
    first of these stages needs Earth Engine credentials and the second needs the detector extra,
    so between them a mistyped key would surface either to someone who had already authenticated
    or to someone who had already fetched thirty scenes.

    `boxes` is a mapping of name to rectangle, and there is more than one of them because the
    archive is not the run. The chain's own study area is 17 km of open water chosen for its
    traffic, and it contains no fixed structures at all — so an archive drawn from it alone can
    never show a representation separating a turbine from a ship, whatever the representation
    does. The boxes are named rather than listed because a name reaches the provenance: two clips
    of one acquisition over two rectangles are two different pieces of water, with two sea states
    and two noise floors, and calling them one scene would put them in the same acquisition for
    every check that asks.

    `score_threshold` is the other number here worth arguing about. The chain's own threshold is
    chosen for precision, because every unmatched detection it publishes is a claim someone may
    be sent out on. An archive is not published and makes no claims: it is what a representation
    is fitted on, and a representation fitted only on the objects a detector is already certain
    about has never been shown the ones it is not. See docs/decisions.md.
    """
    archive = config["archive"]
    boxes = {name: Bounds(**bounds) for name, bounds in archive["boxes"].items()}
    if not boxes:
        raise ValueError("an archive with no boxes in it draws on no water at all")

    return {
        "boxes": boxes,
        "window": DateWindow(**_window_from(archive["window"])),
        "polarisations": tuple(config["imagery"]["polarisations"]),
        "crs": config["area"]["crs"],
        "resolution_m": float(config["imagery"]["resolution_m"]),
        "directory": (relative_to / archive["scenes"]).resolve(),
        "crops": (relative_to / archive["crops"]).resolve(),
        "score_threshold": float(archive["score_threshold"]),
    }


def embedding_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """Everything the embedding level takes from a config file.

    Nothing here imports torch, for the reason `training_request_from` states: the framework is
    an optional extra, so a test that had to import it to check a spelling would not run in CI at
    all. The schedule comes back as a plain dict and the command that has torch turns it into a
    `Schedule`.

    `speckle_looks` may be empty, which means views that change no pixel value — the eight
    symmetries and a translation and nothing else. It is spelled out rather than allowed by
    omission because it is the strongest statement this level makes about what a representation
    is asked to ignore.
    """
    embedding = config["embedding"]
    schedule = embedding["schedule"]
    looks = embedding["speckle_looks"]

    return {
        "enabled": bool(embedding["enabled"]),
        "crop_px": int(embedding["crop_px"]),
        "margin_px": int(embedding["margin_px"]),
        "dim": int(embedding["dim"]),
        "speckle": None if looks is None else Speckle(looks=float(looks)),
        "encoder": (relative_to / embedding["encoder"]).resolve(),
        "checkpoints": Checkpoints(
            (relative_to / embedding["checkpoints"]).resolve(), keep=int(embedding["keep"])
        ),
        "journal": Journal((relative_to / embedding["metrics"]).resolve()),
        "schedule": {
            "epochs": int(schedule["epochs"]),
            "batch_size": int(schedule["batch_size"]),
            "learning_rate": float(schedule["learning_rate"]),
            "temperature": float(schedule["temperature"]),
            "seed": int(schedule["seed"]),
        },
        "retrieval": {
            "queries": int(embedding["retrieval"]["queries"]),
            "neighbours": int(embedding["retrieval"]["neighbours"]),
            "sheet": (relative_to / embedding["retrieval"]["sheet"]).resolve(),
            "record": (relative_to / embedding["retrieval"]["record"]).resolve(),
        },
    }


def _scenes(config_path: Path) -> int:
    """Fetch every acquisition the archive's window covers, box by box.

    One subdirectory per box, because the file is named by the acquisition and one Sentinel-1
    product can cover two of them: written flat, the second clip would either overwrite the first
    or be skipped as already fetched, and both are the same wrong archive.
    """
    config = load_config(config_path)
    request = archive_request_from(config, config_path.parent)
    catalogue = earth_engine(project=config.get("earthengine", {}).get("project"))

    for name, area in request["boxes"].items():
        found = export_archive(
            catalogue=catalogue,
            area=area,
            window=request["window"],
            polarisations=request["polarisations"],
            crs=request["crs"],
            resolution_m=request["resolution_m"],
            directory=request["directory"] / name,
        )
        print(f"{name}: {len(found)} acquisition(s) in {request['directory'] / name}")

    return 0


def _crops(config_path: Path) -> int:
    """Cut every detection out of every archived scene, and keep them in one file.

    This runs the detector and places its answers on the ground without going through
    `pipeline.run`, and that is deliberate rather than a shortcut around the seam. The chain
    answers one question — which of these detections declared themselves — and it discards the
    pixel coordinates on the way, because nothing downstream of it needs them. A crop is a window
    on the image and needs exactly those, and it asks nothing of AIS at all.

    Resumable by acquisition, like `scenes`: the archive already names the scenes it holds, so a
    session that stopped after twenty of thirty picks up at the twenty-first. A scene is named by
    its box as well as its acquisition, so the same product clipped to two rectangles is two
    scenes here — which is what it is.
    """
    config = load_config(config_path)
    relative_to = config_path.parent
    request = archive_request_from(config, relative_to)
    embedding = embedding_request_from(config, relative_to)

    tiling = _tiling_from(config["tiling"])
    check_tile_size(config["run"], tiling)
    detector = _detector_from(
        config["run"], relative_to, score_threshold=request["score_threshold"]
    )

    path = request["crops"]
    archive = Archive.read(path) if path.exists() else None
    already = set(archive.scenes()) if archive is not None else set()

    scenes = [
        (name, scene_path)
        for name in request["boxes"]
        for scene_path in sorted((request["directory"] / name).glob("*.tif"))
    ]
    if not scenes:
        print(f"no scenes under {request['directory']}; run `darkvessel scenes` first")
        return 1

    for box, scene_path in scenes:
        name = f"{box}/{scene_path.stem}"
        # Decided before the file is opened, not after. `Scene.from_geotiff` reads the band
        # eagerly, so a resumed session that opened every scene to find out it had already cut it
        # would decode a gigabyte to skip it — which is not what "picks up at the twenty-first"
        # means.
        if name in already:
            continue

        scene = Scene.from_geotiff(scene_path)
        _check_working_crs(scene.crs, config["area"]["crs"])
        # An acquisition the catalogue listed and the clip came back empty of. `search` asks
        # whether a footprint *intersects* the rectangle, not whether it covers it, so a scene
        # can be exported whole and hold no water at all: three of the fifty over the Anholt box
        # are like this. Skipped and said out loud rather than crashing the run — an empty clip
        # is a known state of this archive, not a fault in it.
        if not has_measurements(scene.image):
            print(f"  {name}: no water in this clip; the acquisition does not cover the box")
            continue

        cut = _crops_of(
            scene,
            name,
            detector=detector,
            tiling=tiling,
            crop_px=embedding["crop_px"],
            margin_px=embedding["margin_px"],
        )
        archive = cut if archive is None else archive.with_more(cut)
        archive.write(path)
        print(
            f"  {name}: {len(cut)} crop(s) at {looks_of(scene.image):.1f} looks, "
            f"{len(archive)} in the archive"
        )

    for box in request["boxes"]:
        held = archive.provenance["scene"].str.startswith(f"{box}/").sum()
        print(f"{box}: {held} crops")
    print(f"{len(archive)} crops from {len(archive.scenes())} scene(s) -> {path}")
    return 0


def _crops_of(
    scene: Scene,
    name: str,
    *,
    detector: Detector,
    tiling: Tiling,
    crop_px: int,
    margin_px: int,
) -> Archive:
    """One scene's detections, as crops with everything needed to point at them again."""
    found = detect_scene(scene.image, detector, tiling)
    placed = to_ground(found, scene)

    return Archive(
        crops=crops_for(scene.image, found, crop_px=crop_px, margin_px=margin_px),
        provenance=pd.DataFrame(
            {
                "scene": [name] * len(found),
                "acquired_at": [scene.acquired_at.isoformat()] * len(found),
                "row": [detection.row for detection in found],
                "col": [detection.col for detection in found],
                "x": placed.geometry.x.to_numpy(),
                "y": placed.geometry.y.to_numpy(),
                "score": [detection.score for detection in found],
            }
        ),
        crop_px=crop_px,
        margin_px=margin_px,
    )


def _embed(config_path: Path) -> int:
    """Fit a representation to the archive, without labels.

    torch is imported here rather than at the top of the module, the way `_train` imports it: the
    chain's acceptance condition is that it installs and runs with no framework, and an embedding
    stage nobody enabled must not change that.
    """
    import torch

    from darkvessel.embed.contrastive import Schedule, train

    config = load_config(config_path)
    relative_to = config_path.parent
    request = embedding_request_from(config, relative_to)
    archive = Archive.read(archive_request_from(config, relative_to)["crops"])

    print(f"{len(archive)} crops of {archive.crop_px} px from {len(archive.scenes())} scene(s)")
    train(
        archive=archive,
        stretch=_stretch_for(config["run"], relative_to),
        speckle=request["speckle"],
        schedule=Schedule(**request["schedule"]),
        dim=request["dim"],
        # The distance at which this project already says two positions are one vessel. The
        # embedding level borrows it rather than declaring a second one; see `Archive.co_located`.
        tolerance_m=fusion_settings_from(config)["tolerance_m"],
        checkpoints=request["checkpoints"],
        journal=request["journal"],
        device=torch.device("cpu"),
    )

    # Copied into place only once the schedule is finished, so a session that stopped halfway
    # cannot leave the chain pointing at an encoder that was still moving. The detector level does
    # this by hand and records the checkpoint's hash in the config; this run is minutes rather
    # than evenings, so it is done here and said out loud.
    latest = request["checkpoints"].latest()
    if latest is not None and latest[0] >= request["schedule"]["epochs"]:
        request["encoder"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(latest[1], request["encoder"])
        print(f"epoch {latest[0]} -> {request['encoder']}")

    print(f"metrics in {request['journal'].path}")
    return 0


def _retrieve(config_path: Path) -> int:
    """Ask the archive what resembles what, and write down whether the answer means anything.

    The queries are chosen by spreading them over the range of target sizes in the archive rather
    than picked out by hand. A contact sheet of six queries is the most flattering figure this
    level can produce, and choosing the six is exactly where the flattery would get in.
    """
    config = load_config(config_path)
    relative_to = config_path.parent
    request = embedding_request_from(config, relative_to)
    retrieval = request["retrieval"]

    from darkvessel.embed.contrastive import ContrastiveEmbedder

    archive = Archive.read(archive_request_from(config, relative_to)["crops"])
    embedder = ContrastiveEmbedder(checkpoint=request["encoder"])
    vectors = embedder(archive.crops)
    names = _names_of(archive)

    sizes = extent(archive.crops, archive.crop_px)
    same_as = archive.co_located(fusion_settings_from(config)["tolerance_m"])
    scored = agreement(vectors, sizes, same_as=same_as)
    objects = same_object(vectors, same_as, archive.provenance["scene"].tolist())
    queries = queries_over(sizes, retrieval["queries"])
    found = [retrieve(vectors, names, query, retrieval["neighbours"]) for query in queries]

    epochs = request["journal"].entries()
    last = epochs[-1] if epochs else None

    print(f"{len(archive)} crops from {len(archive.scenes())} scene(s), {embedder.dim} dimensions")
    if last is not None:
        print(
            f"  twin recall {last['twin_recall']:.3f} against {last['chance']:.3f} at chance, "
            f"at epoch {last['epoch']}"
        )
    print(f"  {scored.line()}")
    print(f"  {objects.line()}")
    print(retrieval_table(found))

    retrieval["sheet"].parent.mkdir(parents=True, exist_ok=True)
    retrieval["sheet"].write_text(
        contact_sheet(archive.crops, found, stretch=embedder.stretch, crop_px=embedder.crop_px)
        + "\n"
    )
    retrieval["record"].parent.mkdir(parents=True, exist_ok=True)
    retrieval["record"].write_text(
        json.dumps(
            {
                "archive": {
                    "crops": len(archive),
                    "scenes": archive.scenes(),
                    "crop_px": archive.crop_px,
                    "margin_px": archive.margin_px,
                },
                "encoder": request["encoder"].name,
                "dim": embedder.dim,
                "twin_recall": last,
                "target_size_agreement": {
                    "retrieved_px": scored.retrieved,
                    "chance_px": scored.chance,
                },
                "same_object": {
                    "retrieved": objects.retrieved,
                    "chance": objects.chance,
                    "elsewhere_in_the_acquisition": objects.elsewhere,
                    "tolerance_m": fusion_settings_from(config)["tolerance_m"],
                },
                "queries": [
                    {
                        "query": row.name,
                        "neighbours": [
                            {"name": near.name, "similarity": near.similarity} for near in row.found
                        ],
                    }
                    for row in found
                ],
                "sheet": retrieval["sheet"].name,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {retrieval['sheet']}\nwrote {retrieval['record']}")
    return 0


def _names_of(archive: Archive) -> list[str]:
    """What each crop is called wherever it is reported: its acquisition, and which one it is.

    Short enough to fit under a cell of the contact sheet and long enough to find again — the
    provenance in the archive is what turns one of these back into a coordinate.
    """
    return [
        f"{str(acquired)[:16].replace('-', '').replace(':', '')}#{index}"
        for index, acquired in enumerate(archive.provenance["acquired_at"])
    ]


def _embedder_from(config: dict[str, Any], relative_to: Path) -> Embedder | None:
    """Build the embedder the config asks for, or none at all. The optional injection point.

    Absent and disabled are the same answer, and both are the answer for every config in this
    project that predates this level. The import is inside the branch, the way `_detector_from`
    imports the trained detector: a run with no embedding stage must not pull two gigabytes of
    CUDA wheels to write a layer that carries no vectors.
    """
    embedding = config.get("embedding")
    if embedding is None or not embedding.get("enabled", False):
        return None

    from darkvessel.embed.contrastive import ContrastiveEmbedder

    # Through the request function rather than reading the path here, so that the config a run
    # actually loads its encoder from is the config a test parses. Only `enabled` is read above,
    # and it is read before the rest because a stage that is off does not have to be spelled out
    # — every config in this project written before this level has no block at all.
    return ContrastiveEmbedder(checkpoint=embedding_request_from(config, relative_to)["encoder"])


def _stretch_for(run_config: dict[str, Any], relative_to: Path) -> DecibelStretch:
    """The window between decibels and amplitude the encoder is fitted under.

    The detector's own, rather than a second one declared beside it. A config that stated the
    window twice would be a config where the two could differ, and the difference would be
    invisible: both runs would work, and the representation would be fitted on an image the
    detector never sees. One statement of what a decibel means, per config.
    """
    if run_config["detector"] != "trained":
        raise ValueError(
            f"this config runs the {run_config['detector']!r} detector, which declares no window "
            "between decibels and amplitude; the embedding stage is fitted under the trained "
            "detector's window, so a config that enables it runs the trained detector"
        )
    return trained_request_from(run_config, relative_to)["stretch"]
