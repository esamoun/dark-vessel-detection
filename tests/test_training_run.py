"""The training run itself: the config it is defined by, and the interruption it is built for.

The claim this ticket rests on is that a session can be killed and the next one continues the
same run. That is not a claim a reader should have to take on trust, and it is not one that
should first be checked on a rented GPU at the end of an evening — so it is checked here, on the
CPU, on a dataset of eight tiles, against the real loop and the real model builder.

What this file does *not* do is assert anything about how well the detector detects. A test that
fixed a precision would turn a measurement into a target. The numbers this run produces are
reported in `docs/` and in the journal the run writes; what is asserted here is that the
machinery around them survives being switched off halfway through.

Skipped where torch is not installed, which includes CI: the chain's acceptance condition is
that it installs and runs without a framework, so the framework is an extra and the suite has to
be honest about running without it.
"""

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from test_dataset import FIXTURE, write_dataset  # noqa: E402

from darkvessel.cli import training_request_from  # noqa: E402
from darkvessel.detect.checkpoints import Checkpoints, Journal  # noqa: E402
from darkvessel.detect.dataset import catalogue, split_by_scene  # noqa: E402
from darkvessel.detect.model import detector_model  # noqa: E402
from darkvessel.detect.train import Reporting, Schedule, train  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"

TILE_PX = 64

pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")


def a_small_labelled_dataset(root: Path) -> Path:
    """Six tiles from a training scene and two from a held-out one, with ships in some of them."""
    return write_dataset(
        root,
        {
            f"{scene:02d}_{index}": [(20, 24, 27, 31)] if index % 2 else []
            for scene, indices in ((1, range(6)), (11, range(2)))
            for index in indices
        },
        size=TILE_PX,
    )


def a_run(tmp_path: Path, epochs: int) -> dict:
    """Everything one session of `train` needs, built fresh — model included.

    The model is fresh on purpose. A resumed session in the real world is a new process on a new
    machine with new weights in memory, and a test that reused the trained model would prove
    nothing about whether the checkpoint on the disk is what continues the run.
    """
    refs = catalogue(a_small_labelled_dataset(tmp_path / "data"), FIXTURE)
    training, held_out = split_by_scene(refs)

    return {
        # Untrained, because a test that downloaded 160 MB of COCO weights is not a test anyone
        # runs. Nothing here depends on the model being any good.
        "model": detector_model(tile_px=TILE_PX, pretrained=False, trainable_backbone_layers=5),
        "training": training,
        "held_out": held_out,
        "checkpoints": Checkpoints(tmp_path / "run", keep=2),
        "journal": Journal(tmp_path / "run" / "metrics.json"),
        "schedule": Schedule(
            epochs=epochs,
            batch_size=2,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=0.0005,
            workers=0,
            seed=1,
        ),
        "reporting": Reporting(tolerance_m=200.0, resolution_m=10.0, thresholds=(0.05, 0.5)),
        "device": torch.device("cpu"),
        "say": lambda line: None,
    }


def test_a_second_session_continues_the_run_the_first_one_started(tmp_path: Path) -> None:
    """The whole design, end to end: run one epoch, throw the process away, run again.

    The second session is given a longer schedule and a model with untrained weights, and it has
    to pick up at epoch 2 rather than starting over — which is what an evening on a free tier
    actually looks like.
    """
    train(**(a_run(tmp_path, epochs=1)))
    assert Checkpoints(tmp_path / "run").next_epoch() == 2

    train(**(a_run(tmp_path, epochs=2)))

    journal = Journal(tmp_path / "run" / "metrics.json")
    assert [entry["epoch"] for entry in journal.entries()] == [1, 2]
    assert Checkpoints(tmp_path / "run").next_epoch() == 3


def test_a_finished_schedule_asked_to_run_again_does_nothing(tmp_path: Path) -> None:
    """Restarting a session after the schedule has finished is the ordinary case on a free tier:
    the notebook is re-run and the cell is the same cell. It should not quietly train a second
    schedule on top of the first."""
    train(**(a_run(tmp_path, epochs=1)))

    train(**(a_run(tmp_path, epochs=1)))

    assert [entry["epoch"] for entry in Journal(tmp_path / "run" / "metrics.json").entries()] == [1]


def test_a_finished_epoch_reports_a_precision_and_a_recall_on_the_held_out_split(
    tmp_path: Path,
) -> None:
    """The output of this ticket, in the file that holds it. What the numbers *are* is not
    asserted — an untrained ResNet on eight tiles has no business being held to a figure — only
    that the run measured the held-out split and wrote down what it found."""
    train(**(a_run(tmp_path, epochs=1)))

    entry = Journal(tmp_path / "run" / "metrics.json").entries()[0]

    assert entry["held_out_tiles"] == 2
    assert entry["held_out_ships"] == 1
    assert [point["score"] for point in entry["at"]] == [0.05, 0.5]
    assert set(entry["at"][0]) == {"score", "precision", "recall", "found", "false", "missed"}


def test_the_shipped_training_config_is_the_one_the_command_parses() -> None:
    """`configs/train.yaml` through the command's own parsing.

    The same gap `export_request_from` exists to close, drawn tighter: every other test here
    builds its own settings, and this file is run on a machine that is rented by the hour and is
    not this one. A mistyped key would otherwise surface after the dataset had been attached, the
    wheels installed and the first epoch begun.
    """
    request = training_request_from(yaml.safe_load(CONFIG.read_text()), CONFIG.parent)

    assert request["schedule"]["epochs"] > 0
    assert request["subset"].empty_per_ship_tile >= 0.0
    assert request["reporting"]["thresholds"] == tuple(sorted(request["reporting"]["thresholds"]))
    assert len(request["model"]["anchor_sizes"]) == 5, "one anchor size per level of the pyramid"
    # The Kaggle attachment points are absolute, and have to survive being read relative to a
    # config file that lives in this repository rather than in the session.
    assert request["root"].is_absolute() and request["checkpoints"].directory.is_absolute()


def test_the_shipped_config_reports_the_tolerance_the_fusion_will_use(tmp_path: Path) -> None:
    """The detector is scored by the rule the chain will later apply to it. Let the two drift and
    the precision in the README stops describing what the pipeline does with the detections."""
    training = training_request_from(yaml.safe_load(CONFIG.read_text()), CONFIG.parent)
    pipeline = yaml.safe_load((CONFIG.parent / "kattegat-lane.yaml").read_text())

    assert training["reporting"]["tolerance_m"] == float(pipeline["fusion"]["match_tolerance_m"])
    assert training["reporting"]["resolution_m"] == float(pipeline["imagery"]["resolution_m"])
