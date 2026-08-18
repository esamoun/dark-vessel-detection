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

import inspect
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from test_dataset import FIXTURE, write_dataset  # noqa: E402

import darkvessel.detect.train as train_module  # noqa: E402
from darkvessel.cli import _train, training_request_from  # noqa: E402
from darkvessel.config import load_config  # noqa: E402
from darkvessel.detect.checkpoints import Checkpoints, Journal  # noqa: E402
from darkvessel.detect.dataset import Layout, catalogue, split_by_scene  # noqa: E402
from darkvessel.detect.metrics import Reporting  # noqa: E402
from darkvessel.detect.model import ANCHOR_SIZES, detector_model  # noqa: E402
from darkvessel.detect.train import Schedule, train  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
LADDER = CONFIG.parent / "ladder"

TILE_PX = 64

# Eight tiles cannot settle where the annotations start counting — no box touches an edge — so
# this one says. Over the real 9000 it is measured; see `dataset._first_index`.
LAYOUT = Layout(image_suffix=FIXTURE.image_suffix, first_index=0)

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


def a_run(tmp_path: Path, epochs: int, lr_schedule: str = "constant") -> dict:
    """Everything one session of `train` needs, built fresh — model included.

    The model is fresh on purpose. A resumed session in the real world is a new process on a new
    machine with new weights in memory, and a test that reused the trained model would prove
    nothing about whether the checkpoint on the disk is what continues the run.
    """
    refs = catalogue(a_small_labelled_dataset(tmp_path / "data"), LAYOUT)
    training, held_out = split_by_scene(refs)

    return {
        # Untrained, because a test that downloaded 160 MB of COCO weights is not a test anyone
        # runs. Nothing here depends on the model being any good.
        "model": detector_model(
            tile_px=TILE_PX, seed=1, pretrained=False, trainable_backbone_layers=5
        ),
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
            lr_schedule=lr_schedule,
        ),
        "reporting": Reporting(tolerance_m=200.0, resolution_m=10.0, thresholds=(0.05, 0.5)),
        "device": torch.device("cpu"),
        # What built the model above. Stated here rather than derived, because a fixture whose
        # build block described a different model would be testing nothing.
        "built": {
            "tile_px": TILE_PX,
            "anchor_sizes": ANCHOR_SIZES,
            "seed": 1,
            "pretrained": False,
            "trainable_backbone_layers": 5,
        },
        "say": lambda line: None,
    }


def a_head(seed: int):
    """The classification layer of a freshly built detector, which is the part with no weights
    to inherit: COCO predicts 91 classes and this predicts two, so the head is always new."""
    model = detector_model(
        tile_px=TILE_PX, seed=seed, pretrained=False, trainable_backbone_layers=5
    )
    return model.roi_heads.box_predictor.cls_score.weight


def test_the_seed_names_the_weights_and_not_only_the_data() -> None:
    """What a Kaggle rebuild found, pinned.

    Saving a version re-runs the whole notebook in a fresh machine, so the same config ran twice
    and reported two different sets of numbers — 1903 detections against 1877 at the same
    threshold of the same epoch. The data pipeline was seeded throughout; the head was not, and
    two runs therefore started from two different models. Nothing in the config recorded the
    difference, which is the part that mattered. See docs/failures.md.
    """
    assert torch.equal(a_head(seed=20260814), a_head(seed=20260814))
    assert not torch.equal(a_head(seed=20260814), a_head(seed=20260815))


def test_a_second_session_continues_the_run_the_first_one_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole design, end to end: start the run, throw the process away, run the same command.

    The second session gets a model with untrained weights and the same schedule the first one
    declared, and it has to pick up at epoch 2 rather than starting over — which is what an
    evening on a free tier actually looks like: the notebook is re-run, and it is the same cell.

    The schedule's length is part of what names the run, so a session does not get to extend it.
    Under a decaying rate the declared length is what the rate is annealed over, so a longer
    horizon is a different experiment rather than a longer one, and `describe` refuses it. That
    is why the interruption here is injected rather than simulated by declaring a shorter
    schedule first, which is what this test used to do.
    """
    real_one_epoch = train_module._one_epoch

    def dies_before_epoch_two(model, optimiser, training, epoch, schedule, device, **kwargs):
        if epoch == 2:
            raise KeyboardInterrupt
        return real_one_epoch(model, optimiser, training, epoch, schedule, device, **kwargs)

    monkeypatch.setattr(train_module, "_one_epoch", dies_before_epoch_two)
    with pytest.raises(KeyboardInterrupt):
        train(**(a_run(tmp_path, epochs=2)))
    monkeypatch.undo()

    assert Checkpoints(tmp_path / "run").next_epoch() == 2

    train(**(a_run(tmp_path, epochs=2)))

    journal = Journal(tmp_path / "run" / "metrics.json")
    assert [entry["epoch"] for entry in journal.entries()] == [1, 2]
    assert Checkpoints(tmp_path / "run").next_epoch() == 3


def test_a_resume_declaring_a_different_number_of_epochs_is_refused(tmp_path: Path) -> None:
    """`epochs` names the run along with everything else in the schedule, and this is the test
    that holds that in place: it was once left out of the run block, and put back on the
    argument below. It is not enough for the argument to live in a comment.

    Under a decaying rate the declared length is what the rate is annealed over — cosine's
    `T_max` is `schedule.epochs` — so a session resumed with a longer horizon is a different
    experiment from the one that was running, not a longer version of it. A schedule that had
    already reached `eta_min` would otherwise resume at a learning rate of zero and train every
    remaining epoch there, silently, on a machine rented by the hour. `describe` refuses the
    resume rather than merging the two.
    """
    train(**(a_run(tmp_path, epochs=2)))

    assert Journal(tmp_path / "run" / "metrics.json").run()["schedule"]["epochs"] == 2

    with pytest.raises(ValueError, match="epochs"):
        train(**(a_run(tmp_path, epochs=4)))


def test_an_epoch_whose_weights_landed_but_whose_score_did_not_is_scored(tmp_path: Path) -> None:
    """The gap the ordering inside an epoch opens, and the thing that closes it.

    Weights are written before the held-out split is scored, on the argument that numbers can be
    recomputed from a checkpoint and an epoch cannot. That argument only holds if something
    recomputes them. Kill the session in exactly that window on the last epoch of the schedule
    and the loop below never runs again — so the final checkpoint would sit there with no
    precision against it and nothing left that would ever produce one.
    """
    run = a_run(tmp_path, epochs=1)
    train(**run)
    Journal(tmp_path / "run" / "metrics.json").path.unlink()

    train(**(a_run(tmp_path, epochs=1)))

    scored = Journal(tmp_path / "run" / "metrics.json").entries()
    assert [entry["epoch"] for entry in scored] == [1]
    # Scored from the checkpoint, so the loss the interrupted session measured is gone. Saying
    # so beats writing a zero nobody measured.
    assert scored[0]["training_loss"] is None
    assert scored[0]["held_out_ships"] == 1


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


def test_a_resumed_session_continues_the_learning_rate_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure a schedule introduces, and the one this design cannot afford.

    Everything about a run is derived from the seed and the epoch number rather than carried in a
    generator's position, so a session resumed at epoch 3 does what an uninterrupted run would
    have done there. A learning-rate scheduler is the first piece of state in this loop that does
    not work that way: left out of the checkpoint it restarts from the top, the resumed session
    trains its remaining epochs at the wrong rate, and nothing anywhere says so.

    So an interrupted run and an uninterrupted one are required to report the same rates. Both
    declare the same four-epoch schedule throughout — cosine anneals over the schedule it is
    told, so a session that only found out at epoch 3 that the run was ever meant to be four
    epochs long would already be annealing over the wrong horizon; a resume has to mean the same
    schedule picked back up, not a shorter one stretched afterwards. The kill is injected between
    epoch 2's checkpoint and epoch 3's training by replacing `_one_epoch` outright — in the same
    spirit as `test_checkpoints.py` raising inside `checkpoints.writing`, one level up — so this
    is a session that never got to run epoch 3 at all.
    """
    straight = tmp_path / "straight"
    train(**(a_run(straight, epochs=4, lr_schedule="cosine")))
    uninterrupted = [
        entry["learning_rate"] for entry in Journal(straight / "run" / "metrics.json").entries()
    ]

    killed = tmp_path / "killed"
    real_one_epoch = train_module._one_epoch

    def dies_before_epoch_three(model, optimiser, training, epoch, schedule, device, **kwargs):
        if epoch == 3:
            raise KeyboardInterrupt
        return real_one_epoch(model, optimiser, training, epoch, schedule, device, **kwargs)

    monkeypatch.setattr(train_module, "_one_epoch", dies_before_epoch_three)
    with pytest.raises(KeyboardInterrupt):
        train(**(a_run(killed, epochs=4, lr_schedule="cosine")))
    monkeypatch.undo()

    train(**(a_run(killed, epochs=4, lr_schedule="cosine")))
    resumed = [
        entry["learning_rate"] for entry in Journal(killed / "run" / "metrics.json").entries()
    ]

    assert len(uninterrupted) == 4
    assert resumed == pytest.approx(uninterrupted)
    # Pinned outright rather than checked for "decreasing": a cosine trajectory is a closed-form
    # property of the schedule, not a measurement of the detector, so a reader who has never
    # computed one can see here what it does. Without this line a `_scheduler` that quietly
    # returned None for "cosine" would still pass every test in this file.
    assert uninterrupted == pytest.approx(
        [0.001, 0.0008535533905932737, 0.0005, 0.00014644660940672628]
    )


def test_a_constant_schedule_reports_the_one_rate_it_trained_at(tmp_path: Path) -> None:
    """The baseline of the ladder, and the shape the failure log's diagnosis rests on: the rate
    never moved, which is why twelve epochs bounced instead of settling."""
    train(**(a_run(tmp_path, epochs=2)))

    rates = [
        entry["learning_rate"] for entry in Journal(tmp_path / "run" / "metrics.json").entries()
    ]

    assert rates == [0.001, 0.001]


def test_a_schedule_this_project_does_not_have_is_refused_by_name() -> None:
    """The refusal lives in `Schedule.__post_init__` now, not in `train`, so it is pinned there
    directly rather than through an entry point it no longer reaches."""
    with pytest.raises(ValueError, match="lr_schedule"):
        Schedule(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=0.0005,
            workers=0,
            seed=1,
            lr_schedule="exponential",
        )


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
    assert request["reporting"].thresholds == tuple(sorted(request["reporting"].thresholds))
    assert len(request["model"]["anchor_sizes"]) == 5, "one anchor size per level of the pyramid"
    # The Kaggle attachment points are absolute, and have to survive being read relative to a
    # config file that lives in this repository rather than in the session.
    assert request["root"].is_absolute() and request["checkpoints"].directory.is_absolute()
    # `lr_schedule` reaches `Schedule` through this key alone. Left out of the dict, `Schedule`'s
    # own default ("constant") takes over silently — no exception, just twelve epochs at the
    # wrong rate, on the run whose entire reason for existing is that this one line differs.
    # `.get` rather than bare indexing, so a deleted key fails this assertion by name instead of
    # raising a `KeyError` that reads as a broken test rather than a caught regression.
    assert request["schedule"].get("lr_schedule") in train_module._LR_SCHEDULES
    # Every key in this dict is unpacked straight into `detector_model(**request["model"])`. A
    # key this dict stops producing is a keyword `detector_model` falls back to its own default
    # for, silently — the same failure mode as `lr_schedule` above, one call away from the
    # builder instead of the scheduler. `tile_px` is `_train`'s own to pass, not this dict's, so
    # it is excluded from the builder's parameter set before comparing.
    builder_params = set(inspect.signature(detector_model).parameters) - {"tile_px"}
    assert set(request["model"]) == builder_params


def test_the_shipped_config_reports_the_tolerance_the_fusion_will_use(tmp_path: Path) -> None:
    """The detector is scored by the rule the chain will later apply to it. Let the two drift and
    the precision in the README stops describing what the pipeline does with the detections."""
    training = training_request_from(yaml.safe_load(CONFIG.read_text()), CONFIG.parent)
    pipeline = yaml.safe_load((CONFIG.parent / "kattegat-lane.yaml").read_text())

    assert training["reporting"].tolerance_m == float(pipeline["fusion"]["match_tolerance_m"])
    assert training["reporting"].resolution_m == float(pipeline["imagery"]["resolution_m"])


def test_the_checkpoint_records_what_built_it(tmp_path: Path) -> None:
    """A checkpoint that does not say what built it can be loaded into the wrong model.

    Anchor sizes are not weights — `AnchorGenerator` holds no parameters, and min_size/max_size
    are attributes of the transform rather than tensors — so a state dict fitted under one set
    loads without complaint under another and then looks for ships of the wrong size, quietly.
    The build block is what lets the side that loads it refuse.
    """
    run = a_run(tmp_path, epochs=1)
    train(**run)

    _, path = Checkpoints(tmp_path / "run").latest()
    state = torch.load(path, map_location="cpu", weights_only=True)

    assert state["built"] == run["built"]
    assert state["built"]["anchor_sizes"] == ANCHOR_SIZES


def test_the_ladder_has_the_four_rungs_the_plan_names() -> None:
    """First, because the three tests below are parametrised over this directory and would all
    pass vacuously on an empty one — which is exactly the state the repository is in before this
    task, and exactly the way a missing rung would go unnoticed after it."""
    assert sorted(path.name for path in LADDER.glob("*.yaml")) == [
        "r1-cosine.yaml",
        "r2-anchors.yaml",
        "r3-stem.yaml",
        "r4-sampler.yaml",
    ]


def test_rung_one_is_the_one_line_it_claims_to_be() -> None:
    """r1's entire reason to exist is `schedule.lr_schedule: cosine`. The parametrised test below
    only checks that *some* schedule with a positive epoch count came out of *some* rung; this
    one names the rung and the value, so a `lr_schedule` that silently fell back to `Schedule`'s
    own default ("constant") is caught against the rung whose whole claim that would break."""
    request = training_request_from(load_config(LADDER / "r1-cosine.yaml"), LADDER)

    # `.get` rather than bare indexing — see the shipped-config parse test for why a deleted key
    # should fail this assertion by name rather than as a `KeyError`.
    assert request["schedule"].get("lr_schedule") == "cosine"


@pytest.mark.parametrize("rung", sorted(LADDER.glob("*.yaml")), ids=lambda path: path.name)
def test_every_rung_of_the_ladder_is_a_training_config_the_command_parses(rung: Path) -> None:
    """Each of these is run on a machine rented by the hour, days apart. A mistyped key in the
    fourth would surface after three evenings had already been spent."""
    request = training_request_from(load_config(rung), rung.parent)

    assert request["schedule"]["epochs"] > 0
    assert request["model"]["stem"] in {"repeat", "single"}


@pytest.mark.parametrize("rung", sorted(LADDER.glob("*.yaml")), ids=lambda path: path.name)
def test_every_rung_writes_its_checkpoints_and_its_metrics_somewhere_of_its_own(
    rung: Path,
) -> None:
    """The trap this closes is quiet and expensive. Rungs share a working directory on Kaggle, and
    a rung that inherited the previous one's checkpoint directory would find a finished schedule
    there and do nothing at all — reporting the previous rung's numbers as its own."""
    request = training_request_from(load_config(rung), rung.parent)
    baseline = training_request_from(load_config(CONFIG), CONFIG.parent)

    assert request["checkpoints"].directory != baseline["checkpoints"].directory
    assert request["journal"].path != baseline["journal"].path


def test_the_rungs_of_the_ladder_do_not_share_a_working_directory() -> None:
    requests = [
        training_request_from(load_config(rung), rung.parent) for rung in LADDER.glob("*.yaml")
    ]

    directories = [request["checkpoints"].directory for request in requests]
    metrics = [request["journal"].path for request in requests]

    assert len(set(directories)) == len(directories)
    assert len(set(metrics)) == len(metrics)


def test_train_hands_its_own_stem_to_the_loop_not_only_to_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`stem` has three landings, and this is the one nothing else here reaches: it feeds
    `request["model"]` — which goes into both `detector_model(**request["model"])` and the
    `built` block written into every checkpoint — and it is also `train`'s own `stem=` keyword,
    passed separately because `train` hands it to the `_Tiles` dataset that turns a tile into a
    tensor. Miss that third wiring and the model is built for one channel while the dataset feeds
    it three — loud, but loud on a rented GPU, which is the cost this test exists to avoid
    paying.

    Cataloguing tiles, splitting by scene, and building the model are all stubbed out, because
    none of them is what this test checks — only the one keyword argument the wiring under test
    controls, captured off the stub in its place. `r3-stem.yaml` is the fixture rather than the
    shipped config because it resolves to `model.stem == "single"`, and `single` differs from
    `train`'s own default of `"repeat"`; a config that resolved to the default would pass whether
    or not the wiring existed.
    """
    captured: dict = {}

    def fake_train(**kwargs: object) -> None:
        captured.update(kwargs)

    # `_train` imports `detector_model` and `train` inside its own function body, so each call
    # re-resolves the name from its defining module — these two have to be patched there, not on
    # `darkvessel.cli`, where no such names exist and a patch would silently do nothing.
    monkeypatch.setattr("darkvessel.detect.train.train", fake_train)
    monkeypatch.setattr("darkvessel.detect.model.detector_model", lambda **kwargs: None)
    # `catalogue` and `split_by_scene` are imported at `cli.py` module scope, so `_train` resolves
    # them as ordinary globals there — the opposite rule from the two above.
    monkeypatch.setattr("darkvessel.cli.catalogue", lambda root, layout: [])
    monkeypatch.setattr("darkvessel.cli.split_by_scene", lambda refs: ([], []))

    _train(LADDER / "r3-stem.yaml")

    assert captured.get("stem", "repeat") == "single"
