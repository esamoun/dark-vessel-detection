"""Reading a config, and the config it stands on.

The ladder in issue #11 is five runs that differ by one line each. A rung that quietly differs by
two lines measures two things and reports one, so the mechanism that lets a rung state only its
own change is load-bearing rather than a convenience — and its failure modes are a cycle, a base
that is not there, and a merge that drops a key.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from darkvessel.cli import _read_ladder
from darkvessel.config import load_config
from darkvessel.detect.checkpoints import Journal
from darkvessel.detect.ladder import WINDOW, Rung, Verdict, judge

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LADDER = CONFIGS / "ladder"

# The precision the swap of 2026-08-16 fixed the chain's operating point at: 0.941 on the
# held-out split, from the weights of 2026-08-14 at a score threshold of 0.75. Rounded down to
# two figures because it is a decision about what this chain is for — a detection it fails to
# match becomes a dark vessel and possibly an accusation — and not a number to be matched to
# the third decimal by whatever weights happen to be loaded.
DECIDED_PRECISION = 0.94

# The one key each shipped rung's own comment says it changes, dotted so it can be read off a
# flattened config. Hand-written rather than derived, because the whole point is to catch a rung
# file that silently changed something this list does not expect.
RUNG_OWN_CHANGE = {
    "r1-cosine.yaml": "schedule.lr_schedule",
    "r2-anchors.yaml": "model.anchor_sizes",
    "r3-stem.yaml": "model.stem",
    "r4-sampler.yaml": "model.rpn_batch_size_per_image",
    "r5-fg-iou.yaml": "model.rpn_fg_iou_thresh",
}


def test_a_config_that_extends_nothing_is_read_as_it_stands(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("schedule:\n  epochs: 12\n  learning_rate: 0.005\n")

    assert load_config(path) == {"schedule": {"epochs": 12, "learning_rate": 0.005}}


def test_a_rung_states_only_what_it_changes(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "schedule:\n  epochs: 12\n  learning_rate: 0.005\nmodel:\n  stem: repeat\n"
    )
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nschedule:\n  lr_schedule: cosine\n")

    assert load_config(rung) == {
        "schedule": {"epochs": 12, "learning_rate": 0.005, "lr_schedule": "cosine"},
        "model": {"stem": "repeat"},
    }


def test_a_rung_overrides_a_value_the_base_already_set(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  anchor_sizes: [[32], [64]]\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nmodel:\n  anchor_sizes: [[4], [8]]\n")

    assert load_config(rung)["model"]["anchor_sizes"] == [[4], [8]]


def test_a_list_is_replaced_and_never_merged(tmp_path: Path) -> None:
    """A merged list would make `anchor_sizes: [[4], [8]]` mean nine levels rather than two, and
    the run would train on something no file states."""
    (tmp_path / "base.yaml").write_text("reporting:\n  thresholds: [0.05, 0.5, 0.9]\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nreporting:\n  thresholds: [0.75]\n")

    assert load_config(rung)["reporting"]["thresholds"] == [0.75]


def test_a_chain_of_rungs_resolves_through_to_the_base(tmp_path: Path) -> None:
    """The ladder is greedy: rung 2 extends rung 1, which extends the baseline."""
    (tmp_path / "base.yaml").write_text("schedule:\n  epochs: 12\n")
    (tmp_path / "r1.yaml").write_text("extends: base.yaml\nschedule:\n  lr_schedule: cosine\n")
    r2 = tmp_path / "r2.yaml"
    r2.write_text("extends: r1.yaml\nmodel:\n  anchor_sizes: [[4]]\n")

    assert load_config(r2) == {
        "schedule": {"epochs": 12, "lr_schedule": "cosine"},
        "model": {"anchor_sizes": [[4]]},
    }


def test_the_base_is_found_beside_the_file_that_names_it(tmp_path: Path) -> None:
    """Every path in this project is read relative to the config that declares it, and the base a
    rung extends is a path like any other. Rung configs live one directory down from the base."""
    (tmp_path / "train.yaml").write_text("schedule:\n  epochs: 12\n")
    (tmp_path / "ladder").mkdir()
    rung = tmp_path / "ladder" / "r1.yaml"
    rung.write_text("extends: ../train.yaml\nschedule:\n  lr_schedule: cosine\n")

    assert load_config(rung)["schedule"] == {"epochs": 12, "lr_schedule": "cosine"}


def test_a_config_that_extends_itself_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text("extends: loop.yaml\nschedule:\n  epochs: 12\n")

    with pytest.raises(ValueError, match="extends itself"):
        load_config(path)


def test_a_cycle_between_two_configs_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")

    with pytest.raises(ValueError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


def test_a_base_that_is_not_there_is_refused_by_name(tmp_path: Path) -> None:
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: ../nowhere/train.yaml\n")

    with pytest.raises(FileNotFoundError, match="train.yaml"):
        load_config(rung)


def test_the_extends_key_does_not_survive_into_the_config(tmp_path: Path) -> None:
    """Whatever reads the result should not have to know the file was assembled from two."""
    (tmp_path / "base.yaml").write_text("schedule:\n  epochs: 12\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\n")

    assert "extends" not in load_config(rung)


def _flatten(config: dict, prefix: str = "") -> dict[str, object]:
    """Every leaf of a nested config, by its dotted path, so two configs can be compared key by
    key rather than dict by dict — a changed `model.stem` should not read as "the whole `model`
    block differs"."""
    flat: dict[str, object] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _differing_keys(before: dict, after: dict) -> set[str]:
    before_flat, after_flat = _flatten(before), _flatten(after)
    return {
        key
        for key in set(before_flat) | set(after_flat)
        if before_flat.get(key) != after_flat.get(key)
    }


@pytest.mark.parametrize("rung", sorted(RUNG_OWN_CHANGE), ids=lambda name: name)
def test_a_rung_of_the_small_target_ladder_changes_exactly_the_one_thing_it_declares(
    rung: str,
) -> None:
    """The premise the whole ladder in issue #11 rests on, and until now it was enforced by
    nothing but the shape of four files and a human's eye — `config.py`'s own docstring and this
    file's both state it, and neither is a check. A rung that quietly changed a second key would
    measure two things and report one: rung 3 would no longer be "the stem, against R2" but "the
    stem and whatever else drifted in, against R2", and the gain the ladder posts would belong to
    neither change alone.

    Every rung also moves `out.checkpoints` and `out.metrics` — that is what keeps five runs from
    writing into one working directory, and it is excluded from "the one thing" for that reason,
    not because it is exempt from being checked at all: `test_every_rung_writes_its_checkpoints_
    and_its_metrics_somewhere_of_its_own` in `test_training_run.py` covers it separately.
    """
    path = LADDER / rung
    extends = yaml.safe_load(path.read_text())["extends"]
    base = (path.parent / extends).resolve()

    changed = _differing_keys(load_config(base), load_config(path))

    assert changed == {RUNG_OWN_CHANGE[rung], "out.checkpoints", "out.metrics"}


@pytest.mark.parametrize("rung", sorted(RUNG_OWN_CHANGE), ids=lambda name: name)
def test_every_rung_resolves_to_the_cosine_schedule_r1_was_kept_for(rung: str) -> None:
    """R1 was kept on 2026-08-23 — 0.836 against a bar of 0.833 — so cosine decay is part of the
    baseline every rung above it stands on, not an option any of them may quietly drop.

    The test above this one does not hold that, and cannot: it compares a rung to *whatever its
    own `extends` names*, so a rung repointed at `../train.yaml` still differs from its base by
    exactly one key and passes. That revert was made before this test was written and all 301
    tests passed. What it would cost is the ladder's whole premise — R2 would gain the anchors
    and lose the decay while reporting one change, and its band would widen back towards R0's
    0.026, loosening the bar R3 has to clear rather than tightening it.

    Repointing an `extends` is a documented and expected edit: it is what the runbook prescribes
    when a rung is *rejected*. This says only where it may point — at a rung that was kept, which
    on this ladder means at or above R1.
    """
    assert load_config(LADDER / rung)["schedule"].get("lr_schedule") == "cosine"


def _the_ladder() -> tuple[dict[str, Rung], list[Verdict]]:
    """The shipped ladder, read the way `darkvessel compare` reads it, and its verdicts.

    Read rather than hard-coded: these two tests are about the chain agreeing with the ladder, and
    a copy of the ladder's answer written into a test would agree with itself.

    Through `cli._read_ladder` rather than through a loop of this file's own, which is the second
    half of that argument and was not honoured until R5 was added to the ladder unrun. This helper
    used to build a `Rung` for every rung the config names, including one whose metrics file does
    not exist yet — `judge` then raised "R5 does not name the run that produced it" and two tests
    about the *chain* failed for a reason that had nothing to do with the chain. `compare` stops at
    the first rung that has not been run, because a ladder read across a gap measures a change
    against the wrong configuration; a helper claiming to read it the same way has to do the same
    thing, and the only way to be sure of that is to call it.
    """
    path = CONFIGS / "ladder.yaml"
    config = load_config(path)

    rungs = {rung.label: rung for rung in _read_ladder(config, path.parent)}

    window = int(config["ladder"].get("window", WINDOW))
    return rungs, judge(list(rungs.values()), window=window)


def test_the_chain_runs_the_weights_of_the_rung_the_ladder_kept() -> None:
    """The chain's checkpoint is the last rung the rule kept, and nothing else.

    This is the one thing in the ladder that measured nobody. Five rungs ran, R1 was kept, and
    `configs/kattegat-lane.yaml` went on naming the weights of 2026-08-14 — F1 0.807 against
    R1's 0.836 — for two days. Every test passed throughout, because a checkpoint path is not a
    number anything compares: the ladder proves which weights are best and the chain is free to
    load any others.

    Written against the verdict rather than against the string "R1", so that the next rung to be
    kept fails this test until the chain is repointed at it. That failure is the whole point.
    """
    rungs, verdicts = _the_ladder()
    kept = [verdict for verdict in verdicts if verdict.kept][-1]

    checkpoint = Path(load_config(CONFIGS / "kattegat-lane.yaml")["run"]["trained"]["checkpoint"])

    assert checkpoint.name.startswith(f"{kept.label.lower()}-"), (
        f"the ladder keeps {kept.label} and the chain loads {checkpoint.name}"
    )


# The README's ladder table, as `_readme_ladder` parses it back out. Unicode minus, because that
# is what the prose uses and `ladder.table` writes ASCII.
_MINUS = "\u2212"


def _readme_ladder() -> list[tuple[str, ...]]:
    """The rows of the ladder table in README.md, as tuples of their cells.

    Every row whose first cell is a rung label. Matched on that rather than on a heading, so the
    section can be retitled or moved without this test quietly matching nothing — which is the
    failure mode a table-scraping test has, and the reason it asserts the row count separately.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    rows = []
    for line in readme.splitlines():
        cells = [cell.strip().replace(_MINUS, "-") for cell in line.split("|")]
        if len(cells) > 2 and re.fullmatch(r"R\d+", cells[1]):
            # `| a | b |` splits to ['', 'a', 'b', ''] — the empty ends are the pipes, not cells.
            rows.append(tuple(cells[1:-1]))
    return rows


def test_the_ladder_table_in_the_readme_is_the_one_the_command_prints() -> None:
    """The six rows the README publishes are read out of six journals by `darkvessel compare`, and
    until now nothing checked that the prose and the command agreed.

    They can part company without anything failing. A rung re-run and its journal recommitted — R1
    already has two executions, and the entry of 2026-08-26 records why — moves every number in
    that row, and the README would go on stating the old ones. So would a rung repointed at a
    different `metrics:` file, or a window changed in `configs/ladder.yaml`: `judge` would return
    different verdicts and the table in the prose would be a table of numbers nothing produced.

    Only the numeric cells and the verdict are compared. "What changed" is deliberately worded
    differently in the two places — the README says "`anchor_sizes` to `[[4], …]`" where the config
    says it without backticks, and R0's row is written for a reader rather than for a diff — and
    pinning prose to prose would make this a formatting test rather than a numbers one.
    """
    rungs, verdicts = _the_ladder()
    published = _readme_ladder()

    assert len(published) == len(verdicts), (
        f"the README publishes {len(published)} rungs and the ladder judges {len(verdicts)}"
    )

    for row, verdict in zip(published, verdicts, strict=True):
        label, _changed, statistic, against, band, gain, kept = row
        where = f"README row {label}"

        assert label == verdict.label, where
        assert statistic == f"{verdict.statistic:.3f}", where
        assert against == (verdict.against or "—"), where
        assert band == ("—" if verdict.band is None else f"{verdict.band:.3f}"), where
        assert gain == ("—" if verdict.gain is None else f"{verdict.gain:+.3f}"), where
        assert kept == ("kept" if verdict.kept else "rejected"), where

    # The claim the section leads with, and the one a seventh rung would silently falsify. R0 is
    # excluded by `against is None` rather than by name: it is the first rung, which `judge` keeps
    # by definition because there is nothing before it to beat, and it changed nothing. "One change
    # of six was kept" is a claim about the five that had something to be measured against.
    assert sum(verdict.kept for verdict in verdicts if verdict.against is not None) == 1, (
        "the README says one change of six was kept"
    )


def test_every_rejected_rpn_rung_cut_the_low_confidence_noise() -> None:
    """`docs/evaluation.md`, failure mode 5, claims one shape three times: every rung that changed
    the RPN's positive/negative bookkeeping — R2's anchor geometry, R4's sampler batch, R5's
    foreground IoU threshold — cut the false alarms at the most permissive threshold reported, and
    none of them moved the statistic, which is decided at 0.75.

    That sentence is now quoted in two documents and is the strongest general claim this project
    makes about its detector, so it is asserted from the journals rather than left as prose. It is
    also the sentence a seventh rung could silently falsify: a rung that raised the low-confidence
    count, or one that was *kept*, would make it false everywhere it is written without failing
    anything. The rungs are read off the ladder's own verdicts rather than named here, so a rung
    added to `configs/ladder.yaml` joins this check by existing.

    The same class of staleness this guards was found by hand on 2026-08-30: failure mode 1 of that
    report still called the threshold untested and pointed at issue #24 as an open question, three
    days after the rung had run and refuted it.
    """
    rungs, verdicts = _the_ladder()
    rejected = {verdict.label for verdict in verdicts if not verdict.kept}

    def false_alarms_at_the_floor(label: str) -> int:
        """The false-detection count at the lowest score threshold that rung reported."""
        entry = rungs[label].epochs[-1]
        return min(entry["at"], key=lambda point: point["score"])["false"]

    # R1 is the standing rung and the one the others are measured against. Its two executions
    # differ here — 27039 on the ladder's run, 25299 on the one the evaluation report describes —
    # so the comparison is made against whichever the ladder is reading, not against a literal.
    standing = false_alarms_at_the_floor("R1")

    rpn_rungs = {"R2", "R4", "R5"}
    assert rpn_rungs <= rejected, (
        f"the report says all three RPN rungs were rejected; the ladder keeps "
        f"{rpn_rungs - rejected}"
    )
    for label in sorted(rpn_rungs):
        assert false_alarms_at_the_floor(label) < standing, (
            f"{label} did not cut the low-confidence noise: "
            f"{false_alarms_at_the_floor(label)} against R1's {standing}"
        )


def test_the_chains_score_threshold_holds_the_precision_the_swap_was_decided_on() -> None:
    """A threshold does not survive a change of weights; the operating point is what does.

    0.75 buys 0.941 precision from the detector of 2026-08-14 and 0.851 from these weights,
    because cosine decay moves the calibration of the scores — the same property the baseline's
    oscillation was made of. Carrying the number across and calling it "unchanged" would have
    quietly moved this chain from one false alarm in seventeen to one in seven, in a commit about
    a checkpoint path.

    So the threshold is held to what it buys: a precision no worse than the swap was decided on.
    It must also be a threshold the run actually scored, because a precision this test cannot
    read is a precision nobody has.

    Read out of the journal the config *names*, not the ladder's. The two are siblings — same
    config, same seed, two executions — and the chain loads the weights of one of them. Checking
    against the other would be the same defect as loading the baseline's weights under R1's
    numbers, one generation smaller.
    """
    trained = load_config(CONFIGS / "kattegat-lane.yaml")["run"]["trained"]
    journal = Journal((CONFIGS / trained["metrics"]).resolve())
    threshold = float(trained["score_threshold"])

    reported = {round(float(point["score"]), 3): point for point in journal.entries()[-1]["at"]}

    assert round(threshold, 3) in reported, (
        f"the chain scores at {threshold}, which its own run never reported: {sorted(reported)}"
    )
    assert reported[round(threshold, 3)]["precision"] >= DECIDED_PRECISION


def test_the_chains_weights_and_the_ladders_verdict_come_from_the_same_config() -> None:
    """The chain may load a re-execution of the kept rung, but not a different rung.

    R1's session was lost before its checkpoint was saved, so the shipped weights come from a
    second execution of `configs/ladder/r1-cosine.yaml`. That is allowed and recorded. What is not
    is the build block quietly differing — anchors, tile size and stem leave no trace in a state
    dict, and `TrainedDetector` can only refuse a disagreement it is told about.
    """
    rungs, verdicts = _the_ladder()
    kept = [verdict for verdict in verdicts if verdict.kept][-1]

    trained = load_config(CONFIGS / "kattegat-lane.yaml")["run"]["trained"]
    shipped = Journal((CONFIGS / trained["metrics"]).resolve()).run()

    assert shipped["built"] == rungs[kept.label].run["built"]
    assert shipped["schedule"]["lr_schedule"] == rungs[kept.label].run["schedule"]["lr_schedule"]


EMBEDDING = CONFIGS / "embeddings.yaml"
# The two-box archive's run. The one-box run beside it — `embedding-kattegat.json` and
# `retrieval-kattegat.json` — is kept as the numbers issue #13 was closed on, and is deliberately
# not what the shipped config points at.
EMBEDDING_METRICS = Path(__file__).resolve().parents[1] / "docs" / "runs" / "embedding-archive.json"


def test_the_embedding_the_chain_loads_was_fitted_by_the_config_that_names_it() -> None:
    """The gap `test_the_chain_runs_the_weights_of_the_rung_the_ladder_kept` exists to close,
    at the level below it.

    An encoder is a file path, and a file path is a number nothing compares. Edit `crop_px` or
    `dim` in the config, leave the checkpoint alone, and every command still runs: the crops are
    cut at one size and described by a model fitted at another, the vectors come back, the
    neighbours are plausible. `ContrastiveEmbedder` reads the geometry off the checkpoint rather
    than out of the config precisely so that nothing downstream can disagree with it — which
    leaves the config free to say something else, and this is what stops it.

    Against the journal rather than the weights, because `*.pt` is not in the repository and this
    test has to run in CI with no torch and no checkpoint.
    """
    config = load_config(EMBEDDING)["embedding"]
    fitted = Journal(EMBEDDING_METRICS).run()

    assert fitted is not None, f"{EMBEDDING_METRICS.name} does not say what run wrote it"
    for key in ("crop_px", "margin_px", "dim"):
        assert fitted["built"][key] == config[key], (
            f"the config asks for {key}={config[key]} and the encoder it names was fitted at "
            f"{fitted['built'][key]}; re-run `darkvessel embed` or put the config back"
        )
    assert fitted["schedule"]["seed"] == config["schedule"]["seed"]
    assert fitted["speckle"]["looks"] == pytest.approx(config["speckle_looks"])


def test_the_recorded_retrieval_check_describes_the_encoder_that_is_shipped() -> None:
    """The numbers in the README come out of a file a command wrote, and that file has to be
    about this encoder — the run that produced it, at the epoch it stopped at."""
    record = json.loads((EMBEDDING_METRICS.parent / "retrieval-archive.json").read_text())
    config = load_config(EMBEDDING)["embedding"]
    epochs = Journal(EMBEDDING_METRICS).entries()

    assert record["encoder"] == Path(config["encoder"]).name
    assert record["dim"] == config["dim"]
    assert record["twin_recall"]["epoch"] == epochs[-1]["epoch"] == config["schedule"]["epochs"]
    # Above chance by a margin nobody could mistake for noise, and the baseline travels with it:
    # a check whose chance level is not recorded is a number with no scale.
    assert record["twin_recall"]["twin_recall"] > 10 * record["twin_recall"]["chance"]
    assert record["same_object"]["retrieved"] > 10 * record["same_object"]["chance"]


def test_the_archive_draws_on_water_the_chain_itself_never_looks_at() -> None:
    """The reason the archive names boxes rather than borrowing the run's area.

    The study area was chosen by measurement for its traffic and holds no fixed structure at all,
    so an archive drawn from it alone can never show a representation telling a turbine from a
    ship — whatever the representation does. Asserted rather than left to a comment, because a
    config edited back to one box would leave every test here green and issue #14 unanswerable.
    """
    config = load_config(EMBEDDING)
    boxes = config["archive"]["boxes"]
    run_area = config["area"]["bounds"]

    assert len(boxes) > 1, "an archive of one box cannot hold both halves of the problem"
    assert any(box != run_area for box in boxes.values()), (
        "every box in this archive is the chain's own study area, which has no turbines in it"
    )
