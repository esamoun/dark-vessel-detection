"""Comparing five runs that differ by one line each.

Nothing here asserts a precision, a recall or an F1 that a model produced. The journals below are
written by hand precisely so that the arithmetic and the refusals can be pinned without pinning a
measurement — the numbers a real run reports are the output of the level, and a test that fixed
one would turn a measurement into a target.

What is pinned is the boundary. A rung whose gain exactly equals the noise band is rejected, and
that single `>` rather than `>=` is the difference between a ladder and a narration of noise.
"""

from pathlib import Path

import pytest

from darkvessel.cli import ladder_request_from
from darkvessel.config import load_config
from darkvessel.detect.ladder import Rung, band, best_f1, judge, table

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

REPORTING = {"tolerance_m": 200.0, "resolution_m": 10.0, "thresholds": [0.5, 0.75]}

# Every F1 in this file is a dyadic rational — a half, a quarter, an eighth — and that is not
# decoration. The boundary case below turns on a gain being *exactly* equal to a band, and a
# statistic of 0.52 would come back from the harmonic mean a fraction of an ulp away from 0.52.
# The test would then pass or fail on the rounding of a number nobody chose.


def an_epoch(epoch: int, precision: float, recall: float) -> dict:
    """One journal entry, at two thresholds, the second of which is the better one."""
    return {
        "epoch": epoch,
        "training_loss": 0.15,
        "learning_rate": 0.005,
        "held_out_tiles": 3000,
        "held_out_ships": 2378,
        "at": [
            {"score": 0.5, "precision": 0.1, "recall": 0.1, "found": 1, "false": 1, "missed": 1},
            {
                "score": 0.75,
                "precision": precision,
                "recall": recall,
                "found": 1,
                "false": 1,
                "missed": 1,
            },
        ],
    }


def a_rung(label: str, f1s: list[float], reporting: dict | None = None) -> Rung:
    """A rung whose final epochs have the F1s given. Precision and recall are set equal, so the
    harmonic mean of the pair is the number asked for."""
    return Rung(
        label=label,
        changed="one line",
        run={"reporting": reporting or REPORTING},
        epochs=[an_epoch(index + 1, f1, f1) for index, f1 in enumerate(f1s)],
    )


def test_the_statistic_is_the_best_f1_across_the_reported_thresholds() -> None:
    entry = an_epoch(1, precision=0.9, recall=0.7)

    assert best_f1(entry) == pytest.approx(2 * 0.9 * 0.7 / (0.9 + 0.7))


def test_a_threshold_that_reported_nothing_scores_zero_rather_than_perfectly() -> None:
    """`Counts.precision` is NaN when nothing was reported, on the argument that a run which
    returned nothing was neither right nor wrong. Carried into JSON, NaN must not sort to the top
    of a maximum."""
    entry = an_epoch(1, precision=float("nan"), recall=float("nan"))
    entry["at"][0] = {"score": 0.5, "precision": 0.6, "recall": 0.6}

    assert best_f1(entry) == pytest.approx(0.6)


def test_the_band_is_the_range_of_the_statistic_over_the_last_four_epochs() -> None:
    epochs = [
        an_epoch(index + 1, f1, f1) for index, f1 in enumerate([0.25, 0.875, 0.5, 0.625, 0.75])
    ]

    # The first epoch is outside the window; 0.875 - 0.5 is the range of the last four.
    assert band(epochs) == pytest.approx(0.375)


def test_the_first_rung_is_kept_because_there_is_nothing_to_compare_it_against() -> None:
    verdicts = judge([a_rung("R0", [0.5, 0.5, 0.5, 0.5])])

    assert verdicts[0].kept is True
    assert verdicts[0].against is None
    assert verdicts[0].band is None


def test_a_rung_whose_gain_exactly_equals_the_band_is_rejected() -> None:
    """The boundary, pinned. A gain that only just reaches the noise the previous rung was
    already showing is noise, and `>=` here would let every rung through on a bad week."""
    previous = a_rung("R0", [0.5, 0.75, 0.5, 0.5])  # band 0.25, final 0.5
    later = a_rung("R1", [0.75, 0.75, 0.75, 0.75])  # final 0.75, gain exactly 0.25

    verdicts = judge([previous, later])

    assert verdicts[1].kept is False


def test_a_rung_that_clears_the_band_by_a_margin_is_kept() -> None:
    previous = a_rung("R0", [0.5, 0.75, 0.5, 0.5])  # band 0.25, final 0.5
    later = a_rung("R1", [0.875, 0.875, 0.875, 0.875])  # final 0.875, gain 0.375

    verdicts = judge([previous, later])

    assert verdicts[1].kept is True
    assert verdicts[1].gain == pytest.approx(0.375)


def test_a_rejected_rung_is_not_what_the_next_one_is_measured_against() -> None:
    """The ladder is greedy: the next rung starts from the last rung that was kept, so a rejected
    rung must not become the baseline for the one after it."""
    r0 = a_rung("R0", [0.5, 0.75, 0.5, 0.5])  # band 0.25, final 0.5
    r1 = a_rung("R1", [0.25, 0.25, 0.25, 0.25])  # worse; rejected
    r2 = a_rung("R2", [0.875, 0.875, 0.875, 0.875])  # measured against R0, not R1

    verdicts = judge([r0, r1, r2])

    assert verdicts[1].kept is False
    assert verdicts[2].against == "R0"
    assert verdicts[2].gain == pytest.approx(0.375)


def test_two_rungs_scored_at_different_tolerances_are_refused() -> None:
    generous = a_rung("R0", [0.5, 0.5, 0.5, 0.5])
    strict = a_rung("R1", [0.6, 0.6, 0.6, 0.6], reporting={**REPORTING, "tolerance_m": 300.0})

    with pytest.raises(ValueError, match="tolerance_m"):
        judge([generous, strict])


def test_two_rungs_scored_at_different_thresholds_are_refused() -> None:
    r0 = a_rung("R0", [0.5, 0.5, 0.5, 0.5])
    r1 = a_rung("R1", [0.6, 0.6, 0.6, 0.6], reporting={**REPORTING, "thresholds": [0.9]})

    with pytest.raises(ValueError, match="thresholds"):
        judge([r0, r1])


def test_two_rungs_scored_over_different_held_out_splits_are_refused() -> None:
    """The one thing every rung of this ladder has in common is the split, and a rung scored over
    a subset of it would report a precision the detector had not earned."""
    r0 = a_rung("R0", [0.5, 0.5, 0.5, 0.5])
    r1 = a_rung("R1", [0.6, 0.6, 0.6, 0.6])
    for entry in r1.epochs:
        entry["held_out_tiles"] = 1500

    with pytest.raises(ValueError, match="held_out_tiles"):
        judge([r0, r1])


def test_a_rung_that_does_not_name_its_run_is_refused() -> None:
    """A bare-list metrics file from before the run block existed. It reads back for a resume and
    it does not serve as a rung, because nothing in it says how it was scored."""
    unnamed = Rung(label="R0", changed="nothing", run=None, epochs=[an_epoch(1, 0.5, 0.5)])

    with pytest.raises(ValueError, match="does not name"):
        judge([unnamed])


def test_the_table_names_every_rung_and_says_which_were_kept() -> None:
    verdicts = judge([a_rung("R0", [0.5, 0.75, 0.5, 0.5]), a_rung("R1", [0.25, 0.25, 0.25, 0.25])])

    rendered = table(verdicts)

    assert "R0" in rendered and "R1" in rendered
    assert "kept" in rendered and "rejected" in rendered


def test_the_shipped_ladder_config_is_the_one_the_command_parses() -> None:
    """The same gap `training_request_from` exists to close. This file names five paths that will
    not all exist until five Kaggle sessions have run, and a mistyped key in it would surface only
    to whoever came back with the last of them."""
    rungs = ladder_request_from(load_config(CONFIGS / "ladder.yaml"), CONFIGS)

    assert [rung["label"] for rung in rungs] == ["R0", "R1", "R2", "R3", "R4"]
    assert all(rung["changed"] for rung in rungs)
    assert all(rung["metrics"].is_absolute() for rung in rungs)


def test_every_rung_of_the_shipped_ladder_reads_its_metrics_from_its_own_file() -> None:
    """Two rungs pointing at one file would compare a run against itself and report a gain of
    zero, which reads as a rejection and would be recorded as one."""
    rungs = ladder_request_from(load_config(CONFIGS / "ladder.yaml"), CONFIGS)

    paths = [rung["metrics"] for rung in rungs]

    assert len(set(paths)) == len(paths)
