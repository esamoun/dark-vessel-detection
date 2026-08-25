"""The precision-recall curve of one run, and the band around it.

Issue #12 asks for a curve on a held-out split. The curve is the easy part; what makes it worth
reading is the second number beside each point — how far that point moved over the run's last
epochs, which differ from one another by nothing but where the run was stopped. A curve drawn
from one epoch of a run that oscillates is a curve with an invisible error bar, and this project
has already published one number that way and had to widen it.

No torch, no journal on disk: everything here is arithmetic over the dicts a journal holds, and
it is the arithmetic that decides what the README claims.
"""

import math

import pytest

from darkvessel.detect.curve import WINDOW, Point, curve, f1, svg, table


def _epoch(points: list[tuple[float, float, float]], **counts: int) -> dict:
    """One epoch's entry: a threshold, a precision and a recall for each point."""
    return {
        "epoch": counts.pop("epoch", 1),
        "held_out_tiles": 3000,
        "held_out_ships": 2378,
        "at": [
            {
                "score": threshold,
                "precision": precision,
                "recall": recall,
                "found": counts.get("found", 0),
                "false": counts.get("false", 0),
                "missed": counts.get("missed", 0),
            }
            for threshold, precision, recall in points
        ],
    }


def test_f1_is_zero_where_a_threshold_reported_nothing() -> None:
    """`Counts.precision` is NaN when nothing was reported — neither right nor wrong — and a NaN
    loose in a comparison would make an empty detector the best point on the curve."""
    assert f1(float("nan"), 0.5) == 0.0
    assert f1(0.0, 0.0) == 0.0
    assert f1(None, None) == 0.0
    assert f1(0.5, 0.5) == pytest.approx(0.5)


def test_a_point_is_the_final_epoch_and_not_the_best_one() -> None:
    """The shipped detector is the last epoch, because that is the checkpoint the chain loads.
    A curve drawn from the best epoch of each threshold would describe a model nobody has."""
    epochs = [
        _epoch([(0.5, 0.99, 0.99)]),
        _epoch([(0.5, 0.60, 0.70)]),
    ]

    assert curve(epochs, window=2)[0].precision == 0.60


def test_the_band_is_how_far_the_point_moved_over_the_last_epochs() -> None:
    epochs = [
        _epoch([(0.5, 0.10, 0.10)]),  # outside the window, and must not widen it
        _epoch([(0.5, 0.80, 0.50)]),
        _epoch([(0.5, 0.90, 0.55)]),
        _epoch([(0.5, 0.85, 0.60)]),
    ]

    point = curve(epochs, window=3)[0]

    assert point.precision_band == pytest.approx(0.10)
    assert point.recall_band == pytest.approx(0.10)


def test_a_run_shorter_than_the_window_bands_what_it_has() -> None:
    """Every rung here ran twelve epochs, but a session killed at three still has a curve, and
    reporting it with no band at all would be the flattering reading."""
    epochs = [_epoch([(0.5, 0.80, 0.50)]), _epoch([(0.5, 0.90, 0.50)])]

    assert curve(epochs, window=WINDOW)[0].precision_band == pytest.approx(0.10)


def test_the_points_come_out_in_threshold_order() -> None:
    epochs = [_epoch([(0.9, 0.95, 0.3), (0.05, 0.10, 0.99), (0.5, 0.70, 0.80)])]

    assert [point.threshold for point in curve(epochs, window=1)] == [0.05, 0.5, 0.9]


def test_epochs_scored_at_different_thresholds_are_refused() -> None:
    """The band subtracts one epoch's number from another's, and the journal is a list of lists:
    line them up by position and a run whose threshold set changed halfway would band 0.9 against
    0.5 and report the difference as noise. Nothing else in this file would look wrong."""
    epochs = [_epoch([(0.5, 0.70, 0.80)]), _epoch([(0.9, 0.95, 0.30)])]

    with pytest.raises(ValueError, match="thresholds"):
        curve(epochs, window=2)


def test_an_empty_run_has_no_curve() -> None:
    with pytest.raises(ValueError, match="no epoch"):
        curve([], window=WINDOW)


def test_the_table_carries_every_point_and_its_band() -> None:
    rows = table(curve([_epoch([(0.5, 0.70, 0.80)])], window=1)).splitlines()

    assert len(rows) == 3  # header, rule, one point
    assert "0.50" in rows[-1]
    assert "0.747" in rows[-1]  # the F1 of 0.70 and 0.80, derived here and nowhere else


def test_the_plot_draws_one_marker_for_every_point() -> None:
    drawn = svg(curve([_epoch([(0.5, 0.70, 0.80), (0.9, 0.95, 0.30)])], window=1))

    assert drawn.startswith("<svg")
    assert drawn.count("<circle") == 2
    assert "viewBox" in drawn


def test_the_plot_places_recall_1_precision_1_in_the_corner_it_belongs_in() -> None:
    """A plot with a flipped axis is a plot that reads as its own opposite, and no test that
    counts elements would notice. The perfect detector goes top right; the useless one bottom
    left, which in SVG coordinates is the larger y."""
    perfect = curve([_epoch([(0.5, 1.0, 1.0)])], window=1)
    useless = curve([_epoch([(0.5, 0.0, 0.0)])], window=1)

    top_right = _marker(svg(perfect))
    bottom_left = _marker(svg(useless))

    assert top_right[0] > bottom_left[0]
    assert top_right[1] < bottom_left[1]


def _marker(drawn: str) -> tuple[float, float]:
    """The centre of the one circle in a plot of a single point."""
    circle = drawn.split("<circle")[1]
    x = float(circle.split('cx="')[1].split('"')[0])
    y = float(circle.split('cy="')[1].split('"')[0])
    return x, y


def test_a_point_that_reported_nothing_is_still_plottable() -> None:
    """NaN reaches the plot as a coordinate, and an SVG with `cx="nan"` renders as nothing at all
    — a missing marker rather than a visible failure."""
    drawn = svg(curve([_epoch([(0.5, float("nan"), 0.0)])], window=1))

    assert "nan" not in drawn.lower()
    assert isinstance(Point, type)
    assert not math.isnan(_marker(drawn)[0])


def test_the_bar_is_the_range_the_run_covered_and_not_a_width_centred_on_the_point() -> None:
    """The final epoch is wherever the wander left it, which is often an end of the range rather
    than its middle. A bar of the right width centred on the point would put the run's real
    interval somewhere it never went — here it would claim precision reached 0.60, which no epoch
    did — and it would look exactly as convincing.
    """
    epochs = [
        _epoch([(0.5, 0.80, 0.50)]),
        _epoch([(0.5, 0.90, 0.50)]),
        _epoch([(0.5, 0.70, 0.50)]),
    ]

    point = curve(epochs, window=3)[0]
    assert point.precision_range == pytest.approx((0.70, 0.90))
    assert point.precision_band == pytest.approx(0.20)

    drawn = svg([point])
    _, marker_y = _marker(drawn)
    top, bottom = _vertical_bar(drawn)

    assert bottom == pytest.approx(marker_y)  # the point is the bottom of its own range
    assert top < marker_y


def _vertical_bar(drawn: str) -> tuple[float, float]:
    """The y ends of the one vertical bar in a plot of a single point."""
    for line in drawn.split("<line")[1:]:
        x1 = float(line.split('x1="')[1].split('"')[0])
        x2 = float(line.split('x2="')[1].split('"')[0])
        y1 = float(line.split('y1="')[1].split('"')[0])
        y2 = float(line.split('y2="')[1].split('"')[0])
        # Vertical, and not the degenerate horizontal bar of a recall that never moved.
        if x1 == x2 and y1 != y2 and "dasharray" not in line:
            return y1, y2
    raise AssertionError("no vertical bar was drawn")


def test_no_label_is_drawn_off_the_canvas() -> None:
    """This curve puts its highest-precision thresholds hard against the right frame, so a label
    always drawn to the right of its marker runs off the figure — where it is not a wrong number,
    it is no number at all."""
    drawn = svg(curve([_epoch([(0.9, 0.95, 0.999), (0.05, 0.08, 1.0)])], window=1))

    for text in drawn.split("<text")[1:]:
        x = float(text.split('x="')[1].split('"')[0])
        anchor = "end" if 'text-anchor="end"' in text else "start"
        width = 26 if anchor == "start" else -26
        assert 0 <= x + width <= 720
