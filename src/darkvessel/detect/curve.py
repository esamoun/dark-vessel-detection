"""One run's precision-recall curve, and how far each point on it wanders.

A detector has a precision at a confidence, not a precision, so the honest report of this one is
the whole trade-off rather than the operating point the chain happens to sit at. That is the
curve. What this module adds to it is the second number beside every point: the range that point
covered over the run's last epochs, which differ from one another by nothing except where the run
was stopped.

The band is not decoration. The run of 2026-08-14 moved precision at a fixed threshold from 0.41
to 0.84 between adjacent epochs while its loss sat still, and a curve drawn from one epoch of
that would have published four significant figures of a coin toss. The rule that judges the
ladder is built on the same measurement — see `ladder.py`, which takes its F1 and its window from
here — and this module draws it rather than deciding anything with it.

No torch and no file: arithmetic over the dicts a journal already holds, so the numbers the
README prints are checked on a laptop in a second.
"""

import math
from dataclasses import dataclass

# How many of a run's last epochs a band is measured over. Four is what a twelve-epoch schedule
# affords while still being past the point where a decaying learning rate has settled. Defined
# here and imported by `ladder.py`, because a band the curve draws and a band the keep/reject
# rule applies that were measured over different windows would be two different numbers under one
# name.
WINDOW = 4


@dataclass(frozen=True)
class Point:
    """One operating point: what the detector bought at one confidence, and how firm that is."""

    threshold: float
    precision: float
    recall: float
    found: int
    false_alarms: int
    missed: int
    # Lowest and highest each figure reached over the last `window` epochs. The interval rather
    # than its width, and the point is not its centre: the final epoch is wherever the wander
    # happened to leave it, and a symmetric bar drawn around it would place the run's real range
    # somewhere it never went. Degenerate on a run of one epoch, which says nothing is known
    # about the wander rather than that there is none.
    precision_range: tuple[float, float]
    recall_range: tuple[float, float]

    @property
    def f1(self) -> float:
        return f1(self.precision, self.recall)

    @property
    def precision_band(self) -> float:
        """The width of the interval — the same measurement the ladder's keep/reject rule uses."""
        return self.precision_range[1] - self.precision_range[0]

    @property
    def recall_band(self) -> float:
        return self.recall_range[1] - self.recall_range[0]


def f1(precision: float | None, recall: float | None) -> float:
    """Zero where a threshold reported nothing.

    `Counts.precision` is NaN when nothing was reported, on the argument that a run which
    returned nothing was neither right nor wrong. That survives into JSON, and a NaN loose in a
    `max` would make an empty detector the best rung on the ladder.
    """
    if precision is None or recall is None:
        return 0.0
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def curve(epochs: list[dict], window: int = WINDOW) -> list[Point]:
    """The final epoch's curve, banded by the epochs before it.

    The final epoch and not the best one, because the final epoch is the checkpoint the chain
    loads: a curve assembled from whichever epoch scored best at each threshold would describe a
    detector that was never saved.
    """
    if not epochs:
        raise ValueError("no epoch has been scored, so there is no curve to draw")

    recent = epochs[-window:]
    _check_aligned(recent)

    points = []
    for index, reported in enumerate(epochs[-1]["at"]):
        precisions = [epoch["at"][index]["precision"] for epoch in recent]
        recalls = [epoch["at"][index]["recall"] for epoch in recent]
        points.append(
            Point(
                threshold=float(reported["score"]),
                precision=float(reported["precision"]),
                recall=float(reported["recall"]),
                found=int(reported["found"]),
                false_alarms=int(reported["false"]),
                missed=int(reported["missed"]),
                precision_range=_range(precisions, float(reported["precision"])),
                recall_range=_range(recalls, float(reported["recall"])),
            )
        )

    return sorted(points, key=lambda point: point.threshold)


def table(points: list[Point]) -> str:
    """The curve as a markdown table, for `docs/` and the README.

    The band is printed beside the figure it belongs to rather than in a column of its own, so
    that a number cannot be quoted out of this table without the range it moved over coming with
    it.
    """
    lines = [
        "| Score threshold | Precision | Recall | F1 | Found | False | Missed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in points:
        lines.append(
            f"| {point.threshold:.2f} "
            f"| {_figure(point.precision)} {_interval(point.precision_range)} "
            f"| {_figure(point.recall)} {_interval(point.recall_range)} "
            f"| {point.f1:.3f} "
            f"| {point.found} | {point.false_alarms} | {point.missed} |"
        )
    return "\n".join(lines)


# The plot, in the one size it is ever drawn at. Numbers rather than a layout engine: this draws
# one figure of six points, and a dependency that draws any figure at all would be the largest
# thing in the install for the smallest thing in the repository.
_WIDTH, _HEIGHT = 720, 460
_LEFT, _RIGHT, _TOP, _BOTTOM = 70, 30, 30, 60
_PLOT_W = _WIDTH - _LEFT - _RIGHT
_PLOT_H = _HEIGHT - _TOP - _BOTTOM

# Mid grey for the frame and strong blue for the curve, both legible on a white README and on a
# dark one. The background is left transparent for the same reason: a white rectangle here is a
# white rectangle in dark mode.
_INK = "#8b949e"
_CURVE = "#1f6feb"


def svg(points: list[Point]) -> str:
    """The curve as a standalone SVG, with a cross at each point for the band it moved over.

    Recall runs left to right and precision bottom to top, so the detector everyone wants sits in
    the top right corner. Stated in a test as well as here, because an axis drawn the other way up
    produces a figure that reads as its own opposite and looks entirely correct.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'width="{_WIDTH}" height="{_HEIGHT}" font-family="system-ui, sans-serif">',
        f'<rect x="{_LEFT}" y="{_TOP}" width="{_PLOT_W}" height="{_PLOT_H}" fill="none" '
        f'stroke="{_INK}" stroke-width="1"/>',
    ]

    for tick in range(0, 11, 2):
        value = tick / 10
        x, y = _x(value), _y(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_TOP}" x2="{x:.1f}" y2="{_TOP + _PLOT_H}" '
            f'stroke="{_INK}" stroke-width="0.5" stroke-dasharray="2 4"/>'
        )
        parts.append(
            f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_LEFT + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="{_INK}" stroke-width="0.5" stroke-dasharray="2 4"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{_TOP + _PLOT_H + 20}" fill="{_INK}" font-size="12" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
        parts.append(
            f'<text x="{_LEFT - 10}" y="{y + 4:.1f}" fill="{_INK}" font-size="12" '
            f'text-anchor="end">{value:.1f}</text>'
        )

    ordered = sorted(points, key=lambda point: _finite(point.recall))
    if ordered:
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{_x(_finite(point.recall)):.1f},"
            f"{_y(_finite(point.precision)):.1f}"
            for index, point in enumerate(ordered)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{_CURVE}" stroke-width="2"/>')

    for point in points:
        x, y = _x(_finite(point.recall)), _y(_finite(point.precision))
        low_recall, high_recall = point.recall_range
        low_precision, high_precision = point.precision_range
        parts.append(
            f'<line x1="{_x(_finite(low_recall)):.1f}" y1="{y:.1f}" '
            f'x2="{_x(_finite(high_recall)):.1f}" y2="{y:.1f}" '
            f'stroke="{_CURVE}" stroke-width="1" opacity="0.6"/>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{_y(_finite(high_precision)):.1f}" '
            f'x2="{x:.1f}" y2="{_y(_finite(low_precision)):.1f}" '
            f'stroke="{_CURVE}" stroke-width="1" opacity="0.6"/>'
        )
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{_CURVE}"/>')
        # A point near the right edge is labelled on its left, because a curve like this one puts
        # its most interesting thresholds against the frame and a label that leaves the canvas is
        # a label that is not there.
        rightish = x > _LEFT + 0.75 * _PLOT_W
        parts.append(
            f'<text x="{x - 9 if rightish else x + 9:.1f}" y="{y - 8:.1f}" fill="{_CURVE}" '
            f'font-size="12" text-anchor="{"end" if rightish else "start"}">'
            f"{point.threshold:.2f}</text>"
        )

    parts.append(
        f'<text x="{_LEFT + _PLOT_W / 2:.1f}" y="{_HEIGHT - 16}" fill="{_INK}" font-size="13" '
        f'text-anchor="middle">Recall</text>'
    )
    parts.append(
        f'<text x="20" y="{_TOP + _PLOT_H / 2:.1f}" fill="{_INK}" font-size="13" '
        f'text-anchor="middle" transform="rotate(-90 20 {_TOP + _PLOT_H / 2:.1f})">Precision</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _check_aligned(epochs: list[dict]) -> None:
    """Refuse a run whose epochs did not report the same thresholds.

    The band lines epochs up by position, because that is how a journal stores them. A run whose
    threshold set changed halfway through would subtract 0.9's precision from 0.5's and print the
    difference as noise, and every other number in the file would look right.
    """
    thresholds = [tuple(point["score"] for point in epoch["at"]) for epoch in epochs]
    if len(set(thresholds)) > 1:
        raise ValueError(
            f"these epochs were scored at different thresholds ({sorted(set(thresholds))}), "
            "so they cannot be banded against one another"
        )


def _range(values: list[float], reported: float) -> tuple[float, float]:
    """The lowest and highest of the epochs that reported a number at all.

    An epoch that reported nothing has no precision to be part of an interval; dropping it is the
    only reading that does not invent one. Where no epoch reported, the interval collapses onto
    the point itself and the band is zero — which is what "nothing is known about the wander"
    looks like in a figure.
    """
    finite = [value for value in values if value is not None and not math.isnan(value)]
    if not finite:
        return (reported, reported)
    return (min(finite), max(finite))


def _finite(value: float) -> float:
    """NaN as zero, and only for geometry.

    A threshold that reported nothing has no precision, and the ladder already scores that as an
    F1 of zero. Plotting it there keeps the figure and the rule saying the same thing; leaving the
    NaN in would emit `cx="nan"`, which renders as a marker that is simply not on the chart.
    """
    return 0.0 if value is None or math.isnan(value) else value


def _figure(value: float) -> str:
    return "—" if value is None or math.isnan(value) else f"{value:.3f}"


def _interval(bounds: tuple[float, float]) -> str:
    low, high = bounds
    return f"({_figure(low)}–{_figure(high)})"


def _x(recall: float) -> float:
    return _LEFT + max(0.0, min(1.0, recall)) * _PLOT_W


def _y(precision: float) -> float:
    return _TOP + (1.0 - max(0.0, min(1.0, precision))) * _PLOT_H
