"""Five runs that differ by one line each, and what separates a result from a draw.

Issue #11 asks for three adaptations, each measured against the configuration before it. The
measurement is the hard part. The baseline oscillated by more than any of the three changes is
likely to be worth — precision at a fixed threshold went 0.55, 0.74, 0.75, 0.41, ... 0.28, 0.80
across twelve epochs, under two different draws of the initial weights — so comparing one final
number against another would describe the draw and not the change, and would describe it with
three decimal places.

The rule this module applies is therefore written down in `docs/decisions.md` before any of the
runs it judges, and it is mechanical rather than discretionary: a rung is kept only if it beats
the previous kept rung by more than the noise that rung was already showing. A threshold chosen
after seeing the numbers is not a threshold, it is a narration of them.

No torch. What can go wrong here is a comparison between two runs that were not scored the same
way, which would produce a published claim that is false, and that belongs on the side of the
seam a laptop tests in a second.
"""

import math
from dataclasses import dataclass
from typing import Any

# How many of a rung's last epochs the noise band is measured over. Four is what a twelve-epoch
# schedule affords while still being past the point where a decaying learning rate has settled.
WINDOW = 4

# What two rungs must agree on before their numbers may be put beside one another.
SAME_REPORTING = ("tolerance_m", "resolution_m", "thresholds")
SAME_SPLIT = ("held_out_tiles", "held_out_ships")


@dataclass(frozen=True)
class Rung:
    """One run of the ladder: what it changed, how it was scored, and what it reported."""

    label: str
    changed: str
    run: dict[str, Any] | None
    epochs: list[dict[str, Any]]


@dataclass(frozen=True)
class Verdict:
    """What the rule made of one rung. `against`, `band` and `gain` are None for the first."""

    label: str
    changed: str
    statistic: float
    against: str | None
    band: float | None
    gain: float | None
    kept: bool


def best_f1(entry: dict[str, Any]) -> float:
    """The best F1 this epoch reached across the thresholds it was reported at.

    Derived here rather than recorded by the run, because the journal holds a precision and a
    recall at each threshold and no F1 — and one derivation in one place is the only way the
    number in the README and the number the rule is applied to are the same number.
    """
    return max(_f1(point["precision"], point["recall"]) for point in entry["at"])


def band(epochs: list[dict[str, Any]], window: int = WINDOW) -> float:
    """How much the statistic moved on its own over this rung's last epochs.

    This is the noise a later rung has to beat. It is measured on the rung being compared against
    rather than assumed, so a configuration that settles buys a tighter test for the next change
    and one that does not pays for it.
    """
    recent = [best_f1(entry) for entry in epochs[-window:]]
    return max(recent) - min(recent)


def judge(rungs: list[Rung], window: int = WINDOW) -> list[Verdict]:
    """Walk the ladder, applying the rule, and say what happened to each rung.

    Greedy, which is the issue text's own phrasing — each change is measured against the previous
    configuration. A rejected rung is not what the next one is measured against: the ladder goes
    on from the last rung that was kept, so a change that did not help cannot become the baseline
    that flatters the change after it.
    """
    verdicts: list[Verdict] = []
    standing: Rung | None = None

    for rung in rungs:
        _check_named(rung)
        statistic = best_f1(rung.epochs[-1])

        if standing is None:
            verdicts.append(
                Verdict(
                    label=rung.label,
                    changed=rung.changed,
                    statistic=statistic,
                    against=None,
                    band=None,
                    gain=None,
                    kept=True,
                )
            )
            standing = rung
            continue

        _check_comparable(standing, rung)
        noise = band(standing.epochs, window)
        gain = statistic - best_f1(standing.epochs[-1])
        # Strictly greater. A gain that only reaches the noise the previous rung was already
        # showing is noise, and this is the one character that says so.
        kept = gain > noise

        verdicts.append(
            Verdict(
                label=rung.label,
                changed=rung.changed,
                statistic=statistic,
                against=standing.label,
                band=noise,
                gain=gain,
                kept=kept,
            )
        )
        if kept:
            standing = rung

    return verdicts


def table(verdicts: list[Verdict]) -> str:
    """The ladder as a markdown table, for `docs/` and the README."""
    lines = [
        "| Rung | What changed | Best F1 | Against | Band | Gain | |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for verdict in verdicts:
        lines.append(
            f"| {verdict.label} | {verdict.changed} | {verdict.statistic:.3f} "
            f"| {verdict.against or '—'} | {_maybe(verdict.band)} "
            f"| {_maybe(verdict.gain, sign=True)} "
            f"| {'kept' if verdict.kept else 'rejected'} |"
        )
    return "\n".join(lines)


def _check_named(rung: Rung) -> None:
    if rung.run is None:
        raise ValueError(
            f"{rung.label} does not name the run that produced it, so how it was scored is "
            "unknown; metrics files written before 2026-08-17 are bare lists and cannot be rungs"
        )


def _check_comparable(earlier: Rung, later: Rung) -> None:
    """Refuse two rungs whose numbers do not mean the same thing.

    This is the one place in the ladder where an error would be silent and would end up published.
    A tolerance of 300 m against one of 200 m, or a split of 1500 tiles against one of 3000, gives
    two tables that look alike, subtract cleanly, and mean nothing.
    """
    for field in SAME_REPORTING:
        here, there = earlier.run["reporting"][field], later.run["reporting"][field]
        if here != there:
            raise ValueError(
                f"{earlier.label} and {later.label} were scored with different {field} "
                f"({here!r} against {there!r}), so their numbers cannot be put beside one another"
            )

    for field in SAME_SPLIT:
        here, there = earlier.epochs[-1][field], later.epochs[-1][field]
        if here != there:
            raise ValueError(
                f"{earlier.label} and {later.label} were scored over different splits: "
                f"{field} {here!r} against {there!r}"
            )


def _f1(precision: float | None, recall: float | None) -> float:
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


def _maybe(value: float | None, sign: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if sign else f"{value:.3f}"
