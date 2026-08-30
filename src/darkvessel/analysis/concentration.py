"""Where the dark candidates concentrate, against each variable the context sampling attached.

`context/gee_layers.py` puts a distance, a depth, a zone and a fishing-effort figure on every
detection, and its own closing note says plainly that sampling is not analysis. This is the
analysis: over the whole archive rather than one scene, what fraction of the detections standing
in a given water were undeclared, and how much of the shape of that answer is real.

**The reported quantity is a rate, not a count.** A histogram of where dark detections are found
is mostly a picture of where detections are found at all — the shipping lane carries most of the
traffic in this box and would carry most of the dark candidates under any hypothesis, including
the null one. Dividing by the detections in the same water is what turns "there are more of them
here" into "a larger share of what is here is undeclared", which is the question the ticket asks.

**Every interval is resampled over acquisitions, not over detections.** Two detections from one
scene share a sea state, a morning, a pass direction and frequently a vessel that was there again
a week later. Treating the archive's 189 detections as 189 independent trials narrows every
interval by roughly a third, and a finding that lives only inside that third is arithmetic rather
than water. So the bootstrap draws whole scenes with replacement; `interval_over` is where that
happens and `test_concentration.py` holds it.

**A variable nobody could sample is reported unavailable, never as a null result.** Carried
straight through from `gee_layers.py`, and it is the difference between "the dark candidates are
spread evenly across EEZs" and "no EEZ layer exists in the public catalogue". Only the second is
true here.

What this module deliberately does not do is fit anything. There is no model of the water and no
p-value: four bands, a rate in each, and an interval around each rate, which is the most that 189
detections over ten weeks of one rectangle will support. The confound the archive-wide run put on
the row — the sea state — is reported as a rank correlation over scenes beside the distributions,
because a dark rate that tracked the wind would be a statement about the detector rather than
about the traffic.

No network and no torch: this reads the GeoPackage the chain wrote and writes a JSON and some
SVG, so the numbers in the README are re-derivable on a laptop in a few seconds.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from darkvessel.context.gee_layers import UNAVAILABLE
from darkvessel.fusion.match import DARK

# 1.96, and written as the constant rather than the number so that a reader knows which
# convention is in force. Every interval on this page is a 95% one, the bootstrap's percentiles
# included, so that a figure and a table cannot be quoting two different levels of confidence.
Z = 1.959963984540054
CONFIDENCE = 0.95

# Seeds the Monte Carlo error of the bounds is measured over. Twelve is enough to see the range
# without turning a hundredth-of-a-second command into a slow one, and it is a range rather than a
# standard deviation because what a reader needs is "the printed digit moved by this much".
SPREAD_SEEDS = 12

SCENE = "scene"
STATUS = "status"
EEZ = "eez"
SEA_LEVEL = "sea_level_db"
SEA_SPREAD = "sea_spread_db"


@dataclass(frozen=True)
class Measure:
    """One contextual column, and how to say it out loud.

    `scale` divides the raw column for display only — metres to kilometres — because a band edge
    of 26785.6 m in a README is a number nobody reads and 26.8 km is one they do. The arithmetic
    happens on the column as sampled.
    """

    column: str
    label: str
    unit: str
    scale: float = 1.0


MEASURES = (
    Measure("distance_to_shore_m", "Distance to shore", "km", 1000.0),
    Measure("depth_m", "Water depth", "m"),
    Measure("fishing_hours", "Recorded fishing effort", "hours"),
)


@dataclass(frozen=True)
class Rate:
    """A count of detections, how many were undeclared, and how firmly that share is known.

    Shared by the banded variables and the categorical one. The arithmetic of a share, and the
    rule for deciding whether two shares differ, do not change when the thing being sliced stops
    being a number and starts being the name of a zone — and two copies of `separated_from` would
    be two places for the comparison every finding on this page is stated against to drift apart.
    """

    total: int
    dark: int
    # Over the rows, which assumes they are independent and they are not. Reported beside the
    # other one rather than instead of it, so that the cost of that assumption is visible.
    wilson: tuple[float, float]
    # Over the acquisitions the rows came from. This is the one the findings are stated against.
    interval: tuple[float, float]

    @property
    def rate(self) -> float:
        return float("nan") if self.total == 0 else self.dark / self.total

    @property
    def estimated(self) -> bool:
        """Whether the bootstrap could produce an interval at all.

        False when every detection of the band came out of one acquisition: there is then one
        morning to resample and no uncertainty is measurable from it. Distinct from a wide
        interval, and reported differently.
        """
        return not any(math.isnan(bound) for bound in self.interval)

    def separated_from(self, other: "Rate") -> bool:
        """Whether two scene-wise intervals fail to overlap.

        The weakest claim worth making here and the only comparison this module makes. It is not
        a test — non-overlapping 95% intervals is a stricter bar than a 5% two-sample test, which
        is the direction to err in for a page that will be read as a result.
        """
        if any(math.isnan(bound) for bound in (*self.interval, *other.interval)):
            return False
        return self.interval[1] < other.interval[0] or other.interval[1] < self.interval[0]


@dataclass(frozen=True)
class Band(Rate):
    """One slice of a variable's range, and the share of its detections that were undeclared."""

    low: float = float("nan")
    high: float = float("nan")


@dataclass(frozen=True)
class Zone(Rate):
    """One named water, and the share of the detections standing in it that were undeclared.

    The categorical counterpart of a band. It carries an interval for the same reason a band
    does: "a larger share of what is here is undeclared" is a claim about a difference, and a
    difference between two counts with no interval around either is not a finding.
    """

    name: str = UNAVAILABLE


@dataclass(frozen=True)
class Profile:
    """One variable's whole answer: its bands, and what could not be sampled for it."""

    variable: str
    label: str
    unit: str
    scale: float
    measured: int
    unmeasured: int
    bands: tuple[Band, ...]

    @property
    def available(self) -> bool:
        return self.measured > 0 and len(self.bands) > 0

    @property
    def figure(self) -> str:
        return f"concentration-{self.variable}.svg"

    @property
    def comparable(self) -> bool:
        """Whether any two bands carry intervals that could be compared at all."""
        return sum(1 for band in self.bands if band.estimated) >= 2

    @property
    def separations(self) -> tuple[tuple[int, int], ...]:
        """Every pair of bands whose intervals do not overlap, by position."""
        return tuple(
            (first, second)
            for first in range(len(self.bands))
            for second in range(first + 1, len(self.bands))
            if self.bands[first].separated_from(self.bands[second])
        )

    def lines(self) -> list[str]:
        if not self.available:
            return [
                f"{self.label}: unavailable — "
                f"{self.unmeasured} of {self.measured + self.unmeasured} detections carry no "
                f"`{self.variable}` value, so no distribution is reported"
            ]
        out = [f"{self.label} ({self.unit})"]
        for band in self.bands:
            low, high = band.low / self.scale, band.high / self.scale
            estimated = (
                "not estimated"
                if not band.estimated
                else f"[{band.interval[0]:.1%}, {band.interval[1]:.1%}]"
            )
            out.append(
                f"  {low:8.2f} to {high:8.2f}  n={band.total:3d}  dark={band.dark:3d}  "
                f"{band.rate:6.1%}  {estimated}"
            )
        if self.unmeasured:
            out.append(f"  {self.unmeasured} detections unsampled, excluded")
        if self.separations:
            pairs = ", ".join(
                f"band {first + 1} vs {second + 1}" for first, second in self.separations
            )
            out.append(f"  intervals do not overlap: {pairs}")
        elif not self.comparable:
            # The distinction this whole module is built on, and the place it is easiest to lose.
            # A band whose detections all came from one acquisition has no interval at all, and
            # saying "no concentration established" of it would report "we looked and found
            # nothing" where the truth is "we could not look".
            out.append("  no interval could be estimated; no bands are comparable")
        else:
            out.append("  every interval overlaps every other; no concentration established")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "label": self.label,
            "unit": self.unit,
            "available": self.available,
            "comparable": self.comparable,
            "measured": self.measured,
            "unmeasured": self.unmeasured,
            "figure": self.figure,
            "separations": [list(pair) for pair in self.separations],
            "bands": [
                {
                    "low": band.low,
                    "high": band.high,
                    "total": band.total,
                    "dark": band.dark,
                    "rate": band.rate,
                    "wilson": list(band.wilson),
                    "interval": list(band.interval),
                }
                for band in self.bands
            ],
        }


@dataclass(frozen=True)
class Category:
    """A variable whose values are words rather than numbers. Today that is the EEZ alone."""

    variable: str
    label: str
    zones: tuple[Zone, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {zone.name: zone.total for zone in self.zones}

    @property
    def dark(self) -> dict[str, int]:
        return {zone.name: zone.dark for zone in self.zones}

    @property
    def available(self) -> bool:
        """False when every detection reads `unavailable`, which was the state until #35.

        A layer that answered for some detections and not others is available and carries an
        `unavailable` count beside the zones it did name — a different and more interesting
        situation, and the one a fetch that did not cover the whole archive would produce.
        """
        return any(zone.name != UNAVAILABLE for zone in self.zones)

    @property
    def comparable(self) -> bool:
        """Whether any two named zones carry intervals that could be compared at all."""
        return sum(1 for zone in self.zones if zone.estimated and zone.name != UNAVAILABLE) >= 2

    @property
    def separations(self) -> tuple[tuple[str, str], ...]:
        """Every pair of named zones whose intervals do not overlap, by name.

        By name rather than by position, because a zone is not ordered: "band 1 vs band 2" means
        something about a range and "Denmark vs Sweden" is the only way to say this one.
        """
        named = [zone for zone in self.zones if zone.name != UNAVAILABLE]
        return tuple(
            (first.name, second.name)
            for index, first in enumerate(named)
            for second in named[index + 1 :]
            if first.separated_from(second)
        )

    def lines(self) -> list[str]:
        if not self.available:
            total = sum(zone.total for zone in self.zones)
            return [
                f"{self.label}: unavailable — all {total} detections, because the boundaries have "
                f"not been fetched; run `darkvessel eez` and then `darkvessel zones`"
            ]
        out = [self.label]
        for zone in self.zones:
            estimated = (
                "not estimated"
                if not zone.estimated
                else f"[{zone.interval[0]:.1%}, {zone.interval[1]:.1%}]"
            )
            out.append(
                f"  {zone.name:<16} n={zone.total:3d}  dark={zone.dark:3d}  "
                f"{zone.rate:6.1%}  {estimated}"
            )
        if self.separations:
            pairs = ", ".join(f"{first} vs {second}" for first, second in self.separations)
            out.append(f"  intervals do not overlap: {pairs}")
        elif not self.comparable:
            out.append("  no interval could be estimated; no zones are comparable")
        else:
            out.append("  every interval overlaps every other; no concentration established")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "label": self.label,
            "available": self.available,
            "comparable": self.comparable,
            "counts": dict(sorted(self.counts.items())),
            "dark": dict(sorted(self.dark.items())),
            "separations": [list(pair) for pair in self.separations],
            "zones": [
                {
                    "name": zone.name,
                    "total": zone.total,
                    "dark": zone.dark,
                    "rate": zone.rate,
                    "wilson": list(zone.wilson),
                    "interval": list(zone.interval),
                }
                for zone in self.zones
            ],
        }


@dataclass(frozen=True)
class Sea:
    """The confound, measured over acquisitions.

    One scene has one sea however many detections came out of it, so the correlation is over the
    scenes and not over the rows. Correlating rows would weight each acquisition by its own
    detection count, and the detection count is on the other side of the very question being
    asked.
    """

    scenes: int
    rate_against_level: float
    count_against_level: float
    rate_against_spread: float

    def lines(self) -> list[str]:
        if math.isnan(self.rate_against_level):
            # Two different reasons, and saying the wrong one is worse than saying neither: too
            # few acquisitions to rank, or acquisitions whose sea never moved — which includes
            # the layer that was never sampled for it at all.
            why = (
                "too few to correlate against"
                if self.scenes < 3
                else "the sea state does not vary across them, or was never sampled"
            )
            return [f"Sea state: {self.scenes} acquisition(s), {why}"]
        return [
            f"Sea state, over {self.scenes} acquisitions (Spearman)",
            f"  dark rate vs sea level   {self.rate_against_level:+.2f}",
            f"  detections vs sea level  {self.count_against_level:+.2f}",
            f"  dark rate vs sea spread  {self.rate_against_spread:+.2f}",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenes": self.scenes,
            "rate_against_level": _jsonable(self.rate_against_level),
            "count_against_level": _jsonable(self.count_against_level),
            "rate_against_spread": _jsonable(self.rate_against_spread),
        }


@dataclass(frozen=True)
class Concentration:
    """The whole answer, ready to be printed, written as JSON, or drawn."""

    detections: int
    dark: int
    scenes: int
    wilson: tuple[float, float]
    interval: tuple[float, float]
    # How far the two bounds above move when only the seed changes. Reported so that the digits
    # printed everywhere else on the page can be read at the resolution they actually carry.
    monte_carlo: tuple[float, float]
    profiles: tuple[Profile, ...]
    categories: tuple[Category, ...]
    sea: Sea
    bands: int
    draws: int
    seed: int

    @property
    def rate(self) -> float:
        return self.dark / self.detections

    def lines(self) -> list[str]:
        out = [
            f"{self.detections} detections over {self.scenes} acquisitions, "
            f"{self.dark} of them dark",
            f"  dark rate {self.rate:.1%}  "
            f"[{self.interval[0]:.1%}, {self.interval[1]:.1%}] over acquisitions, "
            f"[{self.wilson[0]:.1%}, {self.wilson[1]:.1%}] if the rows were independent",
            f"  the bounds move {self.monte_carlo[0]:.2%} and {self.monte_carlo[1]:.2%} over "
            f"{SPREAD_SEEDS} seeds at {self.draws} draws; read them at whole-percent resolution",
        ]
        for profile in self.profiles:
            out.extend(profile.lines())
        for category in self.categories:
            out.extend(category.lines())
        out.extend(self.sea.lines())
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "detections": self.detections,
            "dark": self.dark,
            "scenes": self.scenes,
            "rate": self.rate,
            "confidence": CONFIDENCE,
            "wilson": list(self.wilson),
            "interval": list(self.interval),
            "monte_carlo": list(self.monte_carlo),
            "monte_carlo_seeds": SPREAD_SEEDS,
            "bands": self.bands,
            "draws": self.draws,
            "seed": self.seed,
            "profiles": [profile.as_dict() for profile in self.profiles],
            "categories": [category.as_dict() for category in self.categories],
            "sea": self.sea.as_dict(),
        }


def wilson(dark: int, total: int) -> tuple[float, float]:
    """A 95% interval on a proportion that stays inside [0, 1] at either end of the range.

    The normal approximation is the one everybody writes and it is wrong exactly where this
    analysis lives: a band of 47 detections with one dark gets 0.021 +/- 0.041, whose lower bound
    is a negative probability, and a band with none gets zero width, which reads as certainty from
    forty-odd observations. Wilson is the score interval — it solves for the proportions the
    observation would not reject rather than laying a symmetric bar over the estimate — so it is
    asymmetric near the edges and never leaves the unit interval.

    Reported beside the bootstrap rather than instead of it. This one assumes the rows are
    independent draws, which they are not; the pair of them is the honest statement.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    proportion = dark / total
    denominator = 1.0 + Z * Z / total
    centre = (proportion + Z * Z / (2 * total)) / denominator
    half = (
        Z
        * math.sqrt(proportion * (1.0 - proportion) / total + Z * Z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def interval_over(
    dark: np.ndarray, clusters: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float]:
    """A 95% interval on the dark rate, resampling whole acquisitions with replacement.

    **The one decision this module turns on.** The unit that varies is the acquisition, not the
    detection: a scene has one sea, one morning, one pass geometry, and the same vessel can stand
    in three of them. Bootstrapping rows would pretend each detection was an independent trial
    and hand back an interval about a third as wide as the truth, which is the width most of the
    interesting comparisons here are decided inside.

    Whole clusters move: a scene drawn twice contributes both of its detections twice. That is
    what carries the within-scene correlation into the resample, and it is why a draw here can
    only produce rates the scenes can actually make.

    Missing when there are fewer than two clusters, because a bootstrap over one scene resamples
    the same scene every draw and reports zero uncertainty about a single morning.
    """
    if dark.size == 0:
        return (float("nan"), float("nan"))

    names, index = np.unique(clusters, return_inverse=True)
    if names.size < 2:
        return (float("nan"), float("nan"))

    # Rows grouped by cluster once, so a draw is a concatenation rather than a scan of the layer.
    order = np.argsort(index, kind="stable")
    grouped = dark[order]
    starts = np.searchsorted(index[order], np.arange(names.size))
    stops = np.searchsorted(index[order], np.arange(names.size), side="right")
    sums = np.array([grouped[start:stop].sum() for start, stop in zip(starts, stops, strict=True)])
    sizes = stops - starts

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, names.size, size=(draws, names.size))
    drawn_dark = sums[picks].sum(axis=1)
    drawn_total = sizes[picks].sum(axis=1)
    rates = drawn_dark / drawn_total

    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.percentile(rates, [100.0 * tail, 100.0 * (1.0 - tail)])
    return (float(low), float(high))


def monte_carlo_spread(
    dark: np.ndarray, clusters: np.ndarray, *, draws: int, seeds: int
) -> tuple[float, float]:
    """How far each bound of the interval moves when only the seed changes.

    A bootstrap percentile is an estimate with an error of its own, and that error does not go
    away with draws — it shrinks like one over their square root and keeps moving the digit a
    README prints long after the arithmetic has stopped being cheap. Measuring it is the
    alternative to claiming it is small: the figure the page quotes comes out of this function
    and lands in the committed report, so a reader can see how much of the printed precision is
    real.

    Returns the range of the low bound and of the high bound over `seeds` consecutive seeds,
    which is what tells a reader at what resolution to read the bounds elsewhere on the page.
    """
    bounds = [interval_over(dark, clusters, draws=draws, seed=seed) for seed in range(seeds)]
    lows = [low for low, _ in bounds if not math.isnan(low)]
    highs = [high for _, high in bounds if not math.isnan(high)]
    if not lows or not highs:
        return (float("nan"), float("nan"))
    return (max(lows) - min(lows), max(highs) - min(highs))


def cut(values: np.ndarray, *, bands: int) -> np.ndarray:
    """Band edges at the quantiles of every detection's value, dark and declared alike.

    Cutting on the dark subset would define the bins by the thing being measured — the bands
    would follow wherever the dark detections happen to sit and the rate per band would tend to
    flat by construction. Cutting on the population asks a fixed question of each slice of water.

    Quantiles rather than equal widths, so that every band carries about the same number of
    detections and no interval is wide because its band was empty. Duplicate edges collapse: a
    variable that returned one value across the archive is one band, not four, of which three are
    empty and imply a gradient nobody measured. ETOPO1 over a box this size can do exactly that.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([])
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bands + 1)))
    if edges.size == 1:
        return np.array([edges[0], edges[0]])
    return edges


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation, as Pearson over the ranks.

    Written out rather than imported because `scipy` is not a dependency of this project and
    would not become one for eight lines — the same reasoning `pyproject.toml` applies to torch.
    Ranks rather than values because nothing here claims the dark rate is a linear function of
    the sea; the question is only whether it moves with it.

    Ties take their average rank, which is what makes this the standard coefficient rather than
    a near miss of it: the archive has scenes with identical detection counts and the answer must
    not depend on the order they were read in.
    """
    if first.size < 3:
        return float("nan")
    ranked_first = pd.Series(first).rank().to_numpy()
    ranked_second = pd.Series(second).rank().to_numpy()
    if ranked_first.std() == 0 or ranked_second.std() == 0:
        return float("nan")
    return float(np.corrcoef(ranked_first, ranked_second)[0, 1])


def concentrate(
    detections: gpd.GeoDataFrame, *, bands: int = 4, draws: int = 4000, seed: int = 0
) -> Concentration:
    """The distribution of dark candidates against every contextual variable on the layer.

    Reads an accumulated archive layer — the single-scene one has no `scene` column and no
    distribution worth reading anyway — and returns everything the report, the console output and
    the figures are built from, so that all three are the same numbers rather than three
    computations of them.
    """
    if len(detections) == 0:
        raise ValueError(
            "no detections to analyse; the layer is empty, so there is no distribution to describe"
        )
    if SCENE not in detections.columns:
        raise ValueError(
            f"the layer has no `{SCENE}` column, so an interval cannot be resampled over "
            f"acquisitions; `darkvessel analyse` reads the accumulated layer that "
            f"`darkvessel archive-run` writes, not a single run's output"
        )

    dark = (detections[STATUS] == DARK).to_numpy()
    scenes = detections[SCENE].to_numpy()

    return Concentration(
        detections=len(detections),
        dark=int(dark.sum()),
        scenes=int(pd.unique(scenes).size),
        wilson=wilson(int(dark.sum()), len(detections)),
        interval=interval_over(dark, scenes, draws=draws, seed=seed),
        monte_carlo=monte_carlo_spread(dark, scenes, draws=draws, seeds=SPREAD_SEEDS),
        # Every measure, including one whose column is not on the layer at all. Filtering those
        # out would report a variable that was never sampled by saying nothing about it, which is
        # the same silence this module refuses for a column of nulls.
        profiles=tuple(
            _profile(detections, measure, dark, scenes, bands=bands, draws=draws, seed=seed)
            for measure in MEASURES
        ),
        categories=(
            (_zones(detections, dark, scenes, draws=draws, seed=seed),)
            if EEZ in detections.columns
            else ()
        ),
        sea=_sea(detections, dark, scenes),
        bands=bands,
        draws=draws,
        seed=seed,
    )


def _profile(
    detections: gpd.GeoDataFrame,
    measure: Measure,
    dark: np.ndarray,
    scenes: np.ndarray,
    *,
    bands: int,
    draws: int,
    seed: int,
) -> Profile:
    values = _column(detections, measure.column)
    sampled = np.isfinite(values)
    edges = cut(values, bands=bands)

    built: list[Band] = []
    for position in range(max(edges.size - 1, 0)):
        low, high = float(edges[position]), float(edges[position + 1])
        # Left-closed everywhere so the lowest value is in the first band, right-closed
        # thereafter so no value falls between two edges. A row lost through a gap would quietly
        # leave the denominator of a rate rather than raise anything.
        inside = sampled & (values <= high) & (values >= low if position == 0 else values > low)
        built.append(
            Band(
                low=low,
                high=high,
                total=int(inside.sum()),
                dark=int(dark[inside].sum()),
                wilson=wilson(int(dark[inside].sum()), int(inside.sum())),
                # Seeded per band, so that adding a variable to the config cannot move the
                # numbers already published for the others.
                interval=interval_over(
                    dark[inside], scenes[inside], draws=draws, seed=seed + position
                ),
            )
        )

    return Profile(
        variable=measure.column,
        label=measure.label,
        unit=measure.unit,
        scale=measure.scale,
        measured=int(sampled.sum()),
        unmeasured=int((~sampled).sum()),
        bands=tuple(built) if sampled.any() else (),
    )


def _zones(
    detections: gpd.GeoDataFrame,
    dark: np.ndarray,
    scenes: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> Category:
    zones = detections[EEZ].astype("string").fillna(UNAVAILABLE).to_numpy()
    names = sorted(set(zones.tolist()))
    return Category(
        variable=EEZ,
        label="EEZ",
        zones=tuple(
            Zone(
                name=name,
                total=int((zones == name).sum()),
                dark=int(dark[zones == name].sum()),
                wilson=wilson(int(dark[zones == name].sum()), int((zones == name).sum())),
                # Seeded by position in the sorted names, the convention `_profile` uses: adding
                # a variable, or a zone, cannot move the numbers already published for the rest.
                interval=interval_over(
                    dark[zones == name], scenes[zones == name], draws=draws, seed=seed + position
                ),
            )
            for position, name in enumerate(names)
        ),
    )


def _sea(detections: gpd.GeoDataFrame, dark: np.ndarray, scenes: np.ndarray) -> Sea:
    """The dark rate of each acquisition against the sea it was acquired in."""
    frame = pd.DataFrame(
        {
            SCENE: scenes,
            "dark": dark,
            SEA_LEVEL: _column(detections, SEA_LEVEL),
            SEA_SPREAD: _column(detections, SEA_SPREAD),
        }
    )
    # `first` rather than `mean`: the two sea columns are properties of the acquisition and are
    # identical down every one of its rows, so an average would be the same number arrived at in
    # a way that would hide a scene whose rows disagreed.
    per_scene = frame.groupby(SCENE, sort=True).agg(
        detections=("dark", "size"),
        dark=("dark", "sum"),
        level=(SEA_LEVEL, "first"),
        spread=(SEA_SPREAD, "first"),
    )
    rate = (per_scene.dark / per_scene.detections).to_numpy(dtype=float)
    return Sea(
        scenes=int(len(per_scene)),
        rate_against_level=spearman(rate, per_scene.level.to_numpy(dtype=float)),
        count_against_level=spearman(
            per_scene.detections.to_numpy(dtype=float), per_scene.level.to_numpy(dtype=float)
        ),
        rate_against_spread=spearman(rate, per_scene.spread.to_numpy(dtype=float)),
    )


def _column(detections: gpd.GeoDataFrame, name: str) -> np.ndarray:
    if name not in detections.columns:
        return np.full(len(detections), np.nan)
    return pd.to_numeric(detections[name], errors="coerce").to_numpy(dtype=float)


def _jsonable(value: float) -> float | None:
    return None if math.isnan(value) else value


# The palette and geometry of `detect/curve.py`, for the reason stated there: these figures are
# read on a page whose background is white or near-black depending on the reader, so nothing is
# painted and the ink is a grey that carries on both.
_INK = "#8b949e"
_BAR = "#1f6feb"
_OVERALL = "#f78166"
_WIDTH = 640
_HEIGHT = 320
_LEFT = 64
_RIGHT = 24
_TOP = 24
_BOTTOM = 72
_PLOT_W = _WIDTH - _LEFT - _RIGHT
_PLOT_H = _HEIGHT - _TOP - _BOTTOM


def svg(profile: Profile, *, overall: float | None = None) -> str:
    """One variable's bands, left to right in the order they are reported, with their intervals.

    The bar is the rate and the whisker is the scene-wise interval, and the whisker is the point:
    four bars alone would draw differences this data does not support, and every finding on this
    page is a statement about whether two whiskers overlap. A figure that dropped them would
    contradict the prose beside it while looking like its evidence.

    The x order is the variable's own order — nearest shore leftmost, shallowest leftmost — and
    it is asserted in the test as well as stated here, because a figure drawn the other way round
    reads as the analysis's opposite and looks entirely correct.
    """
    if not profile.available:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} 80" '
            f'width="{_WIDTH}" height="80" font-family="system-ui, sans-serif">'
            f'<text x="{_WIDTH / 2:.0f}" y="44" fill="{_INK}" font-size="14" '
            f'text-anchor="middle">{profile.label}: not sampled, no distribution</text>'
            f"</svg>"
        )

    ceiling = max(
        [band.interval[1] for band in profile.bands if not math.isnan(band.interval[1])]
        + [band.rate for band in profile.bands]
        + [0.1]
    )
    top = min(1.0, math.ceil(ceiling * 10.0) / 10.0)

    def y_of(value: float) -> float:
        return _TOP + _PLOT_H * (1.0 - value / top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'width="{_WIDTH}" height="{_HEIGHT}" font-family="system-ui, sans-serif">',
        f'<rect x="{_LEFT}" y="{_TOP}" width="{_PLOT_W}" height="{_PLOT_H}" fill="none" '
        f'stroke="{_INK}" stroke-width="1"/>',
    ]

    ticks = 5
    for step in range(ticks + 1):
        value = top * step / ticks
        y = y_of(value)
        parts.append(
            f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_LEFT + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="{_INK}" stroke-width="0.5" stroke-dasharray="2 4"/>'
        )
        parts.append(
            f'<text x="{_LEFT - 10}" y="{y + 4:.1f}" fill="{_INK}" font-size="12" '
            f'text-anchor="end">{value:.0%}</text>'
        )

    if overall is not None:
        y = y_of(min(overall, top))
        parts.append(
            f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_LEFT + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="{_OVERALL}" stroke-width="1.5" stroke-dasharray="6 3"/>'
        )
        parts.append(
            f'<text x="{_LEFT + _PLOT_W - 4}" y="{y - 6:.1f}" fill="{_OVERALL}" font-size="11" '
            f'text-anchor="end">archive {overall:.1%}</text>'
        )

    slot = _PLOT_W / len(profile.bands)
    width = slot * 0.5
    for position, band in enumerate(profile.bands):
        centre = _LEFT + slot * (position + 0.5)
        baseline = y_of(0.0)
        top_of_bar = y_of(min(band.rate, top))
        parts.append(
            f'<rect data-band-x="{centre:.1f}" x="{centre - width / 2:.1f}" '
            f'y="{top_of_bar:.1f}" width="{width:.1f}" '
            f'height="{max(baseline - top_of_bar, 0.0):.1f}" fill="{_BAR}" opacity="0.55"/>'
        )
        low, high = band.interval
        if not (math.isnan(low) or math.isnan(high)):
            parts.append(
                f'<line data-interval="{low:.4f},{high:.4f}" x1="{centre:.1f}" '
                f'y1="{y_of(min(high, top)):.1f}" x2="{centre:.1f}" '
                f'y2="{y_of(min(low, top)):.1f}" stroke="{_BAR}" stroke-width="2"/>'
            )
            for bound in (low, high):
                y = y_of(min(bound, top))
                parts.append(
                    f'<line x1="{centre - 8:.1f}" y1="{y:.1f}" x2="{centre + 8:.1f}" '
                    f'y2="{y:.1f}" stroke="{_BAR}" stroke-width="2"/>'
                )
        else:
            # One acquisition in the band: the rate is drawn and the interval is deliberately
            # absent, which says nothing is known about it rather than that it is tight.
            parts.append(
                f'<line data-interval="none" x1="{centre:.1f}" y1="{baseline:.1f}" '
                f'x2="{centre:.1f}" y2="{baseline:.1f}" stroke="none"/>'
            )

        parts.append(
            f'<text x="{centre:.1f}" y="{_TOP + _PLOT_H + 18:.1f}" fill="{_INK}" font-size="11" '
            # "to" rather than an en dash: depth arrives negative, and "-49.0--42.0" is a
            # label a reader has to decode rather than read.
            f'text-anchor="middle">{band.low / profile.scale:.1f} to '
            f"{band.high / profile.scale:.1f}</text>"
        )
        parts.append(
            f'<text x="{centre:.1f}" y="{_TOP + _PLOT_H + 33:.1f}" fill="{_INK}" font-size="11" '
            f'text-anchor="middle">n={band.total}</text>'
        )

    parts.append(
        f'<text x="{_LEFT + _PLOT_W / 2:.1f}" y="{_HEIGHT - 12}" fill="{_INK}" font-size="13" '
        f'text-anchor="middle">{profile.label} ({profile.unit})</text>'
    )
    parts.append(
        f'<text x="18" y="{_TOP + _PLOT_H / 2:.1f}" fill="{_INK}" font-size="13" '
        f'text-anchor="middle" transform="rotate(-90 18 {_TOP + _PLOT_H / 2:.1f})">'
        f"Share undeclared</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def report(result: Concentration) -> str:
    return json.dumps(result.as_dict(), indent=2) + "\n"


def write(result: Concentration, *, report_path: Path, figures: Path) -> list[Path]:
    """The JSON and one SVG per variable, returned in the order they were written."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(result))
    written = [report_path]

    figures.mkdir(parents=True, exist_ok=True)
    for profile in result.profiles:
        path = figures / profile.figure
        path.write_text(svg(profile, overall=result.rate) + "\n")
        written.append(path)
    return written


def analysis_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What the analysis is asked for, read out of a config file.

    The same shape as the other `*_request_from` functions in `cli.py`, and here for the reason
    they are there: what can be checked without doing the work is checked before the work starts.
    A seed and a draw count in the file rather than in the code because the intervals they produce
    are quoted in a README, and a number nobody can re-derive from a committed file is not a
    measurement.
    """
    analysis = config.get("analysis", {})
    draws = int(analysis.get("draws", 4000))
    bands = int(analysis.get("bands", 4))
    if bands < 2:
        raise ValueError(f"analysis.bands is {bands}; a distribution needs at least two bands")
    if draws < 1:
        raise ValueError(f"analysis.draws is {draws}; the interval is a bootstrap and needs draws")
    return {
        "bands": bands,
        "draws": draws,
        "seed": int(analysis.get("seed", 0)),
        "report": (relative_to / str(analysis.get("report", "analysis.json"))).resolve(),
        "figures": (relative_to / str(analysis.get("figures", "figures"))).resolve(),
    }
