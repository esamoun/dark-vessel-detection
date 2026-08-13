"""Where to put the study area, measured out of the archive rather than picked off a map.

The study area is the one choice above which nothing can be recovered. Every level of this chain
can be correct and still have nothing to say, because a rectangle with no traffic in it produces
a detector's false positives and an AIS slice with nothing in it to explain them. That is what
happened over Anholt: thirty acquisitions, nineteen of them with no declared vessel in the frame
at all, and the largest ship ever present 15 m long.

So the rectangle is measured. The measurement has to survive three ways of being wrong, and each
one favours a different rectangle:

  - **Counting reports** measures dwell and transmit rate. A vessel alongside a quay reports for
    twelve hours; one crossing at 14 knots is gone in twenty minutes. A harbour wins.
  - **Counting vessels over a day** measures throughput, not presence. A rectangle one ship
    crosses at dawn scores like one that always has traffic in it, and an acquisition arrives at
    a moment nobody chose.
  - **Counting everything afloat** measures the wrong fleet. At 10 m pixels a 15 m sailing boat
    is a pixel and a half, so a rectangle chosen for how many of them cross it is chosen on
    evidence the radar cannot see either way.

What is counted here is the number of distinct vessels, at least `min_length_m` long and under
way rather than moored, standing inside the rectangle during a window the width of the one
`darkvessel ais` actually fetches. Averaged over every window of the day, including the empty
ones — that average is what a single acquisition can expect to catch.

The archive is a parameter, as it is for the ingestion: 662 MB of Danish AIS is not something a
test suite downloads, and the whole day is streamed and discarded rather than stored.
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from darkvessel.data.ais import TIMESTAMP_FORMAT, from_a_vessel, read_day
from darkvessel.data.area import Bounds
from darkvessel.data.dma import Archive

# What the survey reads out of the archive's twenty-six columns. The ingestion's five, less the
# timestamp's role — here it decides which window a report falls in rather than whether it is in
# one at all — plus the speed, which is the only thing that separates a lane from an anchorage.
COLUMNS = {
    "# Timestamp": "timestamp",
    "Type of mobile": "mobile",
    "MMSI": "mmsi",
    "Latitude": "lat",
    "Longitude": "lon",
    "SOG": "speed_kn",
    "Length": "length_m",
}

_DAY_S = 24 * 60 * 60


@dataclass(frozen=True)
class Grid:
    """The lattice candidate rectangles stand on: where they may go, and how far apart.

    One object because the two numbers only mean anything together. `stride` is both how far
    apart candidate rectangles are placed and the size of the cells their counts accumulate on,
    and `region` is both the bound on where a rectangle may stand and the origin those cells are
    counted from. Split across a pair of parameters they travelled through every function here.
    """

    region: Bounds
    stride: float

    def steps(self, span: float, axis: str) -> int:
        """How many whole steps of `stride` make up `span`.

        Refused when it is not a whole number, because the cells a rectangle is scored from have
        to tile it. Off by a fraction of a step, the rectangle offered is not the rectangle
        measured: the count belongs to a slightly different box, and every part of the answer
        still looks exactly right.
        """
        steps = span / self.stride
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"{span:g} degrees {axis} is not a whole number of steps of {self.stride:g}; the "
                "cells a rectangle is scored from have to tile it, or the count belongs to a "
                "different box"
            )
        return int(round(steps))

    def cells_east(self) -> int:
        return self.steps(self.region.east - self.region.west, "wide")

    def cells_north(self) -> int:
        return self.steps(self.region.north - self.region.south, "high")

    def cell_east(self, longitudes: pd.Series) -> pd.Series:
        return self._cell(longitudes, self.region.west)

    def cell_north(self, latitudes: pd.Series) -> pd.Series:
        return self._cell(latitudes, self.region.south)

    def _cell(self, degrees: pd.Series, origin: float) -> pd.Series:
        """Which step of the grid a coordinate falls in, counted from the region's own corner.

        A coordinate standing exactly on the far edge of the region belongs to the last cell
        rather than to a cell past the end of it, which is the one place this runs off the grid.
        """
        steps = np.floor((degrees - origin) / self.stride + 1e-9).astype("int64")
        return steps.clip(lower=0)

    def rectangle(self, west: int, south: int, across: int, up: int) -> Bounds:
        """The candidate rectangle whose south-west corner is that cell."""
        return Bounds(
            west=self.region.west + west * self.stride,
            south=self.region.south + south * self.stride,
            east=self.region.west + (west + across) * self.stride,
            north=self.region.south + (south + up) * self.stride,
        )


@dataclass(frozen=True)
class Candidate:
    """One rectangle the study area could be, and what stood inside it.

    `vessels` is the day's throughput and `mean_in_window` is what one acquisition can expect.
    They rank rectangles differently and both are reported, because a reader choosing between two
    rectangles is choosing between two kinds of evidence and should see both.
    """

    bounds: Bounds
    vessels: int
    mean_in_window: float
    fewest_in_window: int
    empty_windows: int

    def line(self) -> str:
        """One row of the survey, as the command prints it."""
        return (
            f"{self.bounds.west:6.2f} {self.bounds.south:6.2f}  "
            f"to {self.bounds.east:6.2f} {self.bounds.north:6.2f}  "
            f"{self.vessels:5d} over the day  "
            f"{self.mean_in_window:5.2f} in a window  "
            f"fewest {self.fewest_in_window:3d}  "
            f"{self.empty_windows:4d} windows empty"
        )


def survey(
    *,
    archive: Archive,
    day: date,
    region: Bounds,
    box: tuple[float, float],
    stride: float,
    window: timedelta,
    min_length_m: float,
    under_way_kn: float,
) -> list[Candidate]:
    """Score every rectangle of size `box` that fits inside `region`, best first.

    Args:
        archive: Where the daily file comes from. Injected, so the whole survey runs in a test.
        day: The day measured. One archive, streamed once.
        region: Where candidate rectangles may stand. A rectangle is never allowed to hang over
            the edge: its count would be short by however much of it lay outside, and nothing in
            the answer would say so.
        box: The rectangle being searched for, as (longitude, latitude) degrees. The size of the
            study area a config would then declare, or the measurement is about a box nobody is
            going to export.
        stride: How far apart candidate rectangles are placed, in degrees. Also the cell size the
            counts are accumulated on, so `box` must be a whole number of steps across.
        window: How long each stand-in acquisition is. The ingestion's own window, so the number
            reported is the number of declarations a run would actually have to match against.
        min_length_m: The shortest vessel the radar is taken to resolve. Vessels that never
            declared a length are not counted, which understates rather than overstates.
        under_way_kn: The speed at or above which a vessel counts as traffic rather than as
            something moored or at anchor.

    Returns:
        One `Candidate` per rectangle, ordered so the first is the one to take.
    """
    grid = Grid(region=region, stride=stride)
    across, up = grid.steps(box[0], "wide"), grid.steps(box[1], "high")

    presence = _presence(archive, day, grid, window, min_length_m, under_way_kn)
    windows = math.ceil(_DAY_S / window.total_seconds())

    return sorted(
        (
            _score(
                presence, grid.rectangle(west, south, across, up), west, south, across, up, windows
            )
            for west in range(grid.cells_east() - across + 1)
            for south in range(grid.cells_north() - up + 1)
        ),
        key=lambda candidate: (
            candidate.fewest_in_window,
            candidate.mean_in_window,
            candidate.vessels,
        ),
        reverse=True,
    )


def _presence(
    archive: Archive,
    day: date,
    grid: Grid,
    window: timedelta,
    min_length_m: float,
    under_way_kn: float,
) -> pd.DataFrame:
    """One row per vessel, cell and window it was seen under way in.

    Collapsed to that as early as it can be, which is what keeps a day of Danish AIS out of
    memory: a vessel reporting every two seconds for twenty minutes arrives as six hundred rows
    and leaves as one.

    Length is the exception and cannot be decided per chunk. The archive merges static data into
    position rows only when the receiver has it, so a vessel declares its size on a few rows
    scattered through the day and leaves the field blank on the rest — the longest it ever
    claimed is only known once the day has been read.
    """
    seen, declared = [], []
    for chunk in read_day(archive, day, COLUMNS):
        here = _inside(chunk, grid, day, window, under_way_kn)
        # The length is taken off every row and the rows are collapsed afterwards. The other way
        # round, a vessel whose first report of a window happened to carry no static data would
        # lose its size to the copy that survived the collapse — and an unknown length is not
        # counted, so the vessel would silently stop being traffic.
        declared.append(here.groupby("mmsi", observed=True)["length_m"].max())
        seen.append(here.drop(columns=["length_m"]).drop_duplicates())

    presence = pd.concat(seen, ignore_index=True) if seen else _no_presence()
    presence = presence.groupby(["mmsi", "cell_east", "cell_north", "window"], as_index=False)[
        "under_way"
    ].max()

    lengths = pd.concat(declared) if declared else pd.Series(dtype="float64")
    long_enough = lengths.groupby(level=0).max() >= min_length_m

    return presence[
        presence["under_way"] & presence["mmsi"].map(long_enough).fillna(False).astype(bool)
    ]


def _no_presence() -> pd.DataFrame:
    """An archive with nothing readable in it. A day with no traffic is a result, not an error."""
    return pd.DataFrame(
        {
            "mmsi": pd.Series(dtype="string"),
            "cell_east": pd.Series(dtype="int64"),
            "cell_north": pd.Series(dtype="int64"),
            "window": pd.Series(dtype="int64"),
            "under_way": pd.Series(dtype="bool"),
            "length_m": pd.Series(dtype="float64"),
        }
    )


def _inside(
    chunk: pd.DataFrame,
    grid: Grid,
    day: date,
    window: timedelta,
    under_way_kn: float,
) -> pd.DataFrame:
    """The reports of this chunk that stand inside the region, placed in a cell and a window.

    Position first, because it is a numeric comparison over every row of the archive and the
    timestamps are the expensive parse. What survives is a small fraction of what arrives.

    Base stations and aids to navigation go here rather than later: the archive carries every
    transmitter in Danish waters, they never move, and a rectangle chosen for the shore beside it
    is the mistake this whole measurement exists to avoid.
    """
    region = grid.region
    lat = pd.to_numeric(chunk["lat"], errors="coerce").astype("float64")
    lon = pd.to_numeric(chunk["lon"], errors="coerce").astype("float64")
    inside = (
        from_a_vessel(chunk)
        & lat.between(region.south, region.north)
        & lon.between(region.west, region.east)
    )

    here = chunk[inside]
    when = pd.to_datetime(here["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce", utc=True)
    # A report whose timestamp cannot be read belongs to no window. Dropped rather than counted,
    # and not counted out loud either: this is a survey of where traffic is, not a claim about
    # which vessels declared themselves, so an unreadable row costs a rectangle a fraction of a
    # count rather than costing a vessel its declaration.
    readable = when.notna()

    return pd.DataFrame(
        {
            "mmsi": here["mmsi"][readable].astype("string"),
            "cell_east": grid.cell_east(lon[inside][readable]),
            "cell_north": grid.cell_north(lat[inside][readable]),
            "window": _window(when[readable], day, window),
            "under_way": pd.to_numeric(here["speed_kn"][readable], errors="coerce").astype(
                "float64"
            )
            >= under_way_kn,
            "length_m": pd.to_numeric(here["length_m"][readable], errors="coerce").astype(
                "float64"
            ),
        }
    )


def _window(when: pd.Series, day: date, window: timedelta) -> pd.Series:
    """Which stand-in acquisition a report belongs to.

    Whole windows counted from midnight, rather than a window centred on each report. Every
    report then falls in exactly one, so a vessel crossing a boundary is counted once on each
    side rather than smeared across both.
    """
    midnight = pd.Timestamp(day, tz="UTC")
    return (
        ((when - midnight).dt.total_seconds() // window.total_seconds())
        .astype("int64")
        .clip(lower=0)
    )


def _score(
    presence: pd.DataFrame,
    bounds: Bounds,
    west: int,
    south: int,
    across: int,
    up: int,
    windows: int,
) -> Candidate:
    """What stood inside one candidate rectangle.

    Counted over the cells the rectangle covers together rather than cell by cell. Rectangles are
    placed a step apart and are several steps across, so a vessel crossing a cell boundary stands
    in two of the cells one rectangle is scored from — and summed per cell, one ship becomes two.
    A rectangle straddling a lane would then be the one that wins.
    """
    here = presence[
        presence["cell_east"].between(west, west + across - 1)
        & presence["cell_north"].between(south, south + up - 1)
    ]
    in_window = (
        here.groupby("window")["mmsi"]
        .nunique()
        .reindex(range(windows), fill_value=0)
        .astype("int64")
    )

    return Candidate(
        bounds=bounds,
        vessels=int(here["mmsi"].nunique()),
        mean_in_window=float(in_window.mean()),
        fewest_in_window=int(in_window.min()),
        empty_windows=int((in_window == 0).sum()),
    )
