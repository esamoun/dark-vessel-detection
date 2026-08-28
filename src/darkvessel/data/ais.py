"""AIS ingestion from the Danish Maritime Authority open archives.

Downloads daily archives, parses position reports, filters to the study area and time window,
and cleans what is left. Raw AIS is noisy — duplicated messages, implausible jumps, missing
identifiers — and cleaning is part of the work rather than a preliminary to it. Every rule below
was written against a shape of noise found in the real archive, and every row it removes is
counted: a cleaning step nobody can audit is a filter that quietly decides what the answer is.

The direction of the errors matters and is not symmetric. A declaration wrongly removed is a
detection published as a dark vessel — a finding this project would be reporting, and the fault
the whole fusion level exists to remove. A declaration wrongly kept is a match that explains
nothing, which is quieter and less damaging. So the filters are wide where they can be (the
study area is grown by a margin, a vessel that reported once survives) and the removals are
confined to reports that cannot be part of any track: no position, no vessel behind them, or a
position the rest of the vessel's own track cannot reach.

The archive itself is a parameter — see `dma.py` for why, and for how 3.3 GB of CSV gets read on
a machine that cannot hold it.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from darkvessel.data.area import WGS84 as _WGS84
from darkvessel.data.area import Bounds
from darkvessel.data.dma import Archive

# AIS reports positions in WGS84; everything downstream works in the projected CRS.
AIS_CRS = "EPSG:4326"

# The six of the archive's twenty-six columns this project reads, and what it calls them.
COLUMNS = {
    "# Timestamp": "timestamp",
    "Type of mobile": "mobile",
    "MMSI": "mmsi",
    "Latitude": "lat",
    "Longitude": "lon",
    # How big the vessel says it is. Not used to filter anything: it is carried so that a match,
    # or the absence of one, can be read. At 10 m pixels a 15 m sailing boat is a pixel and a
    # half and a 200 m tanker is twenty, so a scene full of small craft and a scene full of
    # cargo are different claims about what the radar could have seen at all.
    "Length": "length_m",
}
# Day first, and no zone: the archive is published in UTC and says so nowhere in the file. Read
# as local time this is out by two hours over a Danish summer, which is 40 km of vessel track and
# would surface only as a scene that looks full of undeclared ships. The assumption is checked
# against the first real run rather than trusted; see README.
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

# The archive carries every transmitter in Danish waters. A base station is on land and an aid to
# navigation is a buoy or a lighthouse: both are real, neither is a vessel, and a detection one of
# them explained would read in the output exactly like a ship that declared itself. Fixed
# structures standing in a radar scene are the detector's problem — a wind farm's turbines are
# the case this project has already documented — and not one the fusion level may explain away.
VESSEL_CLASSES = ("Class A", "Class B")

# Rows parsed at a time. Large enough that the per-chunk overhead disappears, small enough that a
# day of Danish AIS never stands in memory at once.
CHUNK_ROWS = 500_000

# How close a report has to be to the middle of its own track before the outlier rule stops
# having an opinion. AIS positions are good to tens of metres and two reports a second apart
# differ by that much from a vessel standing still, so without a floor the rule would fire on
# the accuracy of the archive rather than on its mistakes. Set at the order of the match
# tolerance: a report displaced by less than that radius cannot change any verdict, so there is
# nothing to gain by removing it and a vessel to lose by getting it wrong.
POSITION_ACCURACY_M = 200.0

_KNOT_MS = 0.514444


@dataclass(frozen=True)
class Cleaning:
    """What the archive gave, and what was left after each rule had taken its share.

    Every count is a rule someone can disagree with, which is the point of publishing them: a
    slice is a claim about which vessels declared themselves, and it is only as good as what was
    thrown away on the way to it.
    """

    read: int
    no_position: int
    no_timestamp: int
    in_area_and_window: int
    not_a_vessel: int
    missing_identifier: int
    duplicated: int
    contradictory: int
    implausible_jump: int
    kept: int

    def lines(self) -> list[str]:
        """The report, said out loud, for a command to print."""
        removed = (
            self.not_a_vessel
            + self.missing_identifier
            + self.duplicated
            + self.contradictory
            + self.implausible_jump
        )
        return [
            f"{self.read} position reports read, {self.no_position} of them with no usable "
            "position",
            f"{self.in_area_and_window} in the study area and the window, "
            f"{self.no_timestamp} more inside the area with no readable timestamp",
            f"of those, {removed} removed by cleaning: {self.not_a_vessel} not a vessel, "
            f"{self.missing_identifier} with no nine-digit identifier, {self.duplicated} "
            f"duplicated, {self.contradictory} contradicting another report of the same instant, "
            f"{self.implausible_jump} at a position the rest of their own track cannot reach",
            f"{self.kept} declared positions kept",
        ]


def slice_for(
    *,
    archive: Archive,
    acquired_at: datetime,
    area: Bounds,
    window: timedelta,
    margin_m: float,
    max_speed_kn: float,
) -> tuple[gpd.GeoDataFrame, Cleaning]:
    """The declarations that could explain a detection in this scene, and what cleaning removed.

    Args:
        archive: Where the daily files come from. Injected, so the whole ingestion runs in a test.
        acquired_at: The moment the radar imaged the scene. Decides both which days are opened
            and where the window sits inside them.
        area: The study area, in WGS84 degrees — the same rectangle the scene was cut to.
        window: How far either side of the acquisition reports are taken from. A vessel is placed
            by interpolating between the reports bracketing the acquisition, so this only has to
            be wide enough to contain a bracket, not a voyage.
        margin_m: How far outside `area` a report is still taken. See `Bounds.grown_by`.
        max_speed_kn: How fast a vessel is allowed to have been, in deciding whether one of its
            reports stands somewhere the others say it cannot have been. A ceiling on the
            physically possible rather than on the usual: it is here to remove positions nothing
            on the water could have reached, and a ferry at 40 knots must pass it untouched.

    Returns:
        One row per surviving report, in WGS84, with `mmsi`, `timestamp`, `length_m` and point
        geometry — what `interpolate.py` places and `match.py` compares against — and the count
        of what every rule removed on the way.
    """
    return slices_for(
        archive=archive,
        acquisitions=[acquired_at],
        area=area,
        window=window,
        margin_m=margin_m,
        max_speed_kn=max_speed_kn,
    )[acquired_at]


def slices_for(
    *,
    archive: Archive,
    acquisitions: Sequence[datetime],
    area: Bounds,
    window: timedelta,
    margin_m: float,
    max_speed_kn: float,
) -> dict[datetime, tuple[gpd.GeoDataFrame, Cleaning]]:
    """The same slice as `slice_for`, for several acquisitions, opening each day once.

    The archive is one file per day and an archive-wide run is many acquisitions over few days:
    the Kattegat archive is 50 scenes on 32 dates, and asking for them one at a time downloads
    18 days twice or three times. A day is 662 MB across the wire, so that is not an efficiency
    detail — it is 12 GB, and it is the whole cost of an archive-wide run.

    Grouping is the only thing this shares between acquisitions. Each one is filtered, cleaned
    and counted against its own window exactly as `slice_for` does, and a scene's slice does not
    depend on which other scenes were asked for in the same call. That is what makes this a
    grouping of the existing ingestion rather than a second one: the single-acquisition case
    above now runs through here, so there is no version of this logic that only many-scene runs
    exercise.

    Args:
        acquisitions: The moments to slice around, in any order. Every one gets its own entry in
            the result, keyed by the instant asked for.

    Returns:
        One `(reports, cleaning)` pair per acquisition, keyed by acquisition instant.

    Raises:
        ValueError: If no acquisition is given, or if one is given twice — the result is keyed by
            instant, so a repeat would silently return one slice where two were asked for.
    """
    if not acquisitions:
        raise ValueError("no acquisitions to slice around; an archive-wide run needs at least one")
    repeated = sorted({one for one in acquisitions if list(acquisitions).count(one) > 1})
    if repeated:
        raise ValueError(
            f"{len(repeated)} acquisition instant(s) asked for more than once, the first being "
            f"{repeated[0].isoformat()}; the result is keyed by instant, so a repeat would come "
            "back as one slice where two were asked for"
        )

    searched = area.grown_by(margin_m)
    # Which acquisitions each day serves, worked out before anything is downloaded: a day is
    # opened once and every window that reaches into it is filled from the same pass.
    touching: dict[date, list[datetime]] = {}
    for acquired_at in acquisitions:
        for day in _days_the_window_touches(acquired_at, window):
            touching.setdefault(day, []).append(acquired_at)

    surviving: dict[datetime, list[_Surviving]] = {one: [] for one in acquisitions}
    for day in sorted(touching):
        for chunk in read_day(archive, day, COLUMNS):
            # Once per chunk rather than once per acquisition. The position filter runs over
            # every row of the archive and the timestamp parse is the expensive one; the window
            # is a comparison over what survives both, which is a thousandth as many rows.
            placed = _placed(chunk, searched)
            for acquired_at in touching[day]:
                surviving[acquired_at].append(_within_window(placed, acquired_at, window))

    return {
        acquired_at: _cleaned(
            reports=_joined([chunk.reports for chunk in chunks]),
            read=sum(chunk.read for chunk in chunks),
            no_position=sum(chunk.no_position for chunk in chunks),
            no_timestamp=sum(chunk.no_timestamp for chunk in chunks),
            max_speed_kn=max_speed_kn,
        )
        for acquired_at, chunks in surviving.items()
    }


def independent_groups(acquisitions: Sequence[datetime], window: timedelta) -> list[list[datetime]]:
    """The acquisitions split into sets that share no archive day with any other set.

    An archive-wide ingestion is 21 GB across the wire and takes hours, so it has to be resumable,
    and the unit it can resume at is a set of acquisitions that can be sliced without opening a
    day twice. Splitting anywhere else would either re-download a day or slice a scene from half
    its archive.

    Almost always a group is one day: a Sentinel-1 pass over a Danish box happens near 05:30 or
    17:00 and a fifteen-minute window sits well inside the date. The acquisition just after
    midnight is the exception this exists for — it pulls the day before it into its own group,
    rather than being quietly sliced from one of the two files it needs.
    """
    root: dict[date, date] = {}

    def find(day: date) -> date:
        root.setdefault(day, day)
        while root[day] != day:
            root[day] = root[root[day]]
            day = root[day]
        return day

    for acquired_at in acquisitions:
        days = _days_the_window_touches(acquired_at, window)
        for other in days[1:]:
            first, second = find(days[0]), find(other)
            if first != second:
                root[first] = second

    grouped: dict[date, list[datetime]] = {}
    for acquired_at in acquisitions:
        anchor = find(_days_the_window_touches(acquired_at, window)[0])
        grouped.setdefault(anchor, []).append(acquired_at)
    return [sorted(grouped[anchor]) for anchor in sorted(grouped)]


def _joined(slices: list[pd.DataFrame]) -> pd.DataFrame:
    """The chunks that survived, as one frame. An archive with nothing in the window gives none."""
    if not slices:
        return pd.DataFrame(columns=["timestamp", "mobile", "mmsi", "lat", "lon", "length_m"])
    return pd.concat(slices, ignore_index=True)


def _days_the_window_touches(acquired_at: datetime, window: timedelta) -> list[date]:
    """One archive per day, and a window is not aligned to one.

    An acquisition ten minutes after midnight has most of its window in the previous day's file.
    Opening only the day of the acquisition would halve what every vessel is placed from, without
    saying so.
    """
    first = (acquired_at - window).date()
    last = (acquired_at + window).date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


@dataclass(frozen=True)
class _Surviving:
    """One chunk of the archive after the filters, and what the chunk cost to get there."""

    reports: pd.DataFrame
    read: int
    no_position: int
    no_timestamp: int


def read_day(archive: Archive, day: date, columns: dict[str, str]) -> Iterator[pd.DataFrame]:
    """A day's archive, in pieces small enough to hold, with `columns` renamed as they are read.

    A generator rather than a list, which is the whole point: the day is inflated, parsed,
    filtered and discarded a chunk at a time, and collecting the chunks first would put all
    3.3 GB of the file in memory to save none of it.

    Read as text throughout. The archive writes a blank where a field is unavailable and a
    sentinel where a position is, and a parser told to expect numbers turns the first into an
    error and the second into a coordinate; both are decided by the caller, where they can be
    counted.

    `columns` is a parameter because two readers want different subsets of the same file: the
    ingestion takes the five that place a report, and `survey.py` also wants the speed, which
    only means something when the question is whether a vessel was moving. One reader with a
    parameter, rather than two that could drift apart on chunk size or on how text is parsed.
    """
    with archive.open_day(day) as stream:
        for chunk in pd.read_csv(
            stream,
            usecols=list(columns),
            dtype="string",
            chunksize=CHUNK_ROWS,
        ):
            yield chunk.rename(columns=columns)


def _placed(chunk: pd.DataFrame, area: Bounds) -> _Surviving:
    """The reports in this chunk that stand inside the area, with their timestamps parsed.

    Everything that depends on the day's file and not on which acquisition is being asked about,
    done once: a day serving three scenes parses its timestamps once rather than three times.
    What comes back still holds rows whose timestamp could not be read — `_within_window` drops
    them, and they are counted here where the parse happened.

    Position first, because it is a numeric comparison over every row of the archive and the
    timestamps are the expensive parse. What survives is a thousandth of what arrives.
    """
    reports = chunk
    # Down to plain floats rather than the nullable kind the text columns produce: a comparison
    # against a missing value there is missing rather than false, and a row counted as neither
    # placeable nor unplaceable is a row that leaves the total not adding up.
    lat = pd.to_numeric(reports["lat"], errors="coerce").astype("float64")
    lon = pd.to_numeric(reports["lon"], errors="coerce").astype("float64")

    # 91 and 181 are what the archive writes when a position is unavailable, and read as numbers
    # they are coordinates rather than absences.
    placeable = lat.between(-90, 90) & lon.between(-180, 180)
    inside = placeable & lat.between(area.south, area.north) & lon.between(area.west, area.east)

    # Blank on most rows, because length arrives in a static message the receiver merges in only
    # when it has one. Absent, not zero: a vessel that never said how big it is has an unknown
    # length, and reading the blank as a number would make it the smallest thing on the water.
    length = pd.to_numeric(reports["length_m"], errors="coerce").astype("float64")

    here = reports[inside].assign(lat=lat[inside], lon=lon[inside], length_m=length[inside])
    when = pd.to_datetime(here["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce", utc=True)

    return _Surviving(
        reports=here.assign(timestamp=when),
        read=len(chunk),
        no_position=int((~placeable).sum()),
        # A report whose timestamp cannot be read falls outside every window in both directions
        # and would leave without a word. Counted here rather than allowed to: a declaration that
        # disappears on its way to the matching is a detection published as a dark vessel, and one
        # that disappears without being counted cannot be found afterwards either.
        no_timestamp=int(when.isna().sum()),
    )


def _within_window(placed: _Surviving, acquired_at: datetime, window: timedelta) -> _Surviving:
    """The reports of an already-placed chunk that fall inside one acquisition's window.

    The counts pass through unchanged. They are properties of the chunk the day gave — how many
    rows it held, how many carried no position, how many no readable timestamp — and every
    acquisition this day serves has to answer for the same file.
    """
    when = placed.reports["timestamp"]
    in_window = when.between(acquired_at - window, acquired_at + window)

    return _Surviving(
        reports=placed.reports[in_window],
        read=placed.read,
        no_position=placed.no_position,
        no_timestamp=placed.no_timestamp,
    )


def _cleaned(
    reports: pd.DataFrame,
    read: int,
    no_position: int,
    no_timestamp: int,
    max_speed_kn: float,
) -> tuple[gpd.GeoDataFrame, Cleaning]:
    """Every rule, in the order that lets the next one mean something.

    Identifiers before duplicates, because a duplicate is a repetition of one vessel's report and
    without an identifier there is no vessel. Duplicates and contradictions before the outlier
    rule, because that rule compares each report against the median of its vessel's own track: a
    report the archive wrote out three times counts three times towards where the median sits,
    and a contradictory pair drags it to a midpoint the vessel never occupied.
    """
    in_area_and_window = len(reports)

    reports = reports.sort_values(["mmsi", "timestamp"], kind="stable").reset_index(drop=True)

    afloat = reports["mobile"].isin(VESSEL_CLASSES)
    reports, not_a_vessel = reports[afloat], int((~afloat).sum())

    identified = _identified(reports)
    reports, missing_identifier = reports[identified], int((~identified).sum())

    # Length is a property of the vessel, not of the row that happened to carry it, so it is
    # spread across the vessel's own reports as soon as there is a vessel to spread it across.
    # Before the duplicate rule rather than after: two identical rows differ only in whether the
    # receiver had the static data yet, and whichever the archive wrote first would otherwise
    # decide whether the ship has a size at all.
    reports = reports.assign(
        length_m=reports.groupby("mmsi", sort=False, observed=True)["length_m"].transform("max")
    )

    repeated = reports.duplicated(subset=["mmsi", "timestamp", "lat", "lon"], keep="first")
    reports, duplicated = reports[~repeated], int(repeated.sum())

    # Two different positions for one vessel at one instant. At most one of them is true and
    # nothing here says which, so both go and the vessel keeps whatever else it reported.
    conflicting = reports.duplicated(subset=["mmsi", "timestamp"], keep=False)
    reports, contradictory = reports[~conflicting], int(conflicting.sum())

    stranded = _beyond_reach(reports, max_speed_kn)
    reports, implausible_jump = reports[~stranded], int(stranded.sum())

    return _positions(reports), Cleaning(
        read=read,
        no_position=no_position,
        no_timestamp=no_timestamp,
        in_area_and_window=in_area_and_window,
        not_a_vessel=not_a_vessel,
        missing_identifier=missing_identifier,
        duplicated=duplicated,
        contradictory=contradictory,
        implausible_jump=implausible_jump,
        kept=len(reports),
    )


def _identified(reports: pd.DataFrame) -> pd.Series:
    """Which reports carry a nine-digit identifier, which is what an MMSI is.

    The real archive carries four-, seven- and eight-digit identifiers alongside them, and pooling
    those under one key would draw a track between two different ships — the decision
    `interpolate.py` refuses to make and names the ingestion as the place for.
    """
    return reports["mmsi"].str.fullmatch(r"\d{9}").fillna(False).astype(bool)


def from_a_vessel(reports: pd.DataFrame) -> pd.Series:
    """Which reports come from an identified vessel rather than from something else afloat.

    The two rules the ingestion applies first, as one, for the survey to apply too. Not the whole
    cleaning: the ingestion counts what each rule removed, because a slice is a claim about which
    vessels declared themselves and is only as good as what was thrown away on the way to it. A
    survey of where traffic is makes no such claim, and needs the same two rules for a different
    reason — a base station never moves and a four-digit identifier is several ships pooled, and
    a rectangle chosen for either is a rectangle chosen for the shore beside it.
    """
    return reports["mobile"].isin(VESSEL_CLASSES) & _identified(reports)


def _beyond_reach(reports: pd.DataFrame, max_speed_kn: float) -> pd.Series:
    """Positions no vessel could have reached from where the rest of its own reports put it.

    Judged against the middle of the vessel's own track — the median of its positions, at the
    median of its report times. Two obvious rules fail here and both fail quietly. Walking a
    track forward and dropping whatever the last kept report cannot reach anchors the whole track
    on its first position, so one bad report at the start takes the vessel's whole slice with it.
    Judging a report against its immediate neighbours cannot tell a spurious report from a good
    one whose only neighbour is spurious — a jump between three reports takes out all three, two
    of which were the evidence. A median moves for no single report, which is the property both
    of those lack.

    **The reach is per report, and that is what makes the rule bite.** An earlier version allowed
    every report the whole window's worth of travel — 55 km at the shipped settings, against a
    searched area whose diagonal is 35 km, so nothing inside it could ever exceed the ceiling and
    the rule was dead. What a vessel could have covered is the gap between *that report's* time
    and the middle of its track, which for a spurious position among dense reports is seconds.
    A report far from the track in space but next to it in time is exactly what a jump is.

    The floor is what stops the rule from firing on the accuracy of AIS itself: two reports a
    second apart differ by tens of metres from a vessel standing still, and without it the reach
    near the middle of a track shrinks below that noise. It is set at the order of the match
    tolerance, because a report displaced by less than that radius cannot change any verdict —
    so there is nothing to gain by removing it, and a declaration wrongly removed is a detection
    published as a dark vessel.

    A vessel that reported once is its own median, at its own instant, and stands zero metres
    from it. One report is a usable declaration — it is what `interpolate.py` falls back to and
    marks `reported` — and dropping it would publish whatever it explains as dark.
    """
    if reports.empty:
        return pd.Series(False, index=reports.index, dtype=bool)

    track = reports.groupby("mmsi", sort=False)
    centre = track[["lon", "lat"]].transform("median")
    middle = track["timestamp"].transform("median")

    _, _, metres = _WGS84.inv(
        centre["lon"].to_numpy(),
        centre["lat"].to_numpy(),
        reports["lon"].to_numpy(),
        reports["lat"].to_numpy(),
    )
    seconds = (reports["timestamp"] - middle).abs().dt.total_seconds().to_numpy()
    reach_m = POSITION_ACCURACY_M + max_speed_kn * _KNOT_MS * seconds

    return pd.Series(metres > reach_m, index=reports.index)


def _positions(reports: pd.DataFrame) -> gpd.GeoDataFrame:
    """The surviving reports as points, in the CRS the archive expresses them in.

    Not the working CRS: putting the slice there is `load_ais`'s job, and doing it in both places
    is how a reprojection gets applied to coordinates already in metres. Columns are named rather
    than inferred, so a slice a filter emptied comes back as an empty answer of the right shape.
    """
    frame = reports.reindex(columns=["mmsi", "timestamp", "length_m", "lon", "lat"]).reset_index(
        drop=True
    )
    return gpd.GeoDataFrame(
        frame.drop(columns=["lon", "lat"]).astype({"mmsi": "string", "length_m": "float64"}),
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs=AIS_CRS,
    )


def write_ais(reports: gpd.GeoDataFrame, path: Path) -> None:
    """Write a slice as the CSV `load_ais` reads.

    The archive is fetched once and the chain runs from the result as often as it likes, with no
    network — the same split as `export` and `run`. Timestamps go out in ISO 8601 with their
    zone, because a zone dropped here is the two-hour error this module reads the archive to
    avoid, reintroduced on the way out.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "mmsi": reports["mmsi"],
            "timestamp": pd.to_datetime(reports["timestamp"], utc=True).map(
                lambda moment: moment.isoformat()
            ),
            "length_m": reports["length_m"],
            "lon": reports.geometry.x,
            "lat": reports.geometry.y,
        }
    ).to_csv(path, index=False)


def load_ais(path: Path, crs: str) -> gpd.GeoDataFrame:
    """Read position reports from CSV and return them in `crs`.

    Expects columns `mmsi`, `timestamp`, `length_m`, `lon`, `lat`. MMSI is kept as text: it is an
    identifier, never a quantity, and reading it as a number invites a leading zero to be lost
    or a missing value to turn it into a float. `length_m` is a float and is often missing, which
    is a vessel that never said how big it is rather than a vessel of no size.
    """
    reports = pd.read_csv(path, dtype={"mmsi": "string"})
    reports["timestamp"] = pd.to_datetime(reports["timestamp"], utc=True, format="ISO8601")

    return gpd.GeoDataFrame(
        reports.drop(columns=["lon", "lat"]),
        geometry=gpd.points_from_xy(reports["lon"], reports["lat"]),
        crs=AIS_CRS,
    ).to_crs(crs)
