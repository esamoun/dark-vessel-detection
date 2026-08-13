"""Choosing where to look, from the archive rather than from a map.

The study area decides what every level above it can possibly find. Anholt was picked for its
wind farm and turned out to sit off the shipping lane, so the fusion level ran correctly over
water that had nothing in it to fuse — thirty acquisitions, no declared vessel longer than 15 m
in any of them. The rectangle is measured now, and this file tests the measurement.

What it has to get right is what separates a lane from an anchorage, and both mistakes are
quiet. Report counts make a vessel that sat still all day look like a hundred ships. Vessels
counted over a whole day make a rectangle one ship crosses at dawn look like one that always has
traffic in it. A harbour scores higher than either on both. The measure here is the number of
distinct vessels, long enough for the radar to resolve and under way rather than moored, standing
inside the rectangle in a window the width of the one the ingestion actually fetches.

The archive is injected, as it is in `test_ais.py`: 662 MB of Danish AIS is not something a test
suite downloads.
"""

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from darkvessel.cli import survey_request_from
from darkvessel.data.area import Bounds
from darkvessel.data.dma import Archive
from darkvessel.data.survey import Candidate, survey

DAY = date(2026, 8, 9)

# A region four steps wide and four steps high, so a two-by-two rectangle can stand in nine
# places inside it and the tests can say which one the measurement picked.
REGION = Bounds(west=11.0, south=57.0, east=11.4, north=57.4)
BOX = (0.2, 0.2)
STRIDE = 0.1

WINDOW = timedelta(minutes=30)
MIN_LENGTH_M = 100.0
UNDER_WAY_KN = 3.0

# Two places to put a vessel: the south-west corner of the region, and the north-east one. A
# rectangle covering one cannot cover the other.
SOUTH_WEST = (11.05, 57.05)
NORTH_EAST = (11.35, 57.35)

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "survey.yaml"

HEADER = (
    "# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,SOG,Length,Destination"
)

Row = tuple[str, str, str, float, float, float | str, float | str]


def report(
    mmsi: str = "219000001",
    at: tuple[float, float] = SOUTH_WEST,
    when: str = "00:10:00",
    knots: float | str = 12.0,
    length: float | str = 228.0,
    mobile: str = "Class A",
) -> Row:
    """One position report, as the Danish archive writes it."""
    lon, lat = at
    return (f"{DAY.strftime('%d/%m/%Y')} {when}", mobile, mmsi, lat, lon, knots, length)


@dataclass
class FakeArchive(Archive):
    """The Danish Maritime Authority, minus the 662 MB. Filters nothing, cleans nothing."""

    days: dict[date, str]
    opened: list[date] = field(default_factory=list)

    @contextmanager
    def open_day(self, day: date) -> Iterator[io.BytesIO]:
        self.opened.append(day)
        if day not in self.days:
            raise FileNotFoundError(f"the fake archive has no {day}")
        yield io.BytesIO(self.days[day].encode())


def one_day(reports: list[Row]) -> FakeArchive:
    lines = [HEADER]
    for when, mobile, mmsi, lat, lon, knots, length in reports:
        lines.append(
            f"{when},{mobile},{mmsi},{lat},{lon},Under way using engine,{knots},{length},X"
        )
    return FakeArchive(days={DAY: "\n".join(lines) + "\n"})


def measure(
    archive: Archive,
    min_length_m: float = MIN_LENGTH_M,
    under_way_kn: float = UNDER_WAY_KN,
    box: tuple[float, float] = BOX,
    stride: float = STRIDE,
) -> list[Candidate]:
    return survey(
        archive=archive,
        day=DAY,
        region=REGION,
        box=box,
        stride=stride,
        window=WINDOW,
        min_length_m=min_length_m,
        under_way_kn=under_way_kn,
    )


def candidate_at(candidates: list[Candidate], west: float, south: float) -> Candidate:
    """The one candidate rectangle standing at the given south-west corner."""
    here = [
        candidate
        for candidate in candidates
        if candidate.bounds.west == pytest.approx(west)
        and candidate.bounds.south == pytest.approx(south)
    ]
    assert len(here) == 1, f"expected one rectangle at ({west}, {south}), found {len(here)}"
    return here[0]


def test_the_rectangle_with_the_most_traffic_in_it_is_the_one_offered_first() -> None:
    """The whole point: the choice comes out of the archive, in an order someone can act on."""
    archive = one_day(
        [report(mmsi=f"21900000{n}", at=NORTH_EAST, when="00:10:00") for n in range(1, 5)]
        + [report(mmsi="219000009", at=SOUTH_WEST, when="00:10:00")]
    )

    candidates = measure(archive)

    assert candidates[0].bounds == Bounds(west=11.2, south=57.2, east=11.4, north=57.4)
    assert candidates[0].vessels == 4


def test_a_vessel_reporting_all_window_long_is_still_one_vessel() -> None:
    """Report counts are what a harbour wins on.

    A vessel alongside reports every three minutes and never moves; one crossing at 14 knots
    reports every two seconds and is gone in twenty minutes. Counting rows measures dwell and
    transmit rate, neither of which is traffic.
    """
    archive = one_day(
        [report(when=f"00:{minute:02d}:00") for minute in range(0, 30, 2)]
        + [report(mmsi="219000002", at=NORTH_EAST)]
    )

    candidates = measure(archive)

    assert candidate_at(candidates, 11.0, 57.0).vessels == 1
    assert candidate_at(candidates, 11.0, 57.0).mean_in_window == pytest.approx(1 / 48)


def test_a_vessel_at_anchor_is_not_traffic() -> None:
    """The measurement that moved the study area, and the one an anchorage would have broken.

    The busiest rectangles in the northern Kattegat by any count of vessels present are the
    Frederikshavn approach and the Skagen anchorage, where twenty ships of 200 m sit waiting. They
    are large, they declare themselves, and a scene over them would match plenty — while standing
    beside a harbour, with land in the frame, over water no lane runs through.
    """
    archive = one_day(
        [
            report(mmsi="219000001", at=SOUTH_WEST, knots=0.1),
            report(mmsi="219000002", at=SOUTH_WEST, knots=0.0),
            report(mmsi="219000003", at=NORTH_EAST, knots=11.0),
        ]
    )

    candidates = measure(archive)

    assert candidate_at(candidates, 11.0, 57.0).vessels == 0
    assert candidate_at(candidates, 11.2, 57.2).vessels == 1


def test_a_vessel_the_radar_could_not_resolve_is_not_counted() -> None:
    """Anholt's failure, stated as a rule.

    Every declared vessel that crossed the old box over five weeks was a sailing boat or a
    pleasure craft, the largest of them 15 m. At 10 m pixels that is a pixel and a half, so a
    rectangle chosen for how many of them cross it would be chosen on evidence the radar cannot
    see either way.
    """
    archive = one_day(
        [
            report(mmsi="219000001", at=SOUTH_WEST, length=15.0),
            report(mmsi="219000002", at=NORTH_EAST, length=228.0),
        ]
    )

    candidates = measure(archive)

    assert candidate_at(candidates, 11.0, 57.0).vessels == 0
    assert candidate_at(candidates, 11.2, 57.2).vessels == 1


def test_a_vessel_that_never_declared_a_size_is_not_counted_as_a_large_one() -> None:
    """Unknown is not large, and the count understates rather than overstates because of it.

    The two errors are not the same. A rectangle credited with traffic it does not have is chosen,
    exported and run before anyone finds out — which is the mistake this whole ticket exists to
    undo. A rectangle whose count is short is passed over in favour of one whose evidence is
    complete, and the worst that costs is a second-best rectangle.
    """
    archive = one_day(
        [
            report(mmsi="219000001", at=SOUTH_WEST, length=""),
            report(mmsi="219000002", at=NORTH_EAST, length=228.0),
        ]
    )

    candidates = measure(archive)

    assert candidate_at(candidates, 11.0, 57.0).vessels == 0
    assert candidate_at(candidates, 11.2, 57.2).vessels == 1


def test_a_length_declared_on_one_report_counts_for_the_whole_vessel() -> None:
    # The archive merges static data into position rows only when the receiver has it, so a
    # vessel declares its length on a few of its rows and leaves the field blank on the rest.
    archive = one_day(
        [
            report(when="00:10:00", length=""),
            report(when="00:12:00", length=228.0),
            report(when="00:14:00", length=""),
        ]
    )

    assert candidate_at(measure(archive), 11.0, 57.0).vessels == 1


def test_a_base_station_is_not_traffic() -> None:
    # The archive carries every transmitter in Danish waters. A base station stands on land and
    # never moves, and one counted as a vessel is a rectangle chosen for the shore beside it.
    archive = one_day([report(at=SOUTH_WEST, mobile="Base Station")])

    assert candidate_at(measure(archive), 11.0, 57.0).vessels == 0


def test_a_vessel_standing_in_two_cells_of_one_rectangle_is_counted_once() -> None:
    """Candidate rectangles overlap, and inside one of them the cells do too.

    Rectangles are placed a step apart and are several steps across, so a vessel crossing a cell
    boundary stands in two of the cells a single rectangle is scored from. Counted per cell and
    summed, one ship becomes two — and a rectangle straddling a lane would be the one that wins.
    """
    # 11.099 and 11.101 sit either side of the step boundary at 11.1, inside the same rectangle.
    archive = one_day(
        [
            report(when="00:10:00", at=(11.099, 57.05)),
            report(when="00:12:00", at=(11.101, 57.05)),
        ]
    )

    here = candidate_at(measure(archive), 11.0, 57.0)

    assert here.vessels == 1
    assert here.mean_in_window == pytest.approx(1 / 48)


def test_a_window_with_nothing_in_the_rectangle_counts_as_nothing() -> None:
    """A rectangle one ship crosses at dawn is not a rectangle with traffic in it.

    An acquisition arrives at a moment nobody chose, so what matters is the number standing there
    at an arbitrary one. Averaged over only the windows that had something in them, a rectangle
    crossed once a day scores the same as a lane — which is exactly the mistake that put the study
    area off the lane in the first place.
    """
    archive = one_day([report(when="00:10:00"), report(mmsi="219000002", when="00:20:00")])

    here = candidate_at(measure(archive), 11.0, 57.0)

    # Two vessels, both inside the day's first half-hour window, and 47 windows with nothing.
    assert here.vessels == 2
    assert here.mean_in_window == pytest.approx(2 / 48)
    assert here.fewest_in_window == 0
    assert here.empty_windows == 47


def test_a_rectangle_that_is_not_a_whole_number_of_steps_across_is_refused() -> None:
    """The cells the score is summed over have to tile the rectangle being scored.

    Off by a fraction of a step, the rectangle offered is not the rectangle measured — the count
    belongs to a slightly different box, and nothing about the answer says so.
    """
    with pytest.raises(ValueError, match="whole number of steps"):
        measure(one_day([report()]), box=(0.25, 0.2))


def test_every_rectangle_offered_fits_inside_the_region_it_was_asked_for() -> None:
    # A rectangle hanging over the edge is measured against reports the survey never read, so
    # its count is short by however much of it lies outside.
    candidates = measure(one_day([report()]))

    assert len(candidates) == 9
    for candidate in candidates:
        assert candidate.bounds.east <= REGION.east + 1e-9
        assert candidate.bounds.north <= REGION.north + 1e-9


def test_a_day_with_nothing_in_it_is_an_answer_and_not_an_error() -> None:
    # The same rule the ingestion follows: a search that ran and found nothing is a result.
    candidates = measure(one_day([]))

    assert len(candidates) == 9
    assert all(candidate.vessels == 0 for candidate in candidates)


def test_the_shipped_survey_config_describes_a_measurement_the_command_can_run() -> None:
    """`configs/survey.yaml` through the command's own parsing, minus the 662 MB.

    The same gap `export_request_from` and `ais_request_from` exist to close: a mistyped key in a
    config that needs the network would otherwise surface to someone who had already waited for a
    day of Danish AIS to come down the wire.
    """
    config = yaml.safe_load(SHIPPED_CONFIG.read_text())

    request = survey_request_from(config)

    assert isinstance(request["day"], date)
    assert request["report"] > 0
    assert request["region"].east > request["region"].west
    assert request["window"] > timedelta(0)
    assert request["min_length_m"] > 0.0
    # The rectangle being searched for is the size of the study area a config would then declare,
    # or the measurement is about a box nobody is going to export.
    lon_deg, lat_deg = request["box"]
    assert lon_deg > 0.0 and lat_deg > 0.0


def test_the_measurement_is_reported_in_the_order_it_ranks_them() -> None:
    """A survey is read by whoever has to choose, and the first row is the recommendation.

    Ranked on the worst window before the mean: a rectangle that is empty at some point in the
    day is a rectangle an acquisition can catch empty, and no average makes that acceptable.
    """
    archive = one_day(
        [
            # Four vessels crossing the north-east rectangle, all in one window.
            *[report(mmsi=f"21900000{n}", at=NORTH_EAST, when="00:10:00") for n in range(1, 5)],
            # One vessel in the south-west rectangle, present in every window of the day.
            *[
                report(mmsi="219000009", at=SOUTH_WEST, when=f"{hour:02d}:{minute:02d}:00")
                for hour in range(24)
                for minute in (10, 40)
            ],
        ]
    )

    candidates = measure(archive)

    assert candidates[0].bounds.west == pytest.approx(11.0)
    assert candidates[0].fewest_in_window == 1
    assert candidates[0].empty_windows == 0


def test_the_survey_opens_the_day_it_was_asked_for_and_no_other() -> None:
    # A day of Danish AIS is 662 MB across the wire. Opening a second one by accident is not a
    # correctness fault, it is forty seconds and most of a gigabyte.
    archive = one_day([report()])

    measure(archive)

    assert archive.opened == [DAY]


def test_a_report_outside_the_region_is_never_read_into_a_rectangle() -> None:
    archive = one_day([report(mmsi="219000001"), report(mmsi="219000002", at=(12.6, 55.7))])

    candidates = measure(archive)

    assert sum(candidate.vessels for candidate in candidates) == 1


def test_an_acquisition_is_what_the_windows_stand_for() -> None:
    """The window is the ingestion's own, so the number means what someone will act on.

    `darkvessel ais` fetches a window either side of the acquisition and the chain matches
    against what stands in it. Measuring over some other span would produce a ranking about a
    slice nobody fetches.
    """
    archive = one_day([report(when="00:10:00"), report(mmsi="219000002", when="00:40:00")])

    here = candidate_at(
        survey(
            archive=archive,
            day=DAY,
            region=REGION,
            box=BOX,
            stride=STRIDE,
            window=timedelta(hours=1),
            min_length_m=MIN_LENGTH_M,
            under_way_kn=UNDER_WAY_KN,
        ),
        11.0,
        57.0,
    )

    # Both vessels fall inside the first hour of the day, and the day holds 24 such windows.
    assert here.mean_in_window == pytest.approx(2 / 24)
