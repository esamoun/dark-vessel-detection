"""Ingesting a real Danish archive, with the archive injected.

The Danish Maritime Authority publishes one zip per day. The one this project's first real scene
falls in is 662 MB compressed and 3.3 GB of CSV, so the network is the second thing here that
cannot be asserted — and it is handled the same way as Earth Engine: the archive arrives as a
parameter, and a fake hands back a few rows written in the format the real one uses.

What is tested is everything that stays wrong silently. A report that never reaches the matching
is a detection published as dark, so the filters are tested for what they let through as much as
for what they remove. A report that reaches it carrying a position the vessel was never at is a
match that explains nothing, so each cleaning rule is tested on the shape of noise it exists to
remove — every one of which was found in the first 1.18 million rows of the real archive.

What is deliberately not tested: whether the archive's timestamps mean what this package reads
them as. That is a claim about a file on someone else's server, and it is checked against the
first real run and recorded in the README rather than pretended at here.
"""

import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pytest
import yaml

from darkvessel.cli import ais_request_from
from darkvessel.data.ais import Cleaning, load_ais, slice_for, write_ais
from darkvessel.data.area import Bounds
from darkvessel.data.dma import Archive, archive_url, zip_member

WORKING_CRS = "EPSG:25832"

# An acquisition and the area `configs/anholt.yaml` cuts a scene to. The date is a fixture's
# date, not a claim about which scene is shipped — that is `configs/anholt.yaml`'s to make.
ACQUIRED_AT = datetime(2026, 7, 2, 17, 0, 36, tzinfo=UTC)
ANHOLT = Bounds(west=11.15, south=56.58, east=11.40, north=56.71)

# A point comfortably inside the area, and one comfortably outside it.
INSIDE = (11.28, 56.64)
FAR_AWAY = (12.60, 55.70)  # the Øresund, some 120 km south-east

WINDOW = timedelta(minutes=15)
MARGIN_M = 5_000.0
MAX_SPEED_KN = 60.0

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "anholt.yaml"

# Five of the archive's twenty-six columns are read. The other two here are carried so that a
# fixture cannot pass by being narrower than the file it stands for.
HEADER = "# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,SOG,Destination"


def archive_csv(reports: list[tuple[str, str, str, float | str, float | str]]) -> str:
    """Rows as the Danish archive writes them: (timestamp, mobile, mmsi, lat, lon)."""
    lines = [HEADER]
    for when, mobile, mmsi, lat, lon in reports:
        lines.append(f"{when},{mobile},{mmsi},{lat},{lon},Under way using engine,7.4,ANHOLT")
    return "\n".join(lines) + "\n"


def report(
    mmsi: str = "219000001",
    age: timedelta = timedelta(0),
    at: tuple[float, float] = INSIDE,
    mobile: str = "Class A",
) -> tuple[str, str, str, float | str, float | str]:
    """One position report, placed relative to the acquisition."""
    lon, lat = at
    when = (ACQUIRED_AT + age).strftime("%d/%m/%Y %H:%M:%S")
    return (when, mobile, mmsi, lat, lon)


@dataclass
class FakeArchive(Archive):
    """The Danish Maritime Authority, minus the 662 MB.

    It filters nothing and cleans nothing: those are this package's job, and a fake that did
    either would only ever agree with the code under test.
    """

    days: dict[date, str]
    opened: list[date] = field(default_factory=list)

    @contextmanager
    def open_day(self, day: date) -> Iterator[io.BytesIO]:
        self.opened.append(day)
        if day not in self.days:
            raise FileNotFoundError(f"the fake archive has no {day}")
        yield io.BytesIO(self.days[day].encode())


def ingest(
    archive: Archive,
    window: timedelta = WINDOW,
    margin_m: float = MARGIN_M,
    max_speed_kn: float = MAX_SPEED_KN,
) -> tuple[gpd.GeoDataFrame, Cleaning]:
    return slice_for(
        archive=archive,
        acquired_at=ACQUIRED_AT,
        area=ANHOLT,
        window=window,
        margin_m=margin_m,
        max_speed_kn=max_speed_kn,
    )


def one_day(reports: list[tuple[str, str, str, float | str, float | str]]) -> FakeArchive:
    return FakeArchive(days={ACQUIRED_AT.date(): archive_csv(reports)})


def mmsis(reports: gpd.GeoDataFrame) -> set[str]:
    return set(reports["mmsi"])


def test_a_report_in_the_area_and_the_window_reaches_the_matching() -> None:
    archive = one_day([report(age=timedelta(minutes=-1))])

    reports, cleaning = ingest(archive)

    assert mmsis(reports) == {"219000001"}
    assert cleaning.kept == 1
    # In WGS84, as the archive gives it: putting the slice in the working CRS is `load_ais`'s
    # job, and doing it twice is how a reprojection gets applied to already-projected metres.
    assert reports.crs == "EPSG:4326"


def test_a_report_outside_the_study_area_never_reaches_the_matching() -> None:
    # Quadratic matching against a whole day of Danish waters is the smaller reason. The real one
    # is that a vessel 120 km away cannot explain a detection here, and every declaration that
    # can is what the dark verdict rests on.
    archive = one_day([report(mmsi="219000001"), report(mmsi="219000002", at=FAR_AWAY)])

    reports, cleaning = ingest(archive)

    assert mmsis(reports) == {"219000001"}
    assert cleaning.in_area_and_window == 1


def test_a_report_outside_the_window_never_reaches_the_matching() -> None:
    archive = one_day(
        [report(age=timedelta(minutes=-1)), report(mmsi="219000002", age=timedelta(hours=6))]
    )

    reports, _ = ingest(archive)

    assert mmsis(reports) == {"219000001"}


def test_a_vessel_at_the_edge_of_the_scene_keeps_the_reports_either_side_of_it() -> None:
    """The filter is on the area grown by a margin, and this is what the margin is for.

    A vessel imaged just inside the eastern edge reported a few minutes earlier from just
    outside it. Filtered to the area exactly, that report is gone and the vessel has nothing to
    interpolate between — so it is placed at whichever report survived, and the failure this
    level exists to remove comes back through the ingestion instead of through the matching.
    """
    just_outside = (ANHOLT.east + 0.02, 56.64)  # about 1.2 km east of the area
    archive = one_day(
        [
            report(age=timedelta(minutes=-2), at=just_outside),
            report(age=timedelta(minutes=2), at=(ANHOLT.east - 0.01, 56.64)),
        ]
    )

    reports, _ = ingest(archive)

    assert len(reports) == 2


def test_a_window_that_reaches_into_yesterday_opens_yesterdays_archive() -> None:
    # One archive per day, and a window is not aligned to one. An acquisition ten minutes after
    # midnight has most of its window in the previous file, and taking only the day of the
    # acquisition would silently halve what every vessel is placed from.
    just_after_midnight = datetime(2026, 7, 2, 0, 10, tzinfo=UTC)
    archive = FakeArchive(
        days={date(2026, 7, 1): archive_csv([]), date(2026, 7, 2): archive_csv([])}
    )

    slice_for(
        archive=archive,
        acquired_at=just_after_midnight,
        area=ANHOLT,
        window=WINDOW,
        margin_m=MARGIN_M,
        max_speed_kn=MAX_SPEED_KN,
    )

    assert archive.opened == [date(2026, 7, 1), date(2026, 7, 2)]


def test_a_window_inside_one_day_opens_one_archive() -> None:
    archive = one_day([report()])

    ingest(archive)

    assert archive.opened == [ACQUIRED_AT.date()]


def test_a_base_station_is_not_a_vessel_and_cannot_explain_a_detection() -> None:
    """The archive carries every transmitter in Danish waters, not every ship.

    Of the first 1.18 million rows of the real archive for this acquisition, 83 192 were base
    stations and 26 896 aids to navigation. Both are real transmitters at real positions and
    neither is a vessel, so neither can turn a detection into a declared one. A buoy explaining
    a radar target would read, in the output, exactly like a ship that declared itself.
    """
    archive = one_day(
        [
            report(mmsi="219000001", mobile="Class A"),
            report(mmsi="219000002", mobile="Class B"),
            report(mmsi="002190064", mobile="Base Station"),
            report(mmsi="992191234", mobile="AtoN"),
        ]
    )

    reports, cleaning = ingest(archive)

    assert mmsis(reports) == {"219000001", "219000002"}
    assert cleaning.not_a_vessel == 2


def test_a_report_with_no_usable_position_is_counted_rather_than_carried() -> None:
    # 91 and 181 are what the archive writes when a position is unavailable; 5024 of the first
    # 1.18 million rows carried one. Read as a number, 91 degrees north is a coordinate.
    archive = one_day(
        [
            report(mmsi="219000001"),
            report(mmsi="219000002", at=(181.0, 91.0)),
            report(mmsi="219000003", at=("", "")),
        ]
    )

    reports, cleaning = ingest(archive)

    assert mmsis(reports) == {"219000001"}
    assert cleaning.no_position == 2


def test_a_report_inside_the_area_with_no_readable_timestamp_is_counted() -> None:
    """It falls outside the window in both directions, and would leave without a word.

    `interpolate.py` refuses such a report rather than let it fall out of the bracket quietly,
    and names the ingestion as the place to deal with it. Dealing with it means saying how many
    there were: a declaration that disappears is a detection published as dark, and one that
    disappears uncounted cannot be found afterwards either.
    """
    archive = one_day(
        [
            report(mmsi="219000001", age=timedelta(minutes=-1)),
            ("2026-07-02 17:00:00", "Class A", "219000002", 56.64, 11.28),
        ]
    )

    reports, cleaning = ingest(archive)

    assert mmsis(reports) == {"219000001"}
    assert cleaning.no_timestamp == 1
    assert cleaning.in_area_and_window == 1


def test_a_report_without_a_nine_digit_identifier_is_removed_rather_than_pooled() -> None:
    """An MMSI is nine digits, and the real archive carries plenty that are not.

    Pooling them under one key would be worse than dropping them: two unidentified reports are
    two different ships, and a line drawn between them is a track that never existed. This is
    the decision `interpolate.py` refuses to make and names the ingestion as the place for.
    """
    archive = one_day(
        [
            report(mmsi="219000001"),
            report(mmsi="2190064"),
            report(mmsi=""),
        ]
    )

    reports, cleaning = ingest(archive)

    assert mmsis(reports) == {"219000001"}
    assert cleaning.missing_identifier == 2


def test_the_same_report_twice_is_one_report() -> None:
    # The archive repeats itself: two of the first five rows of the real file are the same
    # vessel, instant and position written twice.
    archive = one_day(
        [
            report(age=timedelta(minutes=-1)),
            report(age=timedelta(minutes=-1)),
            report(age=timedelta(minutes=1)),
        ]
    )

    reports, cleaning = ingest(archive)

    assert len(reports) == 2
    assert cleaning.duplicated == 1


def test_two_positions_for_one_vessel_at_one_instant_are_both_dropped() -> None:
    """Not a duplicate: a contradiction, and nothing here says which half of it is true.

    Keeping either would be a coin toss that the output presents as an observation. Both go, and
    the vessel keeps whatever else it reported — it stays in the search, so a detection it
    explains is not published as dark.
    """
    archive = one_day(
        [
            report(age=timedelta(minutes=-1), at=(11.28, 56.64)),
            report(age=timedelta(minutes=-1), at=(11.29, 56.65)),
            report(age=timedelta(minutes=1)),
        ]
    )

    reports, cleaning = ingest(archive)

    assert len(reports) == 1
    assert cleaning.contradictory == 2


def test_a_position_the_rest_of_the_track_cannot_reach_is_removed() -> None:
    """A vessel does not cover 8 km and come back inside two minutes.

    The middle report is 8 km from the other two, which are 100 m apart. Left in, it is what the
    vessel is interpolated from and the position it produces is nowhere the ship has ever been —
    a match that explains nothing, or a declared vessel published as dark.

    The two good reports have to survive it, which is the part that is easy to get wrong: judged
    against its immediate neighbour, each of them is as unreachable from the bad report as the
    bad report is from it, and a rule that cannot tell them apart removes the evidence along with
    the noise.
    """
    # Every position here is inside the area the shipped margin searches, so the rule has to do
    # the work rather than the spatial filter. That is the case an earlier version could not
    # reach: it allowed each report the whole window's worth of travel, 55 km against a searched
    # area 35 km across, so nothing that survived the filter could ever exceed the ceiling.
    archive = one_day(
        [
            report(age=timedelta(minutes=-1), at=(11.20, 56.640)),
            report(age=timedelta(0), at=(11.33, 56.640)),  # 8 km east, at the middle of the track
            report(age=timedelta(minutes=1), at=(11.20, 56.641)),
        ]
    )

    reports, cleaning = ingest(archive)

    assert len(reports) == 2
    assert cleaning.implausible_jump == 1
    assert reports.geometry.x.max() < 11.25


def test_a_vessel_under_way_keeps_every_report_of_its_track() -> None:
    # 550 m in a minute is 18 knots, which is a ferry rather than a fault. The rule is a ceiling
    # on the physically possible, and a slice that removed the vessels actually moving would be
    # removing the ones the interpolation exists for.
    archive = one_day(
        [
            report(age=timedelta(minutes=-2), at=(11.28, 56.640)),
            report(age=timedelta(minutes=-1), at=(11.28, 56.645)),
            report(age=timedelta(minutes=0), at=(11.28, 56.650)),
        ]
    )

    reports, cleaning = ingest(archive)

    assert len(reports) == 3
    assert cleaning.implausible_jump == 0


def test_a_vessel_that_reported_once_survives_cleaning() -> None:
    # It has no neighbour to be unreachable from, and one report is a usable declaration: it is
    # what `interpolate.py` falls back to and marks `reported`. Dropping it would publish
    # whatever detection it explains as a dark vessel.
    archive = one_day([report()])

    reports, cleaning = ingest(archive)

    assert len(reports) == 1
    assert cleaning.implausible_jump == 0


def test_what_the_cleaning_removed_is_counted_and_adds_up() -> None:
    """Every row read is accounted for, or the report is a claim nobody can check.

    Cleaning raw AIS is part of the work rather than a preliminary to it, and a count that does
    not add up is how a rule that quietly removes half the archive goes unnoticed.
    """
    archive = one_day(
        [
            report(mmsi="219000001", age=timedelta(minutes=-1)),
            report(mmsi="219000001", age=timedelta(minutes=-1)),
            report(mmsi="219000002", at=FAR_AWAY),
            report(mmsi="219000003", at=(181.0, 91.0)),
            report(mmsi="2190064"),
            report(mmsi="219000004", mobile="Base Station"),
            report(mmsi="219000005", age=timedelta(hours=6)),
        ]
    )

    _, cleaning = ingest(archive)

    assert cleaning.read == 7
    assert cleaning.no_position == 1
    # Seven read, less the one with no position, the one in the Øresund and the one six hours
    # away: four reach the cleaning, and one of the four is a vessel that declared itself here.
    assert cleaning.in_area_and_window == 4
    removed = (
        cleaning.not_a_vessel
        + cleaning.missing_identifier
        + cleaning.duplicated
        + cleaning.contradictory
        + cleaning.implausible_jump
    )
    assert cleaning.in_area_and_window - removed == cleaning.kept
    assert cleaning.kept == 1


def test_every_count_the_cleaning_keeps_is_one_the_report_says_out_loud() -> None:
    """A rule whose count is not printed is a rule nobody audits.

    The counts exist to be read, and the one way they quietly stop being read is a field added to
    `Cleaning` later and left out of the report. Checked by field rather than by wording so that
    the sentences can be rewritten without this having an opinion about them.
    """
    counted = Cleaning(
        read=1_000,
        no_position=1,
        no_timestamp=2,
        in_area_and_window=3,
        not_a_vessel=4,
        missing_identifier=5,
        duplicated=6,
        contradictory=7,
        implausible_jump=8,
        kept=9,
    )

    said = " ".join(counted.lines())

    for field_name in counted.__dataclass_fields__:
        assert str(getattr(counted, field_name)) in said, f"{field_name} is never reported"


def test_an_empty_slice_is_an_answer_and_not_an_error() -> None:
    # A search that ran and found nothing is what `classify` calls honestly dark. It must come
    # back as an empty slice of the right shape, never as None and never as an exception.
    archive = one_day([report(at=FAR_AWAY)])

    reports, cleaning = ingest(archive)

    assert reports.empty
    assert cleaning.kept == 0
    assert set(reports.columns) >= {"mmsi", "timestamp", "geometry"}


def test_the_slice_written_is_the_slice_the_chain_reads_back(tmp_path: Path) -> None:
    """The ingestion writes what `load_ais` reads, and nothing in between reinterprets it.

    Two commands and one file: the day is fetched once and the chain runs from the result as
    often as it likes, with no network. The round trip is where an MMSI loses a leading zero or
    a timestamp loses its zone, and both are silent.
    """
    archive = one_day(
        [
            report(mmsi="219000001", age=timedelta(minutes=-1)),
            report(mmsi="019000002", age=timedelta(minutes=1), at=(11.30, 56.66)),
        ]
    )
    path = tmp_path / "anholt-ais.csv"

    reports, _ = ingest(archive)
    write_ais(reports, path)
    read_back = load_ais(path, crs="EPSG:4326")

    # The leading zero is still there: an MMSI is an identifier, and read back as a number this
    # one would come back as 19000002 and match nothing.
    assert list(read_back["mmsi"]) == ["019000002", "219000001"]
    assert list(read_back["timestamp"]) == list(reports["timestamp"])
    assert read_back.geometry.x.tolist() == pytest.approx(reports.geometry.x.tolist())


def test_the_first_member_of_a_zip_is_inflated_without_seeking_back() -> None:
    """3.3 GB of CSV, and neither the disk nor the memory of this machine holds it.

    `zipfile` needs a seekable file, which means the whole 662 MB archive on disk before a row is
    read. The member is inflated straight off the response instead, so what crosses the network
    is a day of Danish AIS and what stays is the few thousand reports the filter kept. The stream
    below refuses to seek, which is the property that matters.
    """
    body = archive_csv([report()])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("aisdk-2026-07-02.csv", body)

    class OneShot(io.RawIOBase):
        """A response: readable once, forwards only."""

        def __init__(self, data: bytes) -> None:
            self._data = io.BytesIO(data)

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def readinto(self, target: memoryview) -> int:  # type: ignore[override]
            return self._data.readinto(target)

    name, member = zip_member(io.BufferedReader(OneShot(buffer.getvalue())))

    assert name == "aisdk-2026-07-02.csv"
    assert member.read().decode() == body


def test_the_archive_url_names_the_day_asked_for() -> None:
    assert archive_url(date(2026, 7, 2)).endswith("aisdk-2026-07-02.zip")


def test_the_shipped_config_describes_an_ais_slice_the_command_can_fetch() -> None:
    """`configs/anholt.yaml` through the command's own parsing, minus the 662 MB.

    The same gap `export_request_from` exists to close: the shipped configs are the ones nothing
    in this suite runs, and this one's faults would otherwise surface to someone who had already
    waited for a day of Danish AIS to come down the wire.
    """
    config = yaml.safe_load(SHIPPED_CONFIG.read_text())

    request = ais_request_from(config, SHIPPED_CONFIG.parent)

    assert request["area"] == ANHOLT
    assert request["window"] > timedelta(0)
    assert request["margin_m"] > 0.0
    assert request["max_speed_kn"] > 0.0
    # The slice the ingestion writes is the slice the run reads. The path is written twice in
    # the config, as the scene's is, and this is what holds the two spellings together.
    assert request["path"] == (SHIPPED_CONFIG.parent / config["run"]["ais"]).resolve()


def test_a_margin_grows_the_area_on_every_side() -> None:
    grown = ANHOLT.grown_by(5_000.0)

    assert grown.west < ANHOLT.west
    assert grown.east > ANHOLT.east
    assert grown.south < ANHOLT.south
    assert grown.north > ANHOLT.north
    # 5 km of northing is about 0.045 degrees of latitude anywhere on the ellipsoid.
    assert grown.north - ANHOLT.north == pytest.approx(0.045, abs=0.002)


def test_a_margin_of_nothing_leaves_the_area_alone() -> None:
    assert ANHOLT.grown_by(0.0) == ANHOLT
