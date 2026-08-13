"""Placing a declared vessel at the moment the radar imaged it.

A vessel moves between its last AIS report and the acquisition, and matching against the report
as it stands manufactures dark vessels that were never there. This file tests the placement on
its own, the way `test_tiling.py` tests the tiling geometry: the chain cannot show whether a
position is right, only whether the classification that followed from it looks plausible.

Every expected coordinate below is worked out by hand from the reports the test writes, so an
error in the interpolation cannot agree with the expectation.
"""

from datetime import UTC, datetime, timedelta

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point

from darkvessel.fusion.interpolate import INTERPOLATED, REPORTED, positions_at

WORKING_CRS = "EPSG:25832"
ACQUIRED_AT = datetime(2026, 3, 14, 5, 30, tzinfo=UTC)

# Wide enough that every bracket below fits inside it, so the tests that are not about the gap
# ceiling see it out of the way. The one that is about it names its own.
MAX_GAP = timedelta(minutes=10)


def reports(track: list[tuple[str, timedelta, float, float]]) -> gpd.GeoDataFrame:
    """Position reports, as (mmsi, offset from the acquisition, x, y) in the working CRS."""
    return gpd.GeoDataFrame(
        {
            "mmsi": [mmsi for mmsi, _, _, _ in track],
            "timestamp": pd.to_datetime([ACQUIRED_AT + age for _, age, _, _ in track], utc=True),
        },
        geometry=[Point(x, y) for _, _, x, y in track],
        crs=WORKING_CRS,
    )


def position_of(placed: gpd.GeoDataFrame, mmsi: str) -> pd.Series:
    """The one row placing the given vessel."""
    here = placed[placed["mmsi"] == mmsi]
    assert len(here) == 1, f"expected exactly one position for {mmsi}, found {len(here)}"
    return here.iloc[0]


def test_a_vessel_is_placed_between_the_reports_that_bracket_the_acquisition() -> None:
    # Reported 3 minutes before at x=639000 and 2 minutes after at x=640500: 1500 m in 5 minutes,
    # of which three fifths are behind the acquisition. 639000 + 0.6 * 1500 = 639900.
    track = reports(
        [
            ("219000001", timedelta(minutes=-3), 639_000.0, 6_281_000.0),
            ("219000001", timedelta(minutes=2), 640_500.0, 6_281_000.0),
        ]
    )

    placed = positions_at(track, ACQUIRED_AT, MAX_GAP)

    vessel = position_of(placed, "219000001")
    assert vessel.geometry.x == pytest.approx(639_900.0)
    assert vessel.geometry.y == pytest.approx(6_281_000.0)


def test_an_interpolated_position_says_so_and_says_how_far_the_nearest_report_sits() -> None:
    # An interpolated position is a construction, not an observation, and how much it is worth
    # depends on how tight the bracket was. Nearest real report here is the one 2 minutes after.
    track = reports(
        [
            ("219000001", timedelta(minutes=-3), 639_000.0, 6_281_000.0),
            ("219000001", timedelta(minutes=2), 640_500.0, 6_281_000.0),
        ]
    )

    vessel = position_of(positions_at(track, ACQUIRED_AT, MAX_GAP), "219000001")

    assert vessel["position_basis"] == INTERPOLATED
    assert vessel["position_age_s"] == pytest.approx(120.0)


def test_a_vessel_the_acquisition_does_not_bracket_keeps_its_report_and_is_flagged() -> None:
    """The track runs out before the radar looks, so there is nothing to interpolate between.

    Dropping the vessel here would be the worse answer: nothing would explain its detection and
    a declared vessel would be published as dark. It keeps its nearest report, and the row says
    that is what it is — the weakness travels with the position rather than being assumed away.
    """
    track = reports(
        [
            ("219000001", timedelta(minutes=-9), 638_000.0, 6_281_000.0),
            ("219000001", timedelta(minutes=-4), 639_000.0, 6_281_000.0),
        ]
    )

    vessel = position_of(positions_at(track, ACQUIRED_AT, MAX_GAP), "219000001")

    assert vessel["position_basis"] == REPORTED
    assert vessel["position_age_s"] == pytest.approx(240.0)
    # The nearest report as it stands, not a track prolonged past its last observation.
    assert vessel.geometry.x == pytest.approx(639_000.0)


def test_a_bracket_wider_than_the_gap_allowed_is_not_interpolated_across() -> None:
    """Two reports 80 minutes apart bracket the acquisition, and say almost nothing about it.

    A straight line between them is a claim that the vessel held one course and one speed for
    over an hour. The midpoint of that line is not a position, it is a guess dressed as one, and
    the further apart the reports the more confident the guess looks.
    """
    track = reports(
        [
            ("219000001", timedelta(minutes=-40), 630_000.0, 6_281_000.0),
            ("219000001", timedelta(minutes=40), 650_000.0, 6_281_000.0),
        ]
    )

    vessel = position_of(
        positions_at(track, ACQUIRED_AT, max_gap=timedelta(minutes=10)), "219000001"
    )

    assert vessel["position_basis"] == REPORTED
    # Not 640000, the midpoint an interpolation would have produced.
    assert vessel.geometry.x == pytest.approx(630_000.0)
    assert vessel["position_age_s"] == pytest.approx(2400.0)


def test_a_report_at_the_acquisition_instant_is_taken_as_it_stands() -> None:
    # An observation at the moment of the acquisition is the strongest evidence available, and
    # brackets itself: interpolating would divide by a span of zero.
    track = reports(
        [
            ("219000001", timedelta(minutes=-1), 639_000.0, 6_281_000.0),
            ("219000001", timedelta(0), 639_400.0, 6_281_000.0),
            ("219000001", timedelta(minutes=1), 639_800.0, 6_281_000.0),
        ]
    )

    vessel = position_of(positions_at(track, ACQUIRED_AT, MAX_GAP), "219000001")

    assert vessel["position_basis"] == REPORTED
    assert vessel["position_age_s"] == pytest.approx(0.0)
    assert vessel.geometry.x == pytest.approx(639_400.0)


def test_two_vessels_are_placed_from_their_own_tracks_and_not_from_each_others() -> None:
    # Their reports interleave in time and their tracks cross in space, so a track assembled
    # without regard to MMSI would place both somewhere between the two — plausibly, and wrongly.
    track = reports(
        [
            ("219000001", timedelta(minutes=-2), 639_000.0, 6_281_000.0),
            ("219000002", timedelta(minutes=-1), 639_000.0, 6_285_000.0),
            ("219000001", timedelta(minutes=2), 639_400.0, 6_281_000.0),
            ("219000002", timedelta(minutes=1), 639_600.0, 6_285_000.0),
        ]
    )

    placed = positions_at(track, ACQUIRED_AT, MAX_GAP)

    assert len(placed) == 2
    # Halfway along its own segment, both of them.
    assert position_of(placed, "219000001").geometry.x == pytest.approx(639_200.0)
    assert position_of(placed, "219000002").geometry.x == pytest.approx(639_300.0)


def test_a_track_whose_index_repeats_is_still_placed_from_one_whole_row() -> None:
    """Reports arrive with whatever index the ingestion left on them.

    A day of Danish AIS is several archives concatenated and then filtered to the study area,
    and neither operation renumbers rows. Addressing a report by index label rather than by
    position then returns two rows where one was meant, and the position taken from them is not
    a position at all.
    """
    track = reports(
        [
            ("219000001", timedelta(minutes=-9), 638_000.0, 6_281_000.0),
            ("219000001", timedelta(minutes=-4), 639_000.0, 6_281_000.0),
        ]
    )
    track.index = pd.Index([7, 7])

    vessel = position_of(positions_at(track, ACQUIRED_AT, MAX_GAP), "219000001")

    assert vessel.geometry.x == pytest.approx(639_000.0)
    assert vessel["position_age_s"] == pytest.approx(240.0)


@pytest.mark.parametrize("missing", ["mmsi", "timestamp"])
def test_a_report_missing_what_places_it_is_refused_rather_than_filtered_away(
    missing: str,
) -> None:
    """Raw archives carry both, and neither can take part in a track.

    A report with no MMSI belongs to no vessel and a grouping drops it without a word; a report
    with no timestamp compares false against the acquisition either way and falls out of both
    sides of the bracket, equally quietly. A declaration that vanishes on its way to the matching
    is a detection published as dark, which is the fault this module exists to remove.

    Refused rather than repaired: pooling unidentified reports would draw a line between two
    different ships, and what to do with an unusable row is a decision about cleaning raw AIS
    that belongs to whoever ingests it.
    """
    track = reports(
        [
            ("219000001", timedelta(minutes=-1), 639_000.0, 6_281_000.0),
            ("219000001", timedelta(minutes=1), 639_400.0, 6_281_000.0),
        ]
    )
    track.loc[1, missing] = pd.NaT if missing == "timestamp" else pd.NA

    with pytest.raises(ValueError, match=missing):
        positions_at(track, ACQUIRED_AT, MAX_GAP)


def test_no_reports_at_all_places_no_vessels() -> None:
    # `classify` hands this an AIS slice that a filter may have emptied, and a search that ran
    # and found nothing must come back as an empty answer rather than as an error.
    placed = positions_at(reports([]), ACQUIRED_AT, MAX_GAP)

    assert placed.empty
    assert placed.crs == WORKING_CRS
    assert set(placed.columns) >= {"mmsi", "position_basis", "position_age_s", "geometry"}
