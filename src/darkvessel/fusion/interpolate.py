"""Interpolation of AIS tracks to the exact acquisition time.

A vessel moves between its last AIS report and the moment the radar imaged it — at 12 knots, some
370 m a minute, which is more than the match tolerance. Comparing a detection to a stale position
manufactures dark vessels that do not exist. Positions are interpolated along the track to the
acquisition timestamp before any matching occurs.

Only between two reports that bracket the acquisition, and only across a gap narrow enough for a
straight line to mean something. Where the track gives nothing to interpolate between, the vessel
keeps its nearest report and the answer says so — it stays in the search, because dropping a
vessel that declared itself is how a declared vessel gets published as dark. Nothing is ever
extrapolated past the end of a track. Geometry-critical: covered by tests.
"""

from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
from shapely import Point

# The position was constructed at the acquisition instant, from the two reports either side of it.
INTERPOLATED = "interpolated"
# No pair of reports close enough together brackets the acquisition, so the vessel's nearest
# report is used as it stands. Kept in the search rather than dropped — a vessel that declared
# itself and cannot be placed is still a vessel that declared itself — but marked, because a
# position observed at another moment is exactly the evidence this module exists to stop being
# used silently. `position_age_s` says how far away that moment was, and is 0 for the one case
# where this basis is the strongest available: a report standing on the acquisition instant,
# which brackets itself and has nothing left to interpolate.
REPORTED = "reported"


def positions_at(
    ais: gpd.GeoDataFrame,
    acquired_at: datetime,
    max_gap: timedelta,
) -> gpd.GeoDataFrame:
    """One position per vessel, placed at the acquisition instant where the track allows.

    Args:
        ais: Position reports, with `mmsi`, `timestamp` and point geometry. Interpolation is
            linear in the CRS the frame is in, so that CRS must be a projected one.
        acquired_at: The moment the radar imaged the scene.
        max_gap: The widest bracket a straight line is allowed to span. Required rather than
            defaulted, for the reason the match tolerance is: it decides what the answer is, the
            number that belongs there is provisional, and a run should state it rather than
            inherit it. `configs/*.yaml` set it; the reasoning is in docs/decisions.md.

    Returns:
        One row per MMSI, in `ais`'s CRS, carrying `position_basis` — `interpolated` or
        `reported` — and `position_age_s`, the seconds between the acquisition and the nearest
        real report behind the position. Vessels are ordered by MMSI, so a run is reproducible.
    """
    _refuse_unplaceable(ais)

    acquisition = pd.Timestamp(acquired_at)
    placed = [
        _place(str(mmsi), track.sort_values("timestamp", kind="stable"), acquisition, max_gap)
        for mmsi, track in ais.groupby("mmsi", sort=True)
    ]
    # Columns named rather than inferred from the rows, so that a slice a filter emptied comes
    # back as an empty answer of the right shape instead of a frame with nothing in it at all.
    frame = pd.DataFrame(placed, columns=["mmsi", "position_basis", "position_age_s", "geometry"])
    return gpd.GeoDataFrame(
        frame.astype({"mmsi": "string", "position_age_s": "float64"}),
        geometry=gpd.GeoSeries(frame["geometry"].to_list(), index=frame.index, crs=ais.crs),
        crs=ais.crs,
    )


def _refuse_unplaceable(ais: gpd.GeoDataFrame) -> None:
    """Refuse reports missing what puts them on a track, rather than let them fall out quietly.

    Raw archives carry both kinds. A report without an MMSI belongs to no vessel, and grouping
    by MMSI drops it without a word; a report without a timestamp compares false against the
    acquisition in either direction, so it falls out of both sides of the bracket just as
    quietly. Either way a declaration vanishes on its way to the matching, and a declaration
    that vanishes is a detection published as dark — the fault this module exists to remove.

    Refusing here rather than repairing. Pooling unidentified reports under one missing key
    would be worse than dropping them: two of them are two different ships, and a line drawn
    between them is a track that never existed. What to do with an unusable row is a decision
    about cleaning raw AIS, and it belongs to whoever ingests it.
    """
    for column in ("mmsi", "timestamp"):
        unplaceable = int(ais[column].isna().sum())
        if unplaceable:
            raise ValueError(
                f"{unplaceable} AIS report(s) carry no {column} and cannot be placed on any "
                "vessel's track; drop or repair them when the archive is ingested rather than here"
            )


def _place(
    mmsi: str,
    track: gpd.GeoDataFrame,
    acquisition: pd.Timestamp,
    max_gap: timedelta,
) -> dict:
    """Where this vessel was when the radar looked, and on what evidence."""
    before = track[track["timestamp"] <= acquisition]
    after = track[track["timestamp"] >= acquisition]
    if before.empty or after.empty:
        return _from_report(mmsi, _nearest(track, acquisition), acquisition)

    opening, closing = before.iloc[-1], after.iloc[0]
    # A report standing exactly on the acquisition brackets itself, across a span of zero. It is
    # also the best evidence a vessel can offer, and nothing is left to interpolate.
    gap = closing["timestamp"] - opening["timestamp"]
    if gap == pd.Timedelta(0) or gap > pd.Timedelta(max_gap):
        return _from_report(mmsi, _nearest(track, acquisition), acquisition)
    return _between(mmsi, opening, closing, acquisition)


def _between(
    mmsi: str,
    before: pd.Series,
    after: pd.Series,
    acquisition: pd.Timestamp,
) -> dict:
    """A position on the straight line between the two reports either side of the acquisition."""
    span = after["timestamp"] - before["timestamp"]
    travelled = (acquisition - before["timestamp"]) / span
    return {
        "mmsi": mmsi,
        "position_basis": INTERPOLATED,
        # How tight the bracket was. An interpolated position is a construction, and the nearest
        # real observation behind it is what says how much the construction is worth.
        "position_age_s": min(
            (acquisition - before["timestamp"]).total_seconds(),
            (after["timestamp"] - acquisition).total_seconds(),
        ),
        "geometry": Point(
            before.geometry.x + travelled * (after.geometry.x - before.geometry.x),
            before.geometry.y + travelled * (after.geometry.y - before.geometry.y),
        ),
    }


def _from_report(mmsi: str, report: pd.Series, acquisition: pd.Timestamp) -> dict:
    """A report taken as it stands, because the track gives nothing to interpolate between.

    Deliberately not extrapolated. Prolonging a track past its last observation, from a course
    and speed derived from two earlier points, manufactures a position where no measurement
    exists — the same class of confidently wrong answer as matching against a stale one, and
    harder to see because the result looks like a placement rather than a fallback.
    """
    return {
        "mmsi": mmsi,
        "position_basis": REPORTED,
        "position_age_s": abs((report["timestamp"] - acquisition).total_seconds()),
        "geometry": report.geometry,
    }


def _nearest(track: gpd.GeoDataFrame, acquisition: pd.Timestamp) -> pd.Series:
    """The report closest in time to the acquisition.

    The whole row, never a per-column reduction: taking the minimum of each column separately
    would happily pair one report's timestamp with another report's position.

    Addressed by position rather than by index label. A day of Danish AIS is several archives
    concatenated and then filtered, and neither renumbers rows, so labels repeat — and `.loc` on
    a repeated label hands back two rows where one was asked for. `argmin` takes the first of any
    tie and the track is sorted, so a vessel reporting equally far either side of the acquisition
    resolves the same way on every run.
    """
    staleness = (track["timestamp"] - acquisition).abs().to_numpy()
    return track.iloc[int(staleness.argmin())]
