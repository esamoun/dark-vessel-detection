"""The two failure modes `docs/evaluation.md` stopped evidencing on one frame.

Two jobs, and they are different in kind.

The first is the measurement itself, on frames written here rather than on the archive: a
counterfactual whose recovered count is wrong is worse than no counterfactual, because it is a
number a reader has no way to check. So the cases below build acquisitions where the answer is
known by construction — a vessel steaming across the track, a vessel running along it, a
detection with nothing declared near it — and hold what the module reports about them.

The second is the prose. `docs/evaluation.md` now quotes eighteen numbers out of
`docs/runs/failures-archive.json`, and prose drifts: a journal is re-run, a document goes on saying
what used to be true, and no test goes red. The same argument this repository already makes about
the report's table applies to a paragraph typed after reading a JSON, so every one of those numbers
is held against the committed journal. What is *not* pinned is anything the report argues in
words — that is the author's, and a test that pinned it would pin an opinion.

The one number in the journal that is not a measurement of the water is `reproduces_published`.
It says the corrected pass returned the layer that is on disk, and it is asserted here rather than
read for a reason: false, and every other figure in the entry is a figure about some other run.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from darkvessel.analysis.failures import (
    Acquisition,
    counterfactual,
    duplication,
    failures_request_from,
)
from darkvessel.fusion.azimuth import DESCENDING, Geometry
from darkvessel.fusion.match import DARK, MATCHED

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "evaluation.md"
JOURNAL = ROOT / "docs" / "runs" / "failures-archive.json"

WORKING_CRS = "EPSG:25832"
ACQUIRED_AT = datetime(2026, 8, 9, 5, 31, 24, tzinfo=UTC)
TOLERANCE_M = 200.0
MAX_GAP = timedelta(minutes=10)

# The geometry the archive was corrected with: a descending pass at the middle of an IW swath.
DESCENDING_PASS = Geometry(orbit_pass=DESCENDING, incidence_deg=38.5)

# Well inside the Kattegat box, in the working CRS, so the bearing of the track is the one the
# archive's own scenes see rather than one from some other latitude.
EAST = 639_000.0
NORTH = 6_386_000.0


def declaration(
    mmsi: str,
    *,
    at: tuple[float, float],
    course: tuple[float, float],
    length_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """Two reports bracketing the acquisition, so a position can be interpolated to it.

    `course` is metres per minute east and north; the vessel is placed at `at` at the moment of
    acquisition and its two reports are a minute either side of it. Writing the course rather than
    two positions is what makes the cases below readable: the displacement the correction undoes is
    a function of the across-track component of exactly this vector.
    """
    east, north = at
    per_minute_e, per_minute_n = course
    return gpd.GeoDataFrame(
        {
            "mmsi": [mmsi, mmsi],
            "timestamp": pd.to_datetime(
                [ACQUIRED_AT - timedelta(minutes=1), ACQUIRED_AT + timedelta(minutes=1)], utc=True
            ),
            "length_m": [length_m, length_m],
        },
        geometry=[
            Point(east - per_minute_e, north - per_minute_n),
            Point(east + per_minute_e, north + per_minute_n),
        ],
        crs=WORKING_CRS,
    )


def detections(*places: tuple[float, float], scene: str = "s1") -> gpd.GeoDataFrame:
    """Detections as the archive layer carries them, published verdicts included.

    `status` and `mmsi` are what the corrected pass is checked against, so they are filled in by
    the tests that care and left as the corrected pass will find them by the ones that do not.
    """
    return gpd.GeoDataFrame(
        {
            "score": [0.95] * len(places),
            "scene": [scene] * len(places),
            "status": [MATCHED] * len(places),
            "mmsi": [None] * len(places),
            "length_m": [200.0] * len(places),
        },
        geometry=[Point(east, north) for east, north in places],
        crs=WORKING_CRS,
    )


def measure(*acquisitions: Acquisition):
    return counterfactual(list(acquisitions), tolerance_m=TOLERANCE_M, max_gap=MAX_GAP)


def acquisition(
    rows: gpd.GeoDataFrame,
    ais: gpd.GeoDataFrame | None,
    *,
    scene: str = "s1",
    geometry: Geometry | None = DESCENDING_PASS,
) -> Acquisition:
    return Acquisition(
        scene=scene,
        acquired_at=ACQUIRED_AT,
        detections=rows.reset_index(drop=True),
        ais=ais,
        geometry=geometry,
    )


# --- the counterfactual, on frames whose answer is known by construction -----------------------


def test_a_vessel_crossing_the_track_is_recovered_by_the_correction_and_not_without_it() -> None:
    """The whole failure mode, in one frame.

    A vessel steaming due east at 12 knots is displaced hundreds of metres along the satellite's
    ground track, so the radar draws it well outside the 200 m tolerance of where it says it is.
    Corrected, the declaration moves to where the radar drew it and the pair matches; uncorrected,
    the same detection is published as a dark vessel.
    """
    ais = declaration("219000001", at=(EAST, NORTH), course=(370.0, 0.0))
    # South of the declaration, because the pass is descending: the radar draws a vessel closing on
    # it back along its own ground track. 425 m is beyond the tolerance and inside the correction.
    result = measure(acquisition(detections((EAST, NORTH - 425.0)), ais))

    assert result.matched == 1
    assert result.matched_uncorrected == 0
    assert result.recovered == 1
    assert result.lost == 0
    assert result.dark_uncorrected == 1
    assert result.vessels_recovered == 1
    assert result.scenes_with_recovery == 1
    assert result.shift_max_m > TOLERANCE_M


def test_a_vessel_running_along_the_track_is_matched_either_way() -> None:
    """No across-track velocity, no displacement, nothing for the correction to do.

    The counterpart of the case above and the reason the archive still shows 60 matches without any
    correction at all: most of what a lane carries is not crossing the radar's line of sight.
    """
    ais = declaration("219000002", at=(EAST, NORTH), course=(0.0, -370.0))
    result = measure(acquisition(detections((EAST, NORTH)), ais))

    assert result.matched == 1
    assert result.matched_uncorrected == 1
    assert result.matched_either_way == 1
    assert result.recovered == 0


def test_a_detection_with_nothing_declared_near_it_is_dark_either_way() -> None:
    ais = declaration("219000003", at=(EAST + 9_000.0, NORTH), course=(0.0, 0.0))
    rows = detections((EAST, NORTH))
    rows["status"] = [DARK]
    result = measure(acquisition(rows, ais))

    assert result.matched == 0
    assert result.matched_uncorrected == 0
    assert result.dark_either_way == 1
    assert result.recovered == 0


def test_the_recovered_are_counted_as_vessels_and_as_acquisitions_not_as_rows() -> None:
    """One ship seen on two passes is one vessel and two acquisitions.

    The archive is ten weeks of one lane, so the same hull recurs, and a count of recovered rows
    would report the traffic's regularity as if it were the correction's reach.
    """
    ais = declaration("219000004", at=(EAST, NORTH), course=(370.0, 0.0))
    first = acquisition(detections((EAST, NORTH - 425.0), scene="s1"), ais, scene="s1")
    second = acquisition(detections((EAST, NORTH - 425.0), scene="s2"), ais, scene="s2")
    result = measure(first, second)

    assert result.recovered == 2
    assert result.vessels_recovered == 1
    assert result.scenes_with_recovery == 2
    assert result.scenes == 2


def test_an_acquisition_the_layer_holds_no_detection_for_counts_for_nothing() -> None:
    empty = acquisition(detections(scene="s2"), None, scene="s2")
    result = measure(acquisition(detections((EAST, NORTH)), None), empty)

    assert result.scenes == 1
    assert result.detections == 1


def test_a_published_verdict_the_corrected_pass_does_not_return_is_reported() -> None:
    """The check that makes the other column worth reading.

    Here the layer claims a match the corrected pass cannot find, which is exactly the shape of a
    recomputation that has drifted from the run it claims to describe. The measurement does not
    raise — the counts are still true of the pass it just ran — it says so.
    """
    ais = declaration("219000007", at=(EAST + 9_000.0, NORTH), course=(0.0, 0.0))
    rows = detections((EAST, NORTH))
    rows["status"] = [MATCHED]
    rows["mmsi"] = ["219000007"]
    result = measure(acquisition(rows, ais))

    assert result.reproduces_published is False
    assert result.matched == 0


def test_a_run_that_reproduces_the_layer_says_so() -> None:
    ais = declaration("219000005", at=(EAST, NORTH), course=(0.0, 0.0))
    rows = detections((EAST, NORTH))
    rows["status"] = [MATCHED]
    rows["mmsi"] = ["219000005"]
    result = measure(acquisition(rows, ais))

    assert result.reproduces_published is True
    assert result.matched == 1


# --- duplication -------------------------------------------------------------------------------


def test_two_detections_closer_together_than_the_longer_hull_are_a_candidate_duplicate() -> None:
    rows = detections((EAST, NORTH), (EAST + 150.0, NORTH))
    result = duplication(rows)

    assert result.candidate_duplicates == 1
    assert result.closest_pair_m == pytest.approx(150.0)
    assert result.longest_hull_m == pytest.approx(200.0)


def test_two_detections_further_apart_than_any_hull_are_not() -> None:
    rows = detections((EAST, NORTH), (EAST + 800.0, NORTH))
    result = duplication(rows)

    assert result.candidate_duplicates == 0
    assert result.scenes_with_pairs == 1


def test_pairs_are_never_formed_across_two_acquisitions() -> None:
    """The same berth photographed a week apart is the traffic, not a duplicate."""
    first = detections((EAST, NORTH), scene="s1")
    second = detections((EAST + 10.0, NORTH), scene="s2")
    result = duplication(pd.concat([first, second], ignore_index=True))

    assert result.scenes == 2
    assert result.scenes_with_pairs == 0
    assert result.candidate_duplicates == 0


def test_one_declaration_claimed_by_two_detections_is_counted() -> None:
    rows = detections((EAST, NORTH), (EAST + 9_000.0, NORTH))
    rows["mmsi"] = ["219000006", "219000006"]
    assert duplication(rows).declarations_matched_twice == 1


# --- what the command is asked for -------------------------------------------------------------


def test_the_request_resolves_every_path_against_the_config_file() -> None:
    config = {
        "archive": {
            "scenes": "../data/archive/lane",
            "ais": "../data/real/lane-ais",
            "detections": "../outputs/lane.gpkg",
        },
        "failures": {"report": "../docs/runs/failures-lane.json"},
    }
    request = failures_request_from(config, Path("/repo/configs"))

    assert request["scenes"] == Path("/repo/data/archive/lane")
    assert request["ais"] == Path("/repo/data/real/lane-ais")
    assert request["detections"] == Path("/repo/outputs/lane.gpkg")
    assert request["report"] == Path("/repo/docs/runs/failures-lane.json")


def test_the_subcommand_is_registered_and_dispatched(tmp_path: Path) -> None:
    """A parser entry with no branch behind it, or a branch no parser reaches, is a dead command.

    Reaching `load_config` is the whole assertion: argparse would exit 2 on a subcommand it does
    not know, and the dispatch would fall through to `run` on one it does not handle.
    """
    from darkvessel.cli import main

    with pytest.raises(FileNotFoundError):
        main(["failures", "--config", str(tmp_path / "nothing.yaml")])


def test_the_shipped_config_names_the_journal_the_report_quotes() -> None:
    """The one config in the repository that this command is run against, held to the file it wrote.

    `configs/kattegat-lane.yaml` needs Earth Engine credentials before most of the chain can fail,
    so it goes through the request function here rather than through a run.
    """
    import yaml

    config = yaml.safe_load((ROOT / "configs" / "kattegat-lane.yaml").read_text())
    request = failures_request_from(config, ROOT / "configs")

    assert request["report"] == JOURNAL
    assert request["detections"].name == "kattegat-lane-archive.gpkg"


# --- the prose, held against the journal -------------------------------------------------------


@pytest.fixture(scope="module")
def journal() -> dict:
    return json.loads(JOURNAL.read_text())


@pytest.fixture(scope="module")
def prose() -> str:
    """The report with its line wrapping collapsed.

    A sentence is pinned to a journal, not to the column it happens to be wrapped at, and a test
    that broke when a paragraph was rewrapped would be a test nobody keeps.
    """
    return " ".join(REPORT.read_text().split())


def test_the_corrected_pass_reproduced_the_published_layer(journal: dict) -> None:
    """The claim the whole counterfactual rests on, asserted rather than quoted.

    Every other number in the entry describes the difference between two passes of the matching
    stage. If the first of those passes did not return the layer this project published, the
    difference is between two runs nobody has seen.
    """
    assert journal["correction"]["reproduces_published"] is True


def test_the_counterfactual_is_a_strict_superset(journal: dict) -> None:
    """`lost` is zero, so the report may say the corrected verdict adds and never takes away.

    Not a law: the assignment is confidence-ordered and one-to-one, so moving the declarations
    could hand one to a different detection. It is a measurement, which is why it is held here — a
    run that lost a match would make the sentence in the report false.
    """
    correction = journal["correction"]
    assert correction["lost"] == 0
    assert correction["matched"] == correction["matched_either_way"] + correction["recovered"]
    assert (
        correction["detections"]
        == correction["matched_either_way"]
        + correction["recovered"]
        + correction["dark_either_way"]
    )


def test_the_archive_holds_no_duplicate_of_any_hull(journal: dict) -> None:
    """The negative the report states in words, held as the two numbers it is made of."""
    dup = journal["duplication"]
    assert dup["candidate_duplicates"] == 0
    assert dup["declarations_matched_twice"] == 0
    assert dup["closest_pair_m"] > dup["longest_hull_m"]


def test_the_reports_correction_numbers_are_the_journals(journal: dict, prose: str) -> None:
    c = journal["correction"]
    assert (
        f"**{c['matched']} of the archive's {c['detections']} detections match a declaration. "
        f"Without the correction, {c['matched_uncorrected']}.**" in prose
    )
    assert (
        f"**{c['recovered']} detections are recovered, none is lost, "
        f"{c['matched_either_way']} match either way and {c['dark_either_way']} "
        f"are dark either way**" in prose
    )
    assert (
        f"**{c['vessels_recovered']} distinct vessels over {c['scenes_with_recovery']} of the "
        f"{c['scenes']} acquisitions**" in prose
    )
    assert (
        f"**{c['shift_median_m']:.0f} m at the median and {c['shift_max_m']:.0f} m at most**"
        in prose
    )
    assert f"**{c['rate_uncorrected']:.1%} instead of {c['rate']:.1%}**" in prose
    assert f"those {c['dark_uncorrected']} candidates" in prose
    assert (
        f"**{c['recovered']} of {c['matched']} matches exist only because the declaration was "
        f"moved into the radar's frame**" in prose
    )


def test_the_reports_duplication_numbers_are_the_journals(journal: dict, prose: str) -> None:
    dup = journal["duplication"]
    assert (
        f"**{dup['scenes_with_pairs']} acquisitions carry two detections or more; the closest "
        f"pair anywhere in them is {dup['closest_pair_m']:.0f} m apart, "
        f"{dup['median_closest_pair_m']:.0f} m at the median; and the longest hull estimated "
        f"anywhere in the archive is {dup['longest_hull_m']:.0f} m.**" in prose
    )
    assert (
        f"The gap between {dup['closest_pair_m']:.0f} m and {dup['longest_hull_m']:.0f} m is not "
        f"a near miss" in prose
    )


def test_the_report_names_the_journal_and_the_command_that_writes_it(prose: str) -> None:
    assert "runs/failures-archive.json" in prose
    assert "darkvessel failures --config configs/kattegat-lane.yaml" in prose


def test_the_readme_quotes_the_journal_too(journal: dict) -> None:
    """The one number of this measurement that reaches the front page, held to the same file.

    Level 3's row says the correction is worth something, and a row that said so in words would be
    the kind of claim this project does not make. Pinned here rather than in a README test of its
    own, because what it must not drift from is this journal.
    """
    c = journal["correction"]
    readme = " ".join((ROOT / "README.md").read_text().split())

    assert f"**{c['recovered']} of the archive's {c['matched']} matches**" in readme
    assert f"re-running all {c['scenes']} acquisitions with the correction switched off" in readme
