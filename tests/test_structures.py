"""Finding fixed structures in the archive, and keeping them out of the dark count.

Three seams, and they fail differently. Grouping detections into standing positions is arithmetic
over coordinates, so it is tested against a lattice built by hand where the right answer is known
before the code runs. The clustering is tested for the properties a partition has to have —
determinism, the geometry it ranks in, refusal where it cannot answer — and not for cluster
quality, which is measured on real data and reported rather than asserted. The register is tested
for the one thing that would be silent and catastrophic: a detection excluded from the dark count
must leave a trace in the layer, and a vessel that declared itself must never be swallowed by a
structure it moored beside.

The verification against published coordinates is arithmetic too, and it is tested in both
directions — a register that found every published position while inventing fifty of its own
scores perfectly on one of them.
"""

from datetime import UTC, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Point

from darkvessel.embed.structures import (
    SAME_POSITION_M,
    Clustering,
    cluster,
    describe,
    separation,
    standing,
    verify,
)
from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED
from darkvessel.fusion.register import (
    STRUCTURE,
    Register,
    reduction,
    without_a_register,
)

WORKING_CRS = "EPSG:25832"


def provenance_of(places: list[tuple[str, float, float]]) -> pd.DataFrame:
    """The columns `standing` reads, from a list of (scene, x, y)."""
    return pd.DataFrame(
        {
            "scene": [scene for scene, _, _ in places],
            "acquired_at": [datetime(2026, 6, 1, tzinfo=UTC).isoformat()] * len(places),
            "row": [0] * len(places),
            "col": [0] * len(places),
            "x": [x for _, x, _ in places],
            "y": [y for _, _, y in places],
            "score": [0.5] * len(places),
        }
    )


def a_lattice(masts: int, acquisitions: int, spacing_m: float = 600.0) -> pd.DataFrame:
    """`masts` fixed positions, each detected once in each of `acquisitions` scenes.

    With a wobble of up to 20 m on every sighting, which is what a detection of a 10 m-pixel
    point scatterer does, and comfortably inside the tolerance while the spacing is far outside
    it. The right answer is therefore `masts` positions of `acquisitions` acquisitions each.
    """
    wobble = np.random.default_rng(0).uniform(-20.0, 20.0, size=(acquisitions, masts, 2))
    return provenance_of(
        [
            (
                f"anholt/scene-{acquisition:02d}",
                600_000.0 + mast * spacing_m + wobble[acquisition, mast, 0],
                6_270_000.0 + wobble[acquisition, mast, 1],
            )
            for acquisition in range(acquisitions)
            for mast in range(masts)
        ]
    )


def test_a_lattice_seen_every_week_is_one_standing_position_per_mast():
    found = standing(a_lattice(masts=8, acquisitions=30))

    assert len(found.positions) == 8
    assert set(found.positions["acquisitions"]) == {30}
    assert set(found.positions["crops"]) == {30}


def test_a_position_is_placed_at_the_centre_of_its_sightings_not_the_first_one():
    """Forty detections locate a mast better than whichever one the loop reached first."""
    scattered = provenance_of(
        [
            (f"anholt/scene-{index:02d}", 600_000.0 + offset, 6_270_000.0)
            for index, offset in enumerate([-40.0, 0.0, 40.0])
        ]
    )

    found = standing(scattered)

    assert len(found.positions) == 1
    assert found.positions["x"].iloc[0] == pytest.approx(600_000.0)


def test_a_vessel_under_way_leaves_one_position_per_acquisition():
    """The control the whole method rests on: a ship does not come back to the same water."""
    steaming = provenance_of(
        [
            (f"lane/scene-{index:02d}", 600_000.0 + index * 5_000.0, 6_380_000.0)
            for index in range(12)
        ]
    )

    found = standing(steaming)

    assert len(found.positions) == 12
    assert set(found.positions["acquisitions"]) == {1}


def test_two_masts_closer_together_than_the_tolerance_are_one_position():
    """Stated rather than hidden: the tolerance is what "the same standing object" means here."""
    pair = provenance_of(
        [
            ("anholt/scene-00", 600_000.0, 6_270_000.0),
            ("anholt/scene-00", 600_000.0 + SAME_POSITION_M / 2, 6_270_000.0),
        ]
    )

    assert len(standing(pair).positions) == 1
    assert len(standing(pair, tolerance_m=SAME_POSITION_M / 4).positions) == 2


def test_one_acquisition_seen_twice_at_a_position_counts_once():
    """A hull cut twice in one scene is one acquisition, not two — the evidence is recurrence."""
    twice = provenance_of(
        [
            ("anholt/scene-00", 600_000.0, 6_270_000.0),
            ("anholt/scene-00", 600_000.0 + 30.0, 6_270_000.0),
            ("anholt/scene-01", 600_000.0 + 10.0, 6_270_000.0),
        ]
    )

    found = standing(twice)

    assert len(found.positions) == 1
    assert found.positions["acquisitions"].iloc[0] == 2
    assert found.positions["crops"].iloc[0] == 3


def test_every_crop_is_assigned_to_exactly_one_position():
    found = standing(a_lattice(masts=5, acquisitions=6))

    assert found.of_crop.min() >= 0
    assert len(found.of_crop) == 30
    assert np.bincount(found.of_crop).tolist() == [6] * 5


def test_the_acquisition_count_of_a_crop_is_the_count_of_its_own_position():
    found = standing(a_lattice(masts=4, acquisitions=9))

    assert found.acquisitions_of_crop().tolist() == [9] * 36


def test_the_persistent_positions_come_back_most_persistent_first():
    mixed = pd.concat(
        [a_lattice(masts=2, acquisitions=20), provenance_of([("lane/scene-99", 1.0, 2.0)])],
        ignore_index=True,
    )

    found = standing(mixed)

    assert found.seen_in(10)["acquisitions"].tolist() == [20, 20]
    assert len(found.seen_in(1)) == 3


# --- the clustering -------------------------------------------------------------------------


def two_bundles(count: int = 60) -> np.ndarray:
    """Two tight bundles of unit vectors pointing in directions far apart."""
    noise = np.random.default_rng(7).normal(scale=0.02, size=(2 * count, 4))
    here = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
    there = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (count, 1))
    return np.concatenate([here, there]) + noise


def test_two_bundles_are_found_as_two_clusters():
    found = cluster(two_bundles(), count=2, seed=3)

    first, second = found.labels[:60], found.labels[60:]
    assert len(set(first)) == 1
    assert len(set(second)) == 1
    assert first[0] != second[0]


def test_the_same_seed_returns_the_same_partition():
    """A figure that changes between two runs of one command argues for nothing."""
    vectors = two_bundles()

    assert np.array_equal(
        cluster(vectors, count=4, seed=11).labels, cluster(vectors, count=4, seed=11).labels
    )


def test_clustering_ranks_by_direction_and_not_by_length():
    """Cosine, which is the geometry the contrastive loss was written in."""
    directions = two_bundles()
    stretched = directions * np.linspace(0.1, 10.0, len(directions))[:, None]

    assert np.array_equal(
        cluster(directions, count=2, seed=5).labels, cluster(stretched, count=2, seed=5).labels
    )


def test_more_clusters_than_crops_is_refused():
    with pytest.raises(ValueError, match="cannot be divided"):
        cluster(two_bundles(count=2), count=99, seed=0)


def masts_and_traffic() -> "object":
    """Ninety crops of three masts and ninety crops of ninety passing ships, in that order."""
    return standing(
        pd.concat(
            [
                a_lattice(masts=3, acquisitions=30),
                provenance_of(
                    [
                        (f"lane/scene-{index:02d}", 700_000.0 + index * 9_000.0, 6_380_000.0)
                        for index in range(90)
                    ]
                ),
            ],
            ignore_index=True,
        )
    )


def test_a_cluster_of_standing_crops_is_described_as_persistent():
    """The label-free description: what share of a cluster stands still across acquisitions."""
    found = masts_and_traffic()
    # One cluster per group, by construction: the first 90 crops are the lattice.
    labels = np.array([0] * 90 + [1] * 90)
    described = describe(Clustering(labels=labels, centres=np.eye(2)), found, floor=20)

    assert described.loc[described["cluster"] == 0, "persistent"].iloc[0] == 1.0
    assert described.loc[described["cluster"] == 1, "persistent"].iloc[0] == 0.0


def test_separation_is_one_when_the_two_kinds_point_in_different_directions():
    found = masts_and_traffic()
    apart = np.concatenate([np.tile([1.0, 0.0], (90, 1)), np.tile([0.0, 1.0], (90, 1))])

    assert separation(apart, found, floor=20) == pytest.approx(1.0)


def test_separation_is_a_half_when_the_representation_has_collapsed():
    """The failure this number exists to catch: every crop at one point, ranked at chance."""
    found = masts_and_traffic()

    assert separation(np.ones((180, 2)), found, floor=20) == pytest.approx(0.5)


# --- verification against published coordinates ---------------------------------------------


def frame_of(places: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame({"x": [x for x, _ in places], "y": [y for _, y in places]})


def test_a_register_that_stands_on_every_published_position_verifies_in_both_directions():
    published = frame_of([(0.0, 0.0), (600.0, 0.0), (1200.0, 0.0)])
    registered = frame_of([(5.0, 0.0), (604.0, 0.0), (1197.0, 0.0)])

    checked = verify(registered, published, tolerance_m=200.0)

    assert (checked.found, checked.known) == (3, 3)
    assert checked.unpublished == 0
    assert checked.median_m == pytest.approx(4.0)


def test_a_register_full_of_positions_nobody_published_is_not_hidden_by_its_recall():
    """Finding every turbine while inventing fifty structures is not a verified register."""
    published = frame_of([(0.0, 0.0)])
    registered = frame_of([(0.0, 0.0)] + [(50_000.0 + index * 600.0, 0.0) for index in range(50)])

    checked = verify(registered, published, tolerance_m=200.0)

    assert checked.found == 1
    assert checked.unpublished == 50


def test_verifying_against_no_published_positions_is_refused_rather_than_scored():
    """Nothing to check against is a statement about the reference, not a score for the register."""
    with pytest.raises(ValueError, match="no published positions"):
        verify(frame_of([(0.0, 0.0)]), frame_of([]), tolerance_m=200.0)


def test_a_register_that_found_none_of_the_farm_scores_zero_rather_than_raising():
    """The failure this check exists to catch has to come back as a number.

    A box with published turbines and nothing registered in it is a method that found nothing,
    which is exactly the result the verification is for. Raising there would turn the loudest
    negative result in the level into a crashed command.
    """
    checked = verify(frame_of([]), frame_of([(0.0, 0.0), (600.0, 0.0)]), tolerance_m=200.0)

    assert (checked.known, checked.registered, checked.found) == (2, 0, 0)
    # Infinite and not zero: there is no pair to measure between, and zero would read as perfect.
    assert checked.median_m == float("inf")


# --- the exclusion --------------------------------------------------------------------------


def detections_at(places: list[tuple[float, float]], status: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"status": status},
        geometry=[Point(x, y) for x, y in places],
        crs=WORKING_CRS,
    )


def a_register(places: list[tuple[float, float]], tolerance_m: float = 200.0) -> Register:
    return Register(
        positions=pd.DataFrame(
            {
                "x": [x for x, _ in places],
                "y": [y for _, y in places],
                "acquisitions": [40] * len(places),
                "source": ["archive"] * len(places),
            }
        ),
        crs=WORKING_CRS,
        tolerance_m=tolerance_m,
    )


def test_a_detection_on_a_registered_structure_stops_being_dark():
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at([(600_000.0 + 30.0, 6_270_000.0)], [DARK])
    )

    assert marked["status"].tolist() == [STRUCTURE]
    assert marked["structure_distance_m"].iloc[0] == pytest.approx(30.0)


def test_a_detection_away_from_every_registered_structure_stays_dark_and_says_so():
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at([(650_000.0, 6_270_000.0)], [DARK])
    )

    assert marked["status"].tolist() == [DARK]
    assert np.isnan(marked["structure_distance_m"].iloc[0])


def test_a_vessel_that_declared_itself_is_never_swallowed_by_a_structure_it_moored_beside():
    """An MMSI is stronger evidence than a coordinate on a list, and the distance is kept anyway."""
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at([(600_000.0 + 10.0, 6_270_000.0)], [MATCHED])
    )

    assert marked["status"].tolist() == [MATCHED]
    assert marked["structure_distance_m"].iloc[0] == pytest.approx(10.0)


def test_an_unsearched_detection_on_a_structure_is_still_a_structure():
    """A run with no AIS knows nothing about vessels and everything about where the masts are."""
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at([(600_000.0, 6_270_000.0)], [UNSEARCHED])
    )

    assert marked["status"].tolist() == [STRUCTURE]


def test_the_excluded_rows_stay_in_the_layer_so_the_exclusion_can_be_audited():
    """The design constraint of the whole module: nothing is dropped, only relabelled."""
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at([(600_000.0, 6_270_000.0), (650_000.0, 6_270_000.0)], [DARK, DARK])
    )

    assert len(marked) == 2
    assert marked.geometry.iloc[0].x == pytest.approx(600_000.0)


def test_the_reduction_says_what_the_run_would_have_reported_without_the_register():
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at(
            [(600_000.0, 6_270_000.0), (600_000.0 + 50.0, 6_270_000.0), (650_000.0, 6_270_000.0)],
            [DARK, DARK, DARK],
        )
    )

    assert reduction(marked, searched=True) == (
        "2 detection(s) excluded as fixed structures, leaving 1 dark: without the register this "
        "run would have reported 3"
    )


def test_a_run_with_no_ais_does_not_report_the_structures_it_excluded_as_dark():
    """The layer cannot say it: a fully excluded scene has no unsearched row left to read."""
    marked = a_register([(600_000.0, 6_270_000.0)]).mark(
        detections_at([(600_000.0, 6_270_000.0)], [UNSEARCHED])
    )

    assert reduction(marked, searched=False) == (
        "1 detection(s) excluded as fixed structures, leaving 0 unsearched: without the register "
        "this run would have reported 1"
    )


def test_a_run_with_no_register_carries_the_column_empty_rather_than_not_at_all():
    """A layer whose schema depends on which stages were switched on cannot be stacked."""
    plain = without_a_register(detections_at([(600_000.0, 6_270_000.0)], [DARK]))

    assert "structure_distance_m" in plain.columns
    assert np.isnan(plain["structure_distance_m"].iloc[0])
    assert plain["status"].tolist() == [DARK]


def test_a_register_survives_a_round_trip_through_a_file(tmp_path):
    written = a_register([(600_000.0, 6_270_000.0), (600_600.0, 6_270_000.0)], tolerance_m=150.0)
    written.write(tmp_path / "structures.csv")

    read = Register.read(tmp_path / "structures.csv")

    assert len(read) == 2
    assert read.crs == WORKING_CRS
    assert read.tolerance_m == pytest.approx(150.0)
    assert read.positions["acquisitions"].tolist() == [40, 40]


def test_a_register_file_stating_two_different_frames_is_refused(tmp_path):
    """Two sets of metres in one file is the silent fault this check exists for."""
    written = a_register([(600_000.0, 6_270_000.0), (600_600.0, 6_270_000.0)])
    written.write(tmp_path / "structures.csv")
    confused = pd.read_csv(tmp_path / "structures.csv")
    confused.loc[1, "crs"] = "EPSG:32632"
    confused.to_csv(tmp_path / "structures.csv", index=False)

    with pytest.raises(ValueError, match="different values of crs"):
        Register.read(tmp_path / "structures.csv")


def test_a_register_with_no_radius_is_refused():
    with pytest.raises(ValueError, match="explains nothing"):
        a_register([(600_000.0, 6_270_000.0)], tolerance_m=0.0)


def test_a_register_marks_detections_given_in_another_frame():
    """The register is the fixed thing; the detections are reprojected into it, not the reverse."""
    elsewhere = detections_at([(600_000.0, 6_270_000.0)], [DARK]).to_crs("EPSG:4326")

    marked = a_register([(600_000.0, 6_270_000.0)]).mark(elsewhere)

    assert marked["status"].tolist() == [STRUCTURE]
    assert marked["structure_distance_m"].iloc[0] == pytest.approx(0.0, abs=1.0)
