"""Retrieval over the archive, and the checks that stop a collapsed representation shipping.

A representation that has learned nothing still returns neighbours: ranked, with similarities,
for every query. That is the failure this level has to be able to see, and everything here is
written against it — the twin recall a collapsed encoder scores at chance, the size agreement it
scores at chance, and the sheet a reader can look at.

No torch: what a ranking is, and what a check of one is, is arithmetic.
"""

import numpy as np
import pytest

from darkvessel.detect.amplitude import DecibelStretch
from darkvessel.embed.retrieval import (
    agreement,
    chance_of,
    contact_sheet,
    extent,
    neighbours,
    queries_over,
    retrieve,
    same_object,
    table,
    twin_recall,
    unit,
)

STRETCH = DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-21.0)


def separable(count: int = 12, dim: int = 4, seed: int = 0) -> np.ndarray:
    """Vectors in pairs: each even index and the odd one after it stand close together."""
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(count // 2, dim))
    return np.repeat(anchors, 2, axis=0) + rng.normal(scale=0.01, size=(count, dim))


def test_a_vector_of_length_zero_survives_normalisation_as_itself() -> None:
    """A collapse onto the origin has to come back as a bad answer, not as NaN."""
    normalised = unit(np.array([[3.0, 4.0], [0.0, 0.0]]))

    assert normalised[0] == pytest.approx([0.6, 0.8])
    assert normalised[1] == pytest.approx([0.0, 0.0])


def test_a_query_is_not_its_own_neighbour_and_its_twin_is_the_first_one() -> None:
    vectors = separable()

    assert neighbours(vectors, query=0, count=3)[0] == 1
    assert 0 not in neighbours(vectors, query=0, count=3)


def test_retrieval_carries_the_names_of_what_it_found() -> None:
    vectors = separable(count=4)
    names = ["a-0", "a-1", "b-0", "b-1"]

    found = retrieve(vectors, names, query=2, count=1)

    assert found.name == "b-0"
    assert found.found[0].name == "b-1"
    assert found.indices() == [2, 3]
    assert "b-0" in table([found])


def test_a_representation_that_places_a_second_view_first_scores_one() -> None:
    vectors = separable()
    # The twins are the same objects seen again: near, not identical.
    twins = vectors + np.random.default_rng(1).normal(scale=1e-3, size=vectors.shape)

    assert twin_recall(vectors, twins) == pytest.approx(1.0)


def test_a_collapsed_representation_scores_at_chance_rather_than_at_one() -> None:
    """Every crop mapped to the same point. Neighbours still come back, ranked, with
    similarities near one — which is exactly why the twin recall is the number that is kept."""
    collapsed = np.ones((20, 4))

    assert twin_recall(collapsed, collapsed) == pytest.approx(1.0 / 20, abs=1e-9)
    assert chance_of(np.eye(20, dtype=bool)) == pytest.approx(1.0 / 20)


def test_the_second_cut_of_a_hull_counts_as_the_hull_and_raises_the_bar_with_it() -> None:
    """A detector run at an archive's operating point cuts a large ship several times. Counting
    the second cut as a wrong answer measures the duplication rather than the representation —
    and the chance level has to move with the leniency, or the number is free."""
    vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    twins = vectors[[1, 0, 3, 2]]  # each twin lands on its neighbour rather than on itself
    same_as = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)

    assert twin_recall(vectors, twins) == pytest.approx(0.0)
    assert twin_recall(vectors, twins, same_as=same_as) == pytest.approx(1.0)
    assert chance_of(same_as) == pytest.approx(0.5)


def test_the_extent_of_a_crop_counts_the_object_and_not_the_sea() -> None:
    sea = np.full((3, 16, 16), -21.0, dtype=np.float32)
    sea += np.random.default_rng(0).normal(scale=0.5, size=sea.shape).astype(np.float32)
    sea[0, 7:9, 7:9] = 5.0  # four bright pixels
    sea[1, 6:10, 6:10] = 5.0  # sixteen

    counted = extent(sea, crop_px=16)

    assert counted[0] == 4
    assert counted[1] == 16
    assert counted[2] < 4


def test_neighbours_that_agree_about_size_beat_a_crop_drawn_at_random() -> None:
    # A representation that is exactly the object's size: its neighbours must agree about size,
    # and a random draw must not. The point of the baseline is that the first number alone is a
    # size rather than a result.
    sizes = np.array([1, 1, 5, 5, 20, 20, 60, 60])
    # Two dimensions, not one: cosine similarity reads a direction, and every positive number on
    # a line points the same way. The angle here is monotone in the size.
    vectors = np.stack([sizes.astype(float), np.ones(len(sizes))], axis=1)

    scored = agreement(vectors, sizes, seed=3)

    assert scored.retrieved == 0.0
    assert scored.chance > scored.retrieved
    assert "px of target" in scored.line()


def test_the_contact_sheet_draws_one_image_per_cell_and_captions_each_one() -> None:
    crops = np.random.default_rng(0).normal(loc=-21.0, scale=2.0, size=(6, 24, 24))
    crops = crops.astype(np.float32)
    rows = [
        retrieve(separable(count=6, dim=3), [f"c-{i}" for i in range(6)], query, 2)
        for query in (0, 2)
    ]

    svg = contact_sheet(crops, rows, stretch=STRETCH, crop_px=16)

    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<image") == 6  # two queries, two neighbours each
    # Two lines under each cell: what the crop is called, and how close it stood. Written side by
    # side they are wider than the cell, and a caption that overruns labels the crop next door.
    assert svg.count("<text") == 12
    assert svg.count(">query<") == 2
    assert "data:image/png;base64,iVBORw0KGgo" in svg  # the PNG signature, base64-encoded


def test_a_crop_index_outside_the_archive_is_refused() -> None:
    with pytest.raises(ValueError, match="archive of 12"):
        neighbours(separable(), query=99, count=1)


def test_retrieval_says_whether_it_found_the_object_or_only_the_weather() -> None:
    """The diagnostic a contact sheet cannot show.

    Four crops: two cuts of one hull in acquisition A, and two unrelated objects in B. The
    representation here pairs each cut with the other, which is the right answer; a representation
    that had learned the sea state would pair the two B crops with each other instead, and
    `elsewhere` is where that would show up.
    """
    vectors = np.array([[1.0, 0.0], [0.99, 0.02], [0.0, 1.0], [-0.02, 0.99]])
    same_as = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)

    found = same_object(vectors, same_as, ["A", "A", "B", "B"])

    assert found.retrieved == pytest.approx(0.5)  # the two cuts of the hull, and not the two in B
    assert found.elsewhere == pytest.approx(0.5)  # B's two crops retrieve each other
    # One of A's two crops is the other's own object, so a random *other* crop is the query's
    # object with probability 1/3 for each of A's rows and 0 for each of B's. The reading
    # `twin_recall` uses is a different number — the query is a candidate there — and passing
    # that one here would halve an apparent margin over chance.
    assert found.chance == pytest.approx(1 / 6)
    assert chance_of(same_as) == pytest.approx(0.375)
    assert "another cut" in found.line()


def test_size_agreement_is_measured_against_a_different_object_not_another_cut_of_the_same() -> (
    None
):
    """Two thirds of the nearest neighbours in the real archive are the query's own hull, cut
    twice. Measured over those, this figure restates the duplication rather than saying anything
    about resemblance between objects — so what it ranks over is everything the query is not."""
    sizes = np.array([10, 11, 90, 91])
    vectors = np.stack([sizes.astype(float), np.ones(len(sizes))], axis=1)
    # The two small crops are one object cut twice, and so are the two large ones.
    same_as = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)

    across = agreement(vectors, sizes, same_as=same_as, seed=1)

    # Nothing is left but the other object, some 80 px away — not the 1 px twin beside it.
    assert across.retrieved == pytest.approx(79.5)
    assert agreement(vectors, sizes, seed=1).retrieved == pytest.approx(1.0)
    assert "nearest different object" in across.line()


def test_an_archive_with_no_second_object_in_it_is_refused_rather_than_measured() -> None:
    """An argmax over nothing returns index zero, and a median of nothing is NaN with a warning —
    both of which reach a report looking like a measurement."""
    sizes = np.array([10, 11, 90])
    vectors = np.stack([sizes.astype(float), np.ones(len(sizes))], axis=1)
    # Every crop is the same object as every other: there is nothing else to retrieve at all.
    same_as = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="median of nothing"):
        agreement(vectors, sizes, same_as=same_as)


def test_the_queries_of_a_sheet_are_spread_over_the_sizes_the_archive_holds() -> None:
    """Deterministic and spread, because choosing six crops by hand is exactly where a
    flattering figure would come from."""
    sizes = np.array([0, 3, 5, 9, 20, 40, 80, 200])

    chosen = queries_over(sizes, 4)

    assert [int(sizes[index]) for index in chosen] == [0, 5, 40, 200]
    assert queries_over(sizes, 4) == chosen


def test_a_sheet_of_one_query_takes_the_middle_rather_than_dividing_by_zero() -> None:
    """The spacing below is a division by `count - 1`, so the branch written for a single query
    has to be taken before the arithmetic rather than after it."""
    sizes = np.array([0, 3, 5, 9, 20])

    assert queries_over(sizes, 1) == [2]
    assert queries_over(sizes, 9) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="nothing on it"):
        queries_over(sizes, 0)
