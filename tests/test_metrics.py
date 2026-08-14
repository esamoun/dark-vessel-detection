"""What the detector is worth, counted the same way twice.

This is the file that decides whether a number in the README is a finding or a flattering
accident, so the arithmetic is pinned here on cases small enough to count by hand. The
detector's own numbers are not asserted anywhere — a test that fixes a precision turns a
measurement into a target, and the first thing anyone does with a failing target is move it.
What is asserted is that the counting is honest: one ship cannot be found twice, a ship nobody
detected is missing rather than absent, and demanding more confidence never invents a ship.
"""

import math

import pytest

from darkvessel.detect.dataset import Box
from darkvessel.detect.detector import PixelDetection
from darkvessel.detect.metrics import Counts, measure, tolerance_px


def ship_at(row: float, col: float) -> Box:
    """A four-pixel hull, which is about what a 40 m vessel is at 10 m."""
    return Box(min_row=row - 2, min_col=col - 2, max_row=row + 2, max_col=col + 2)


def seen_at(row: float, col: float, score: float = 1.0) -> PixelDetection:
    return PixelDetection(row=row, col=col, score=score)


def test_a_detection_on_every_ship_and_nothing_else_is_perfect() -> None:
    attempt = measure(
        [seen_at(9.5, 9.5), seen_at(49.5, 79.5)], [ship_at(10, 10), ship_at(50, 80)], 5.0
    )

    assert attempt.at(0.0) == Counts(true_positives=2, false_positives=0, false_negatives=0)


def test_a_detection_beyond_the_tolerance_costs_twice() -> None:
    """Once as a false alarm and once as a ship nobody found. Counting it only as a false alarm
    would let a detector that pointed at the wrong half of the scene keep its recall."""
    attempt = measure([seen_at(40, 40)], [ship_at(10, 10)], 5.0)

    assert attempt.at(0.0) == Counts(true_positives=0, false_positives=1, false_negatives=1)


def test_one_ship_cannot_be_found_twice() -> None:
    """A detector that returns the same hull four times is not four times as good. The extra
    three are false alarms, which is what they would be to anyone sent out to look."""
    attempt = measure([seen_at(9.5, 9.5), seen_at(10.5, 10.0)], [ship_at(10, 10)], 5.0)

    assert attempt.at(0.0) == Counts(true_positives=1, false_positives=1, false_negatives=0)


def test_the_confident_detection_is_the_one_credited_with_the_ship() -> None:
    """Which of two detections on one hull gets the credit decides whether the numbers behave.

    Credit the unconfident one and raising the threshold deletes a true positive and promotes
    the confident detection to a false alarm: the detector appears to get worse the more
    confidence is demanded of it, which is an artefact of the counting and not a property of
    anything. Matching runs in order of score for that reason.
    """
    attempt = measure(
        [seen_at(10.0, 10.0, score=0.2), seen_at(11.0, 10.0, score=0.9)],
        [ship_at(10, 10)],
        5.0,
    )

    assert attempt.at(0.5) == Counts(true_positives=1, false_positives=0, false_negatives=0)


def test_demanding_more_confidence_never_finds_more_ships() -> None:
    attempt = measure(
        [
            seen_at(9.5, 9.5, score=0.9),
            seen_at(50.0, 80.0, score=0.3),
            seen_at(70.0, 20.0, score=0.6),
        ],
        [ship_at(10, 10), ship_at(50, 80)],
        5.0,
    )

    found = [attempt.at(threshold).true_positives for threshold in (0.0, 0.4, 0.7, 0.95)]

    assert found == sorted(found, reverse=True)


def test_a_ship_nobody_looked_for_is_missing_not_absent() -> None:
    attempt = measure([], [ship_at(10, 10), ship_at(50, 80)], 5.0)

    assert attempt.at(0.0) == Counts(true_positives=0, false_positives=0, false_negatives=2)
    assert attempt.at(0.0).recall == 0.0


def test_nothing_detected_has_no_precision_rather_than_a_perfect_one() -> None:
    """A run that returned nothing at this threshold was neither right nor wrong, and reporting
    it as 100% precise is the single most flattering thing this file could do."""
    precision = measure([], [ship_at(10, 10)], 5.0).at(0.0).precision

    assert math.isnan(precision)


def test_an_empty_sea_can_only_lose() -> None:
    attempt = measure([seen_at(10, 10)], [], 5.0)

    assert attempt.at(0.0) == Counts(true_positives=0, false_positives=1, false_negatives=0)
    assert math.isnan(attempt.at(0.0).recall)


def test_tiles_add_up_into_one_score_for_the_split() -> None:
    """Matching is tile-local — a detection in one 800 px cut cannot explain a ship in another —
    but the number reported is for the whole held-out split, so the attempts are added."""
    first = measure([seen_at(9.5, 9.5)], [ship_at(10, 10)], 5.0)
    second = measure([seen_at(40, 40)], [ship_at(10, 10)], 5.0)

    together = first + second

    assert together.at(0.0) == Counts(true_positives=1, false_positives=1, false_negatives=1)
    assert together.ships == 2


def test_the_tolerance_is_a_distance_on_the_ground_read_in_pixels() -> None:
    """The detector works in pixels and the chain it feeds matches in metres. Stating the
    tolerance in metres is what keeps the two comparable when the resolution changes."""
    assert tolerance_px(200.0, resolution_m=10.0) == 20.0


@pytest.mark.parametrize(
    ("counts", "precision", "recall"),
    [
        (Counts(true_positives=3, false_positives=1, false_negatives=0), 0.75, 1.0),
        (Counts(true_positives=3, false_positives=0, false_negatives=1), 1.0, 0.75),
    ],
)
def test_precision_and_recall_are_the_two_ways_of_being_wrong(
    counts: Counts, precision: float, recall: float
) -> None:
    assert (counts.precision, counts.recall) == (precision, recall)
