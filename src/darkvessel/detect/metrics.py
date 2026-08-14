"""What the detector found, and what it missed.

Two decisions about the counting sit here, and both of them change the number that ends up in
the README.

A detection is credited with a ship when it lands within a stated distance of that ship's
centre, not when their boxes overlap by some fraction. Overlap is the wrong instrument at this
resolution: a 60 m vessel is six pixels at 10 m, so a box off by two pixels — a fifth of a hull
— already fails at half overlap, and the score would then be measuring the model's regression of
a box nobody in this chain ever uses. What the chain uses is the *point*: a detection becomes a
coordinate on the ground and is compared against a declared position within a tolerance in
metres. Scoring the detector by the same rule the fusion will apply to it measures the thing
that matters downstream rather than a proxy for it. It is also the more generous rule, and that
is stated rather than hidden — `tolerance_px` reads the distance off the config in metres so
that whoever reads a precision can see how far a hit was allowed to be.

One ship can be found once. Detections are matched in order of confidence and claim a ship
exclusively, so a detector that returns the same hull four times gets one hit and three false
alarms — which is what those four would be to anyone sent out to look at them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from darkvessel.detect.dataset import Box
from darkvessel.detect.detector import PixelDetection


@dataclass(frozen=True)
class Reporting:
    """How a held-out split is scored, in the units the chain downstream works in.

    Here rather than beside the training loop for two reasons. It is a statement about counting,
    which is this module's subject; and it imports nothing, so the command that reads it out of
    a config file can be checked on a laptop with no framework installed — which is the whole
    point of that command existing separately from the one that trains.
    """

    tolerance_m: float
    resolution_m: float
    thresholds: tuple[float, ...]

    def tolerance(self) -> float:
        """The same distance the fusion will use, in the pixels the detector works in."""
        return tolerance_px(self.tolerance_m, self.resolution_m)


@dataclass(frozen=True)
class Counts:
    """The three ways a run ends up, at one operating point."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Of what was reported, how much was there. NaN when nothing was reported: a run that
        returned nothing was neither right nor wrong, and calling that 100% precise is the most
        flattering thing this module could do."""
        return _ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Of what was there, how much was reported. NaN over an empty sea, for the same reason."""
        return _ratio(self.true_positives, self.true_positives + self.false_negatives)

    def line(self, threshold: float) -> str:
        return (
            f"score >= {threshold:.2f}: precision {self.precision:.3f}, recall {self.recall:.3f} "
            f"({self.true_positives} found, {self.false_positives} false, "
            f"{self.false_negatives} missed)"
        )


@dataclass(frozen=True)
class Attempt:
    """Every detection made over a split, and whether it found a ship no other detection had.

    Held as the raw outcome rather than as counts, because the counts depend on where the score
    threshold is put and that is a choice made when reporting, not when detecting. Matching runs
    once, in order of confidence; thresholds are then prefixes of that order, which is what makes
    precision and recall move the way a reader expects as the threshold is raised.
    """

    scores: tuple[float, ...]
    found_a_ship: tuple[bool, ...]
    ships: int

    def __add__(self, other: "Attempt") -> "Attempt":
        """Add another tile's attempt. Matching is tile-local — a detection in one 800 px cut
        cannot explain a ship in another — but the number reported is for the whole split."""
        return Attempt(
            scores=self.scores + other.scores,
            found_a_ship=self.found_a_ship + other.found_a_ship,
            ships=self.ships + other.ships,
        )

    def at(self, threshold: float) -> Counts:
        """The counts for a detector that reported only what it was this confident about."""
        kept = [
            found
            for score, found in zip(self.scores, self.found_a_ship, strict=True)
            if score >= threshold
        ]
        true_positives = sum(kept)
        return Counts(
            true_positives=true_positives,
            false_positives=len(kept) - true_positives,
            false_negatives=self.ships - true_positives,
        )

    def sweep(self, thresholds: Sequence[float]) -> list[tuple[float, Counts]]:
        """The same attempt read at several operating points.

        Reported as a table rather than as one number, because a detector does not have a
        precision — it has a precision at a threshold, and the threshold is a decision about how
        much of an inspection budget a false alarm is worth.
        """
        return [(threshold, self.at(threshold)) for threshold in thresholds]


NOTHING = Attempt(scores=(), found_a_ship=(), ships=0)


def measure(
    detections: Sequence[PixelDetection],
    ships: Sequence[Box],
    tolerance: float,
) -> Attempt:
    """Match one tile's detections against its labels, in order of confidence.

    Greedy, and knowingly so: the most confident detection takes the nearest ship still
    unclaimed, and a later detection that would have fitted that ship better goes without. An
    optimal assignment would score a point or two higher on tiles holding several ships closer
    together than the tolerance, and would make the number depend on a solver rather than on the
    detector. Greedy is also what an operator does with a ranked list.
    """
    centres = [ship.centre() for ship in ships]
    claimed: set[int] = set()
    found = [False] * len(detections)

    # Stable, so detections of equal confidence are taken in the order the detector reported them
    # rather than in an order that depends on the sort.
    for index in sorted(range(len(detections)), key=lambda i: -detections[i].score):
        detection = detections[index]
        nearest, distance = _nearest(detection, centres, claimed)
        if nearest is not None and distance <= tolerance:
            claimed.add(nearest)
            found[index] = True

    return Attempt(
        scores=tuple(detection.score for detection in detections),
        found_a_ship=tuple(found),
        ships=len(ships),
    )


def tolerance_px(tolerance_m: float, resolution_m: float) -> float:
    """How far a detection may sit from a ship and still be that ship, in pixels.

    Stated in metres in the config and converted here, so that the same physical tolerance
    survives a change of resolution instead of silently becoming twice as strict.
    """
    return tolerance_m / resolution_m


def _nearest(
    detection: PixelDetection,
    centres: Sequence[tuple[float, float]],
    claimed: set[int],
) -> tuple[int | None, float]:
    nearest, distance = None, math.inf
    for index, (row, col) in enumerate(centres):
        if index in claimed:
            continue
        candidate = math.hypot(detection.row - row, detection.col - col)
        if candidate < distance:
            nearest, distance = index, candidate

    return nearest, distance


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else math.nan
