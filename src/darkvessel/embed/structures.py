"""Which detections are fixed structures, and by what evidence.

An offshore wind turbine is a bright point scatterer on water, which is what a ship is, and a
detector trained on ships returns turbines happily. Danish waters hold thousands of them. Every
one that reaches the fusion unexplained becomes a dark vessel — a claim someone may be sent out
on — so a chain that publishes dark vessels in these waters has to be able to say which of its
detections are not vessels at all.

Two signals are available and this module computes both, because they turn out to disagree about
how far they can be trusted and the disagreement is the finding.

The first is **recurrence**: a position that carries a detection acquisition after acquisition is
not a ship. A vessel under way is somewhere else a week later; a mast is not. It asks nothing of
the pixels and nothing of a label — only of the provenance the archive already keeps.

The second is the **embedding**, which is what issue #14 expected to do the work: turbines should
cluster apart from vessels, and the cluster be excluded wholesale. They do cluster apart, and
`describe` measures how far apart. What `docs/decisions.md` records, 2026-08-27, is that it is
not far enough to exclude on: at the operating point that costs no dark vessel the embedding
excludes almost nothing, and at the operating point that excludes the turbines it also excludes
a fifth of the control box, where there is no fixed structure at all. So the clustering is
reported and the register is built from recurrence.

Nothing here imports torch, and nothing here decides anything about a *run*. This module answers
what the archive holds; `fusion/register.py` carries the answer to a scene.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The same normalisation retrieval ranks under. Imported rather than rewritten: a clustering
# fitted in one geometry and a ranking read in another would agree on nothing, silently.
from darkvessel.embed.retrieval import unit

# How close two detections must be to count as the same standing object. Anholt's turbines stand
# some 600 m apart and a detection wobbles by a pixel or two, so 100 m separates neighbouring
# masts while holding successive sightings of one mast together.
#
# Deliberately *not* the fusion's 200 m match tolerance, which answers a different question:
# that number asks "could this declared position explain this detection", and this one asks "is
# this the same standing object". Six times the wobble and a sixth of the spacing.
SAME_POSITION_M = 100.0


@dataclass(frozen=True)
class Standing:
    """Every distinct position the archive holds a detection at, and how often it came back.

    `positions` is one row per position — where it is, how many distinct acquisitions saw it, and
    how many crops it accounts for. `of_crop` says which position each crop belongs to, by index,
    so a count over crops and a count over positions can never drift apart.
    """

    positions: pd.DataFrame
    of_crop: np.ndarray

    def acquisitions_of_crop(self) -> np.ndarray:
        """How many acquisitions the position each crop stands at was seen in, one per crop."""
        return self.positions["acquisitions"].to_numpy()[self.of_crop]

    def seen_in(self, floor: int) -> pd.DataFrame:
        """The positions seen in at least `floor` distinct acquisitions, most persistent first."""
        kept = self.positions[self.positions["acquisitions"] >= floor]
        return kept.sort_values("acquisitions", ascending=False).reset_index(drop=True)


def standing(provenance: pd.DataFrame, tolerance_m: float = SAME_POSITION_M) -> Standing:
    """Group the archive's detections into standing positions, and count their acquisitions.

    Greedy: every crop not yet claimed starts a position, and every unclaimed crop within the
    tolerance joins it. Stated as greedy rather than dressed up as clustering — for a lattice
    whose spacing is six times the tolerance the two agree, and where they would not, the
    honest description is the one that says which was run.

    An acquisition is counted by its scene name, which carries the box it was clipped to. Two
    clips of one Sentinel-1 product over two rectangles are two scenes here, which is right while
    the rectangles are disjoint and would double-count a position standing in both if they ever
    overlapped. `configs/embeddings.yaml`'s two boxes are 100 km apart.
    """
    x = provenance["x"].to_numpy(dtype=float)
    y = provenance["y"].to_numpy(dtype=float)
    scenes = provenance["scene"].to_numpy()

    unclaimed = np.ones(len(x), dtype=bool)
    of_crop = np.full(len(x), -1, dtype=np.int64)
    rows = []
    for seed in range(len(x)):
        if not unclaimed[seed]:
            continue
        near = unclaimed & (np.hypot(x - x[seed], y - y[seed]) <= tolerance_m)
        unclaimed &= ~near
        of_crop[near] = len(rows)
        rows.append(
            {
                # The centre of the sightings rather than the first of them: a mast detected
                # forty times is located better by forty detections than by whichever one the
                # loop happened to reach first.
                "x": float(x[near].mean()),
                "y": float(y[near].mean()),
                "acquisitions": len(set(scenes[near])),
                "crops": int(near.sum()),
            }
        )

    return Standing(
        positions=pd.DataFrame(rows, columns=["x", "y", "acquisitions", "crops"]).astype(
            {"x": float, "y": float, "acquisitions": int, "crops": int}
        ),
        of_crop=of_crop,
    )


@dataclass(frozen=True)
class Clustering:
    """A partition of the archive's embeddings, and the direction each cluster points in."""

    labels: np.ndarray
    centres: np.ndarray

    def __len__(self) -> int:
        return len(self.centres)


def cluster(vectors: np.ndarray, count: int, seed: int, rounds: int = 200) -> Clustering:
    """`count` clusters over the embeddings, by cosine — the geometry the loss was written in.

    Spherical k-means: the vectors are normalised, similarity is a dot product and a centre is
    the normalised mean of its members. Ranking by anything else would rank by a quantity the
    contrastive fit never optimised, the way `retrieval.neighbours` says.

    k-means++ for the seeds, from `seed` and nothing else, so two runs of the command return the
    same partition. Written here in twenty lines of numpy rather than taken from scikit-learn,
    which this project does not depend on and would not install for one function.
    """
    if count < 1:
        raise ValueError(f"a partition into {count} clusters is not a partition")
    if len(vectors) < count:
        raise ValueError(f"{len(vectors)} crops cannot be divided into {count} clusters")

    points = unit(np.asarray(vectors, dtype=np.float64))
    generator = np.random.default_rng(seed)

    centres = [points[generator.integers(len(points))]]
    while len(centres) < count:
        # Squared cosine distance to the nearest centre so far, which is k-means++ in the
        # geometry this runs in. Clipped at zero: a similarity of 1 + 1e-16 is a rounding error
        # and a negative probability is an exception.
        apart = np.clip(1.0 - (points @ np.asarray(centres).T).max(axis=1), 0.0, None)
        if not apart.any():
            # Every remaining point sits on a centre already. Duplicated crops do this, and the
            # honest answer is fewer distinct centres rather than a division by zero.
            break
        centres.append(points[generator.choice(len(points), p=apart / apart.sum())])

    moving = unit(np.asarray(centres))
    for _ in range(rounds):
        labels = np.argmax(points @ moving.T, axis=1)
        stepped = unit(
            np.stack(
                [
                    points[labels == index].mean(axis=0) if (labels == index).any() else centre
                    for index, centre in enumerate(moving)
                ]
            )
        )
        if np.allclose(stepped, moving):
            break
        moving = stepped

    return Clustering(labels=np.argmax(points @ moving.T, axis=1), centres=moving)


def describe(clustering: Clustering, standing_at: Standing, floor: int) -> pd.DataFrame:
    """What each cluster is made of, by the one property that needs no label: does it stand still.

    One row per cluster, and the column to read is `persistent` — the share of a cluster's crops
    that sit at a position seen in `floor` or more acquisitions. A cluster of turbines is near 1.
    A cluster of passing ships is near 0. Nothing in this table was labelled by anyone.

    It is a description and not a decision. `docs/decisions.md`, 2026-08-27, records what happened
    when it was used as one.
    """
    persistent = standing_at.acquisitions_of_crop() >= floor
    rows = [
        {
            "cluster": index,
            "crops": int((clustering.labels == index).sum()),
            "persistent": float(persistent[clustering.labels == index].mean())
            if (clustering.labels == index).any()
            else float("nan"),
        }
        for index in range(len(clustering))
    ]
    return pd.DataFrame(rows, columns=["cluster", "crops", "persistent"])


def separation(vectors: np.ndarray, standing_at: Standing, floor: int) -> float:
    """How well the embedding alone orders the persistent crops ahead of the rest, in 0..1.

    The area under the ROC curve of one score: each crop's cosine similarity to the centre of the
    crops recurrence is sure about. 0.5 is a representation that knows nothing about the
    difference, 1.0 one that separates them completely. The centre is built from recurrence and
    not from a label, so this is a check on the representation rather than on an annotation.

    It is the fairest single number for the ticket's first criterion, because it needs no
    threshold — and a threshold is exactly where the embedding turns out to fail on this archive.
    A ranking can be good while every cut through it is bad, and `docs/decisions.md`, 2026-08-27,
    is the measurement of that happening.
    """
    persistent = standing_at.acquisitions_of_crop() >= floor
    if not persistent.any() or persistent.all():
        raise ValueError(
            f"every crop here is on the same side of a floor of {floor} acquisitions, so there "
            "is nothing for a ranking to separate"
        )

    points = unit(np.asarray(vectors, dtype=np.float64))
    scores = points @ unit(points[persistent].mean(axis=0)[None, :])[0]
    return _auc(scores, persistent)


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve, by ranks. Ties share a rank rather than being broken arbitrarily.

    By hand for the reason `retrieval._as_png` is written by hand: this is one formula, and a
    dependency on scikit-learn to evaluate it would be a dependency the chain carries for ever.
    """
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average the ranks inside each run of equal scores, which is what makes a representation
    # that maps everything to one point score 0.5 rather than whatever the sort order was.
    ordered = scores[order]
    start = 0
    for end in range(1, len(ordered) + 1):
        if end == len(ordered) or ordered[end] != ordered[start]:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    positives = int(positive.sum())
    negatives = len(scores) - positives
    ordered_positives = ranks[positive].sum() - positives * (positives + 1) / 2
    return float(ordered_positives / (positives * negatives))


@dataclass(frozen=True)
class Verified:
    """A register of fixed positions, checked against coordinates somebody else published.

    Both directions, because either one alone can be made to look good. `found` of the `known`
    positions says the register did not miss the farm; `unpublished` of the register's own says the
    register is not full of things nobody has ever recorded — and where it is, that is a finding
    rather than an error, since the sea holds fixed structures no turbine list mentions.
    """

    known: int
    registered: int
    found: int
    unpublished: int
    median_m: float
    tolerance_m: float

    def line(self) -> str:
        return (
            f"{self.found} of {self.known} published positions carry a registered structure and "
            f"{self.registered - self.unpublished} of {self.registered} registered "
            f"structures stand at a published position, {self.median_m:.1f} m apart at the "
            f"median, "
            f"within {self.tolerance_m:g} m"
        )


def verify(registered: pd.DataFrame, known: pd.DataFrame, tolerance_m: float) -> Verified:
    """Check a register of fixed positions against published coordinates for the same water.

    Both frames carry `x` and `y` in the same CRS; nothing is reprojected here, because a
    reprojection this function performed silently is exactly how two sets of coordinates come to
    be compared in two different metres.

    The tolerance is not the fusion's and not `SAME_POSITION_M`. It answers a third question —
    "is this the structure that list is talking about" — against a published position whose own
    error is unknown to this project and, for the source this project uses, is documented by that
    source as approximate.
    """
    if known.empty:
        raise ValueError(
            "a verification against no published positions is not a verification; the box has "
            "nothing to check the register against, which is a statement about the reference "
            "rather than a score the register can be given"
        )
    if registered.empty:
        # A method that found none of the farm is the failure this check exists to catch, so it
        # has to come back as a number rather than as an exception. `median_m` is infinite and
        # not zero: there is no pair of positions to measure between, and zero would read as
        # perfect agreement.
        return Verified(
            known=len(known),
            registered=0,
            found=0,
            unpublished=0,
            median_m=float("inf"),
            tolerance_m=float(tolerance_m),
        )

    apart = np.hypot(
        registered["x"].to_numpy(dtype=float)[:, None] - known["x"].to_numpy(dtype=float)[None, :],
        registered["y"].to_numpy(dtype=float)[:, None] - known["y"].to_numpy(dtype=float)[None, :],
    )
    nearest_known = apart.min(axis=1)
    nearest_registered = apart.min(axis=0)

    return Verified(
        known=len(known),
        registered=len(registered),
        found=int((nearest_registered <= tolerance_m).sum()),
        unpublished=int((nearest_known > tolerance_m).sum()),
        median_m=float(np.median(nearest_known)),
        tolerance_m=float(tolerance_m),
    )


def table(described: pd.DataFrame) -> str:
    """The cluster description as lines, in the shape `curve.table` and `retrieval.table` have."""
    return "\n".join(
        f"  cluster {int(row.cluster):>2}  {int(row.crops):>5} crops  "
        f"{row.persistent:.3f} of them at a standing position"
        for row in described.itertuples()
    )
