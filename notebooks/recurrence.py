"""Does the archive contain fixed structures at all, before any method is written to find them?

Issue #14 asks that fixed structures cluster separately in the embedding space and be verified
against known offshore wind farm locations. That is unanswerable on an archive containing none,
and the archive built for issue #13 contained none: the study area moved onto the shipping lane
in August precisely because Anholt had turbines and no ships. The Anholt box was added back for
this reason, and this is the check that it worked — run before the clustering is written, so that
a null result is a fact about the data rather than a verdict on a method.

It asks nothing of the embedding. It asks of the *positions* alone whether the archive holds
objects that come back to the same patch of sea acquisition after acquisition. A ship under way
does not. A turbine does, and Anholt has a documented 111 of them in a lattice.

The Kattegat lane is the control, and it is a real one: same detector, same score threshold, same
ten weeks of dates, same crop geometry. The only difference is the water. If recurrence separates
the two boxes, the missing half of the problem is in the archive.

The prediction, written before this was run: Anholt shows tens of positions standing across most
of its acquisitions, and the lane shows almost none. What it will *not* settle is how many of the
111 turbines are found — that needs the farm's real coordinates, and it is #14's second criterion
rather than this check's.

Costs no GPU and no network. It reads the provenance of the archive, never the pixels.

    python3 notebooks/recurrence.py
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from darkvessel.embed.archive import Archive

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive" / "crops.npz"

# Anholt's turbines stand some 600 m apart and a detection wobbles by a pixel or two, so this
# separates neighbouring masts while holding successive sightings of one mast together. It is
# deliberately *not* the fusion's 200 m match tolerance: that number answers "could this declared
# position explain this detection", and this one answers "is this the same standing object".
TOLERANCE_M = 100.0

# How many acquisitions a position has to be seen in before it is reported as standing. Reported
# at several floors rather than one, because a single threshold invites the reader to take it as
# a definition of "fixed" — which it is not, and which is #14's job to establish.
FLOORS = (2, 5, 10, 20)


def standing_positions(x: np.ndarray, y: np.ndarray, scenes: np.ndarray) -> list[int]:
    """How many distinct acquisitions each standing position was detected in, largest first.

    Greedy: every crop not yet claimed starts a position, and every unclaimed crop within the
    tolerance joins it. Good enough for a lattice whose spacing is six times the tolerance, and
    stated as greedy rather than dressed up as clustering — the clustering is #14.
    """
    unclaimed = np.ones(len(x), dtype=bool)
    seen = []
    for seed in range(len(x)):
        if not unclaimed[seed]:
            continue
        near = unclaimed & (np.hypot(x - x[seed], y - y[seed]) <= TOLERANCE_M)
        unclaimed &= ~near
        seen.append(len(set(scenes[near])))
    return sorted(seen, reverse=True)


def report(archive: Archive, boxes: Sequence[str]) -> None:
    print(f"{len(archive)} crops from {len(archive.scenes())} scenes")
    for box in boxes:
        rows = archive.provenance[archive.provenance["scene"].str.startswith(f"{box}/")]
        if rows.empty:
            print(f"\n{box}: no crops in the archive")
            continue

        seen = standing_positions(
            rows["x"].to_numpy(), rows["y"].to_numpy(), rows["scene"].to_numpy()
        )
        print(
            f"\n{box}: {len(rows)} crops over {rows['scene'].nunique()} acquisitions, "
            f"{len(seen)} standing positions"
        )
        for floor in FLOORS:
            print(f"    seen in {floor:>2}+ acquisitions: {sum(c >= floor for c in seen):>5}")
        print(f"    most persistent position: {seen[0]} acquisitions")
        print(
            f"    crops at a position seen {FLOORS[1]}+ times: "
            f"{sum(c for c in seen if c >= FLOORS[1])} of {len(rows)}"
        )


if __name__ == "__main__":
    archive = Archive.read(ARCHIVE)
    report(archive, sorted({name.split("/")[0] for name in archive.scenes()}))
