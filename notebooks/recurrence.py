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

from darkvessel.embed.archive import Archive
from darkvessel.embed.structures import SAME_POSITION_M, standing

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive" / "crops.npz"

# How many acquisitions a position has to be seen in before it is reported as standing. Reported
# at several floors rather than one, because a single threshold invites the reader to take it as
# a definition of "fixed" — which it is not, and which was #14's job to establish. It since has:
# `configs/embeddings.yaml` sets the floor at 20, and the argument for that number is that it is
# the lowest one at which every registered structure stands on a published one.
FLOORS = (2, 5, 10, 20)


def report(archive: Archive, boxes: Sequence[str]) -> None:
    print(f"{len(archive)} crops from {len(archive.scenes())} scenes")
    for box in boxes:
        rows = archive.provenance[archive.provenance["scene"].str.startswith(f"{box}/")]
        if rows.empty:
            print(f"\n{box}: no crops in the archive")
            continue

        # The package's own grouping, not a second copy of it living in a notebook. This file was
        # written before `embed/structures.py` existed and held the only implementation; keeping
        # its own would mean the pre-check and the method could drift, and the whole value of the
        # pre-check is that it answers the same question the method does.
        found = standing(rows, tolerance_m=SAME_POSITION_M)
        seen = sorted(found.positions["acquisitions"], reverse=True)
        print(
            f"\n{box}: {len(rows)} crops over {rows['scene'].nunique()} acquisitions, "
            f"{len(seen)} standing positions"
        )
        for floor in FLOORS:
            print(f"    seen in {floor:>2}+ acquisitions: {sum(c >= floor for c in seen):>5}")
        print(f"    most persistent position: {seen[0]} acquisitions")
        print(
            f"    crops at a position seen {FLOORS[1]}+ times: "
            f"{found.positions.loc[found.positions['acquisitions'] >= FLOORS[1], 'crops'].sum()} "
            f"of {len(rows)}"
        )


if __name__ == "__main__":
    archive = Archive.read(ARCHIVE)
    report(archive, sorted({name.split("/")[0] for name in archive.scenes()}))
