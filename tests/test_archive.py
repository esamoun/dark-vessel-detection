"""The file the crops accumulate in.

An archive is written once per scene and read by everything after it, over sessions that are not
the same session. What has to survive that is the correspondence between a crop and where it came
from, and the geometry it was cut at — the two things whose loss produces an archive that still
loads, still trains and answers a different question from the one it is asked.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darkvessel.embed.archive import Archive


def archive(count: int = 3, scene: str = "S1A", crop_px: int = 4, margin_px: int = 2) -> Archive:
    side = crop_px + 2 * margin_px
    return Archive(
        crops=np.arange(count * side * side, dtype=np.float32).reshape(count, side, side),
        provenance=pd.DataFrame(
            {
                "scene": [scene] * count,
                "acquired_at": ["2026-08-09T05:31:24+00:00"] * count,
                "row": np.arange(count, dtype=float),
                "col": np.arange(count, dtype=float) * 2,
                # A kilometre apart, so that nothing in a fixture is accidentally the same
                # object as anything else — see `co_located`.
                "x": np.arange(count, dtype=float) * 1000.0 + 619_190.0,
                "y": np.arange(count, dtype=float) * 1000.0 + 6_397_640.0,
                "score": np.linspace(0.1, 0.9, count),
            }
        ),
        crop_px=crop_px,
        margin_px=margin_px,
    )


def test_an_archive_survives_being_written_and_read(tmp_path: Path) -> None:
    written = archive()

    written.write(tmp_path / "crops.npz")
    read = Archive.read(tmp_path / "crops.npz")

    assert np.array_equal(read.crops, written.crops)
    pd.testing.assert_frame_equal(read.provenance, written.provenance)
    assert (read.crop_px, read.margin_px) == (4, 2)


def test_a_crop_nobody_can_place_is_refused() -> None:
    with pytest.raises(ValueError, match="provenance"):
        Archive(
            crops=np.zeros((3, 8, 8), dtype=np.float32),
            provenance=archive(count=2).provenance,
            crop_px=4,
            margin_px=2,
        )


def test_two_archives_cut_at_different_geometries_are_not_one_archive() -> None:
    """An encoder fitted across both would be fitted at two scales and would say so nowhere —
    the same class of silent fault as a checkpoint loaded with the wrong anchors."""
    with pytest.raises(ValueError, match="two scales"):
        archive().with_more(archive(crop_px=8))


def test_archives_from_two_acquisitions_join_and_still_say_which_is_which() -> None:
    joined = archive(count=2, scene="S1A").with_more(archive(count=3, scene="S1B"))

    assert len(joined) == 5
    assert joined.scenes() == ["S1A", "S1B"]


def test_an_interrupted_write_leaves_the_archive_that_was_already_there(tmp_path: Path) -> None:
    """The rule `checkpoints.atomically` exists for, applied to the one file this level grows.

    An archive is appended to across sessions, so a process killed part way through writing it
    would otherwise leave a truncated file under the name the next session reads.
    """
    path = tmp_path / "crops.npz"
    archive(count=2).write(path)

    broken = archive(count=3)
    # An array of things that cannot be pickled fails inside the write rather than before it,
    # which is the case the atomic move exists for.
    object.__setattr__(broken, "crops", np.full((3, 8, 8), lambda: None, dtype=object))
    with pytest.raises((ValueError, TypeError, AttributeError)):
        broken.write(path)

    assert len(Archive.read(path)) == 2
    assert not list(tmp_path.glob("*.partial"))


def test_two_cuts_of_one_hull_are_one_object_and_two_scenes_are_never_one() -> None:
    """A detector run at an archive's operating point cuts a large ship several times, and a
    check that called the second cut a wrong answer would measure the duplication rather than
    the representation. Two acquisitions of the same water are never the same object, however
    close the two positions are: one vessel has moved, or two vessels have passed."""
    close = archive(count=2)
    close.provenance.loc[1, ["x", "y"]] = close.provenance.loc[0, ["x", "y"]].to_numpy() + 30.0

    together = close.co_located(tolerance_m=200.0)
    across = close.with_more(archive(count=2, scene="S1B")).co_located(tolerance_m=200.0)

    assert together.tolist() == [[True, True], [True, True]]
    assert close.co_located(tolerance_m=10.0).tolist() == [[True, False], [False, True]]
    assert not across[0, 2] and not across[1, 3]
