"""Surviving the session, rather than the schedule.

A free-tier GPU session ends when it ends: Kaggle stops the kernel, the browser tab is closed,
the weekly quota runs out mid-epoch. The whole training design assumes that, so the thing that
has to be right is not the loop but what is on the disk when the loop stops — and the moment it
can be wrong is the one where the process dies halfway through writing 300 MB of weights.

Nothing here imports torch, deliberately. What is fragile about resuming is the bookkeeping —
which file is the latest, whether a half-written one can be mistaken for a whole one, what gets
deleted to stay under the disk quota — and none of that is about tensors. Kept on this side of
the seam, it is covered by a suite that runs on a laptop in a second.
"""

from pathlib import Path

import pytest

from darkvessel.detect.checkpoints import Checkpoints, Journal


def test_a_run_that_has_never_started_starts_at_the_first_epoch(tmp_path: Path) -> None:
    checkpoints = Checkpoints(tmp_path / "run")

    assert checkpoints.latest() is None
    assert checkpoints.next_epoch() == 1


def test_a_run_resumes_after_the_last_epoch_it_finished(tmp_path: Path) -> None:
    checkpoints = Checkpoints(tmp_path / "run")
    for epoch in (1, 2, 3):
        with checkpoints.writing(epoch) as path:
            path.write_bytes(b"weights")

    latest = checkpoints.latest()

    assert latest is not None and latest[0] == 3
    assert checkpoints.next_epoch() == 4


def test_a_session_killed_mid_write_leaves_the_last_good_epoch_standing(tmp_path: Path) -> None:
    """The fault this whole module exists for. Weights are hundreds of megabytes and a kernel
    stopped halfway through leaves a truncated file; named as a checkpoint, it becomes the one
    the next session resumes from, and the run continues from a state that was never valid.
    """
    checkpoints = Checkpoints(tmp_path / "run")
    with checkpoints.writing(1) as path:
        path.write_bytes(b"weights")

    with pytest.raises(KeyboardInterrupt):
        with checkpoints.writing(2) as path:
            path.write_bytes(b"half of th")
            raise KeyboardInterrupt

    latest = checkpoints.latest()
    assert latest is not None and latest[0] == 1
    assert latest[1].read_bytes() == b"weights"
    assert checkpoints.next_epoch() == 2


def test_nothing_a_failed_write_left_behind_is_still_on_the_disk(tmp_path: Path) -> None:
    """A free tier gives 20 GB of working space and a checkpoint is a third of a gigabyte. A
    fragment kept from every interrupted session fills that up over a week of evenings."""
    checkpoints = Checkpoints(tmp_path / "run")

    with pytest.raises(RuntimeError):
        with checkpoints.writing(1) as path:
            path.write_bytes(b"half of th")
            raise RuntimeError("out of memory")

    assert list((tmp_path / "run").iterdir()) == []


def test_only_the_last_few_epochs_are_kept(tmp_path: Path) -> None:
    """Resuming needs the last one. The rest are 300 MB each against a 20 GB quota, and the run
    that picks the best of them is a different job — see docs/decisions.md."""
    checkpoints = Checkpoints(tmp_path / "run", keep=2)

    for epoch in range(1, 6):
        with checkpoints.writing(epoch) as path:
            path.write_bytes(b"weights")

    assert sorted(epoch for epoch, _ in checkpoints.all()) == [4, 5]


def test_the_numbers_a_run_reported_outlive_the_session_that_reported_them(tmp_path: Path) -> None:
    """Metrics are the output of this ticket, and they are written where they can be read
    without a GPU, without torch and without unpickling a checkpoint."""
    journal = Journal(tmp_path / "run" / "metrics.json")
    journal.record({"epoch": 1, "recall": 0.31})
    journal.record({"epoch": 2, "recall": 0.44})

    assert Journal(tmp_path / "run" / "metrics.json").entries() == [
        {"epoch": 1, "recall": 0.31},
        {"epoch": 2, "recall": 0.44},
    ]


def test_a_journal_from_a_run_that_never_reported_anything_is_empty(tmp_path: Path) -> None:
    assert Journal(tmp_path / "run" / "metrics.json").entries() == []
