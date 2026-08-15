"""What a run leaves on the disk, so the next session can pick it up.

Free-tier sessions end when the provider says so, not when the schedule does, and the design
follows from that: an epoch is the unit of progress, every epoch is written, and a session that
dies costs at most the epoch it was in. That much is ordinary. What is not ordinary is the
moment of writing — three hundred megabytes of weights take a while to reach the disk, and a
kernel stopped in the middle of it leaves a truncated file sitting under the name the next
session will resume from. So a checkpoint is written under a temporary name and moved into place
in one step: on this filesystem the move is atomic, and a checkpoint therefore either exists
whole or does not exist.

There is no torch here. Which file is the latest, whether a fragment can pass for a checkpoint,
and what is deleted to stay under the quota are questions about a directory, and keeping them on
this side of the seam is what lets them be tested in a second on a laptop with no GPU.
"""

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# `epoch-007.pt`, zero-padded so that a directory listing is in the order the epochs ran.
_CHECKPOINT = re.compile(r"^epoch-(\d+)\.pt$")

# What a half-written file is called while it is being written. Chosen so that `_CHECKPOINT`
# cannot match it: a fragment must not be able to pass for an epoch.
_PARTIAL = ".partial"


@contextmanager
def atomically(path: Path) -> Iterator[Path]:
    """Yield somewhere to write `path`, and put it there once it is whole.

    The one rule this module exists for, in the one place that states it. Anything that goes
    wrong inside — an interrupt, a full disk, an out-of-memory on the way to `state_dict` —
    leaves nothing behind and leaves whatever was already at `path` untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + _PARTIAL)

    try:
        yield partial
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


class Checkpoints:
    """The directory a run resumes from.

    `keep` is a disk budget, not a model-selection policy: the last few epochs are held because
    resuming needs the last one and a spare is cheap insurance, and the older ones go because a
    free tier gives 20 GB of working space against checkpoints of a third of a gigabyte. Picking
    the best epoch rather than the last is a different job, done against the metrics this run
    writes beside its weights.
    """

    def __init__(self, directory: Path, keep: int = 2) -> None:
        self.directory = directory
        self.keep = keep

    def path_for(self, epoch: int) -> Path:
        """Where this epoch's state ends up. Not where it is written — see `writing`."""
        return self.directory / f"epoch-{epoch:03d}.pt"

    def all(self) -> list[tuple[int, Path]]:
        """Every whole checkpoint in the directory, oldest first."""
        if not self.directory.exists():
            return []

        found = [
            (int(match.group(1)), path)
            for path in self.directory.iterdir()
            if (match := _CHECKPOINT.match(path.name))
        ]
        return sorted(found)

    def latest(self) -> tuple[int, Path] | None:
        """The epoch to resume from, and where its state is. None before the first one lands."""
        found = self.all()
        return found[-1] if found else None

    def next_epoch(self) -> int:
        """The epoch this session runs first. Epochs are counted from one, as they are reported."""
        latest = self.latest()
        return 1 if latest is None else latest[0] + 1

    @contextmanager
    def writing(self, epoch: int) -> Iterator[Path]:
        """Yield a path to write this epoch's state to, and keep it only if it is written whole.

        A session that dies inside here leaves the directory exactly as it was, holding the last
        epoch that did finish.
        """
        with atomically(self.path_for(epoch)) as partial:
            yield partial

        self._prune()

    def _prune(self) -> None:
        for _, path in self.all()[: -self.keep]:
            path.unlink(missing_ok=True)


class Journal:
    """The numbers a run reported, in a file that needs nothing to read.

    Written beside the weights and not inside them: the point of this ticket is a precision and a
    recall, and a reader should not need torch, a GPU or an unpickle to see them. Rewritten whole
    each time rather than appended to, so that a session killed mid-write cannot leave half a
    line that the next session reads back as a number.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return list(json.loads(self.path.read_text()))

    def record(self, entry: dict[str, Any]) -> None:
        history = [*self.entries(), entry]
        with atomically(self.path) as partial:
            partial.write_text(json.dumps(history, indent=2))
