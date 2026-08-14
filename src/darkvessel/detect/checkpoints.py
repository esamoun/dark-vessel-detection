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
        """Yield a path to write this epoch's state to, and put it in place once it is whole.

        Anything that goes wrong inside — an interrupt, a full disk, an out-of-memory on the way
        to `state_dict` — leaves the directory exactly as it was, holding the last epoch that
        did finish.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self.directory / f"epoch-{epoch:03d}.pt"
        partial = final.with_suffix(".pt.partial")

        try:
            yield partial
            os.replace(partial, final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(self.path.suffix + ".partial")
        partial.write_text(json.dumps([*self.entries(), entry], indent=2))
        os.replace(partial, self.path)
