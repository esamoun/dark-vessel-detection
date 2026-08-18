"""Training loop.

Written for short, interruptible free-tier sessions: checkpoint every epoch, resume from the
last checkpoint, never assume the session survives to the end of the schedule.

The order inside an epoch is part of that and is not the obvious one. The weights are written
*before* the held-out split is scored, so a session that dies during evaluation costs the
numbers and not the epoch — a checkpoint is the expensive thing to lose and the metrics can be
recomputed from it, never the other way round. That is only true if something actually recomputes
them, so the first thing a session does is score any epoch whose weights landed and whose score
did not. Without it the last epoch of a finished schedule can end up on the disk with no numbers
against it and nothing left that would ever produce them.

What is deliberately absent is mixed precision. It is the obvious way to buy epochs on a T4, and
it adds a scaler whose state has to be saved and restored correctly on a machine that cannot run
the code path that uses it. An untested resume is a worse trade than a slower epoch; see
docs/decisions.md.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from darkvessel.detect.checkpoints import Checkpoints, Journal
from darkvessel.detect.dataset import Box, TileRef, symmetry_for
from darkvessel.detect.metrics import NOTHING, Attempt, Reporting, measure
from darkvessel.detect.model import SHIP, as_model_input, detections_from

# The two names `Schedule.lr_schedule` accepts. Kept as the one list both `__post_init__` and
# `_scheduler` read, so there is one place that decides what this project has rather than two
# lists that could quietly drift apart.
_LR_SCHEDULES = ("constant", "cosine")


@dataclass(frozen=True)
class Schedule:
    """How long the run is, and how hard it pulls."""

    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    workers: int
    seed: int
    # "constant" is what the first run used and what the ladder's baseline keeps. The failure log
    # records what it cost: twelve epochs that reached the neighbourhood of a minimum in three and
    # bounced around it for nine, while the training loss stayed nearly flat and said nothing.
    lr_schedule: str = "constant"

    def __post_init__(self) -> None:
        # Refused here rather than where `_scheduler` builds one, so a config with a mistyped
        # schedule name is caught the moment it is read rather than partway into a GPU session —
        # by which point `describe` has already written a run block for a run that never began,
        # and the corrected config is then refused *against* the typo it is fixing.
        if self.lr_schedule not in _LR_SCHEDULES:
            raise ValueError(
                f"unknown lr_schedule {self.lr_schedule!r}; this project has "
                f"{' and '.join(repr(name) for name in _LR_SCHEDULES)}"
            )


def _scheduler(
    optimiser: torch.optim.Optimizer, schedule: Schedule
) -> "torch.optim.lr_scheduler.LRScheduler | None":
    """How the rate moves across the schedule, or None if it does not.

    Cosine rather than steps because twelve epochs is not many and a `StepLR` would introduce two
    free parameters — where the step falls and how far it drops — that nothing here could justify.
    No warmup for the same reason: it is a third knob, and it becomes a rung of its own if the
    first three epochs turn out to need it.
    """
    if schedule.lr_schedule == "constant":
        return None
    # The only other name `Schedule.__post_init__` lets through.
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=schedule.epochs)


def train(
    *,
    model: torch.nn.Module,
    training: Sequence[TileRef],
    held_out: Sequence[TileRef],
    checkpoints: Checkpoints,
    journal: Journal,
    schedule: Schedule,
    reporting: Reporting,
    device: torch.device,
    built: dict[str, Any],
    stem: str = "repeat",
    say: Callable[[str], None] = print,
) -> None:
    """Run the schedule, or as much of it as this session gets through.

    `stem` is the input stage `model` was built with, and it has to be the one the tiles are cut
    for: a single-channel model takes one channel and the repeat takes three, and the tiles are
    made here rather than by the caller.

    `built` is what constructed `model` — its tile size, anchors and seed — and it is written
    into every checkpoint so that whatever loads one can refuse a model built differently. It is
    required rather than optional because the whole value of it is that no run can omit it.

    Picks up wherever the last session stopped. A run started fresh and a run resumed four times
    do the same epochs over the same tiles in the same order, because everything that would
    otherwise vary — which empty tiles the subset kept, which way each tile is laid down, the
    order they arrive in — is derived from the seed and the epoch number rather than from a
    generator's position in a stream.
    """
    model.to(device)

    # Written before the first epoch, so a metrics file says which configuration produced it. Five
    # rungs of a ladder are five of these files, and one that does not name its run compares to
    # nothing. `describe` refuses a resume under an edited config rather than merging two
    # experiments into one file — `epochs` included: under a cosine schedule the rate for every
    # remaining epoch is a function of how long the schedule was declared to be, so resuming under
    # a different length is a different experiment, not a longer one, and a checkpoint that
    # finished at `eta_min` would otherwise restart at a learning rate of zero and stay there,
    # silently.
    journal.describe(
        {
            "built": built,
            "stem": stem,
            "schedule": asdict(schedule),
            "reporting": {
                "tolerance_m": reporting.tolerance_m,
                "resolution_m": reporting.resolution_m,
                "thresholds": list(reporting.thresholds),
            },
            "training_tiles": len(training),
            "held_out_tiles": len(held_out),
        }
    )

    optimiser = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=schedule.learning_rate,
        momentum=schedule.momentum,
        weight_decay=schedule.weight_decay,
    )
    scheduler = _scheduler(optimiser, schedule)

    resumed = checkpoints.latest()
    if resumed is not None:
        epoch, path = resumed
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        # A scheduler left out of the checkpoint restarts from the top, and the resumed session
        # trains its remaining epochs at rates an uninterrupted run would never have used. None
        # in a checkpoint written by a constant-rate run, which is why this is guarded twice.
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        say(f"resuming after epoch {epoch}, from {path.name}")

        # The epoch whose weights survived the session that wrote them but whose score did not.
        # Scored now, from the checkpoint just loaded, because nothing else ever would: past the
        # end of the schedule the loop below does not run at all.
        if epoch not in {entry["epoch"] for entry in journal.entries()}:
            say(f"epoch {epoch} was never scored; scoring it from its checkpoint")
            _report(
                epoch,
                loss=None,
                learning_rate=None,
                attempt=_score(model, held_out, schedule, reporting, device, stem=stem),
                held_out=held_out,
                reporting=reporting,
                journal=journal,
                say=say,
            )

    first = checkpoints.next_epoch()
    if first > schedule.epochs:
        say(f"the schedule of {schedule.epochs} epochs is already finished; nothing to do")
        return

    say(
        f"epochs {first} to {schedule.epochs} on {device}, {len(training)} tiles, "
        f"batch {schedule.batch_size}"
    )

    for epoch in range(first, schedule.epochs + 1):
        # Read before the step below, so the journal records the rate this epoch was trained at
        # rather than the rate the next one will be.
        rate = optimiser.param_groups[0]["lr"]
        loss = _one_epoch(model, optimiser, training, epoch, schedule, device, stem=stem)

        # Stepped before the checkpoint is written, so what lands on the disk with epoch N is the
        # state a session resuming at epoch N+1 needs, and that session loads it and starts.
        if scheduler is not None:
            scheduler.step()

        # Before the scoring, not after: an interrupted evaluation costs the numbers, and the
        # numbers can be recomputed from the weights.
        with checkpoints.writing(epoch) as partial:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    # Not weights, and that is exactly the point. Anchor sizes leave no trace in
                    # a state dict — `AnchorGenerator` has no parameters — so a checkpoint that
                    # does not name them loads cleanly into a model looking for ships of another
                    # size and never says so. See docs/decisions.md.
                    "built": built,
                },
                partial,
            )
        # `path_for`, not the path `writing` yielded: that one is the temporary name, and by
        # here it has been renamed away. Reporting it told every run that its checkpoint was a
        # `.partial` file — the one thing this module exists to make impossible.
        landed = checkpoints.path_for(epoch)
        # `:.2e`, not `:.5f`: a cosine schedule's later epochs round to 0.00000 under a fixed
        # five decimal places, and that line is the only live feedback a session gets while it
        # runs.
        say(
            f"epoch {epoch}: loss {loss:.4f}, rate {rate:.2e}, "
            f"checkpoint {checkpoints.directory.name}/{landed.name}"
        )

        _report(
            epoch,
            loss=loss,
            learning_rate=rate,
            attempt=_score(model, held_out, schedule, reporting, device, stem=stem),
            held_out=held_out,
            reporting=reporting,
            journal=journal,
            say=say,
        )


def _report(
    epoch: int,
    *,
    loss: float | None,
    learning_rate: float | None,
    attempt: Attempt,
    held_out: Sequence[TileRef],
    reporting: Reporting,
    journal: Journal,
    say: Callable[[str], None],
) -> None:
    """Write down what this epoch was worth, and say it.

    `loss` is None for an epoch scored from a checkpoint rather than at the end of its own pass:
    the number was lost with the session that computed it, and a zero there would be a training
    loss nobody measured. `learning_rate` is None for the same reason and at the same time: the
    rate that epoch trained at was never read, and a value here would be a rate nobody trained at.
    """
    journal.record(
        {
            "epoch": epoch,
            "training_loss": loss,
            "learning_rate": learning_rate,
            "held_out_tiles": len(held_out),
            "held_out_ships": attempt.ships,
            "at": [
                {
                    "score": threshold,
                    "precision": counts.precision,
                    "recall": counts.recall,
                    "found": counts.true_positives,
                    "false": counts.false_positives,
                    "missed": counts.false_negatives,
                }
                for threshold, counts in attempt.sweep(reporting.thresholds)
            ],
        }
    )
    for threshold, counts in attempt.sweep(reporting.thresholds):
        say(f"  {counts.line(threshold)}")


def _one_epoch(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    training: Sequence[TileRef],
    epoch: int,
    schedule: Schedule,
    device: torch.device,
    *,
    stem: str = "repeat",
) -> float:
    """One pass over the training tiles. Returns the mean loss, which is the run's only sign of
    life between one held-out score and the next."""
    loader = DataLoader(
        _Tiles(training, epoch=epoch, seed=schedule.seed, stem=stem),
        batch_size=schedule.batch_size,
        shuffle=True,
        num_workers=schedule.workers,
        collate_fn=_as_batch,
        # Seeded on the epoch, so the tiles arrive in the same order in a resumed session as they
        # would have in the session that was interrupted.
        generator=torch.Generator().manual_seed(schedule.seed * 1000 + epoch),
    )

    # Faster R-CNN samples which anchors and which proposals it learns from, out of torch's
    # global generator. Seeded per epoch rather than once per session, and derived rather than
    # carried, so that a session resumed at epoch 7 draws what an uninterrupted run would have
    # drawn there — the same trick the augmentation uses, and for the same reason.
    torch.manual_seed(schedule.seed * 1000 + epoch)

    model.train()
    total = 0.0
    for images, targets in tqdm(loader, desc=f"epoch {epoch}", leave=False):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        losses = model(images, targets)
        loss = sum(losses.values())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total += float(loss.detach())

    return total / max(len(loader), 1)


def _score(
    model: torch.nn.Module,
    held_out: Sequence[TileRef],
    schedule: Schedule,
    reporting: Reporting,
    device: torch.device,
    *,
    stem: str = "repeat",
) -> Attempt:
    """Run the held-out split and count what came back.

    The split is scored entire, empty tiles included. They are where a false positive happens,
    and a precision measured only over tiles known to contain a ship is a precision the detector
    has not earned.
    """
    loader = DataLoader(
        _Tiles(held_out, stem=stem),
        batch_size=schedule.batch_size,
        shuffle=False,
        num_workers=schedule.workers,
        collate_fn=_as_batch,
    )
    tolerance = reporting.tolerance()

    model.eval()
    attempt = NOTHING
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="held out", leave=False):
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets, strict=True):
                attempt = attempt + measure(detections_from(output), _ships_in(target), tolerance)

    return attempt


class _Tiles(Dataset):
    """The catalogued tiles, read one at a time, in the form torchvision's detectors take.

    `epoch` is what decides the augmentation, and passing None turns it off — which is what the
    held-out split gets. Scoring an augmented split would measure the model against eight views
    of a tile and report the average as if it were one.

    `stem` decides how many channels a tile is handed over in, and has to be the one the model
    was built with.
    """

    def __init__(
        self,
        refs: Sequence[TileRef],
        epoch: int | None = None,
        seed: int = 0,
        stem: str = "repeat",
    ) -> None:
        self.refs = refs
        self.epoch = epoch
        self.seed = seed
        self.stem = stem

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tile = self.refs[index].read()
        if self.epoch is not None:
            tile = symmetry_for(tile.name, self.epoch, self.seed)(tile)

        boxes = torch.tensor([box.to_xyxy() for box in tile.boxes], dtype=torch.float32).reshape(
            -1, 4
        )

        return as_model_input(tile.image, self.stem), {
            "boxes": boxes,
            "labels": torch.full((len(tile.boxes),), SHIP, dtype=torch.int64),
        }


def _as_batch(batch: list[tuple]) -> tuple[tuple, tuple]:
    """Detectors take a list of images of their own sizes, not one stacked tensor."""
    images, targets = zip(*batch, strict=True)
    return images, targets


def _ships_in(target: dict[str, torch.Tensor]) -> list[Box]:
    return [Box.from_xyxy(box) for box in target["boxes"].tolist()]
