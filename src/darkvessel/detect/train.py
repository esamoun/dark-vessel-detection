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
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from darkvessel.detect.checkpoints import Checkpoints, Journal
from darkvessel.detect.dataset import Box, TileRef, symmetry_for
from darkvessel.detect.detector import PixelDetection
from darkvessel.detect.metrics import NOTHING, Attempt, Reporting, measure
from darkvessel.detect.model import SHIP, as_model_input


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
    say: Callable[[str], None] = print,
) -> None:
    """Run the schedule, or as much of it as this session gets through.

    Picks up wherever the last session stopped. A run started fresh and a run resumed four times
    do the same epochs over the same tiles in the same order, because everything that would
    otherwise vary — which empty tiles the subset kept, which way each tile is laid down, the
    order they arrive in — is derived from the seed and the epoch number rather than from a
    generator's position in a stream.
    """
    model.to(device)
    optimiser = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=schedule.learning_rate,
        momentum=schedule.momentum,
        weight_decay=schedule.weight_decay,
    )

    resumed = checkpoints.latest()
    if resumed is not None:
        epoch, path = resumed
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        say(f"resuming after epoch {epoch}, from {path.name}")

        # The epoch whose weights survived the session that wrote them but whose score did not.
        # Scored now, from the checkpoint just loaded, because nothing else ever would: past the
        # end of the schedule the loop below does not run at all.
        if epoch not in {entry["epoch"] for entry in journal.entries()}:
            say(f"epoch {epoch} was never scored; scoring it from its checkpoint")
            _report(
                epoch,
                loss=None,
                attempt=_score(model, held_out, schedule, reporting, device),
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
        loss = _one_epoch(model, optimiser, training, epoch, schedule, device)

        # Before the scoring, not after: an interrupted evaluation costs the numbers, and the
        # numbers can be recomputed from the weights.
        with checkpoints.writing(epoch) as path:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                },
                path,
            )
        say(f"epoch {epoch}: loss {loss:.4f}, checkpoint {checkpoints.directory.name}/{path.name}")

        _report(
            epoch,
            loss=loss,
            attempt=_score(model, held_out, schedule, reporting, device),
            held_out=held_out,
            reporting=reporting,
            journal=journal,
            say=say,
        )


def _report(
    epoch: int,
    *,
    loss: float | None,
    attempt: Attempt,
    held_out: Sequence[TileRef],
    reporting: Reporting,
    journal: Journal,
    say: Callable[[str], None],
) -> None:
    """Write down what this epoch was worth, and say it.

    `loss` is None for an epoch scored from a checkpoint rather than at the end of its own pass:
    the number was lost with the session that computed it, and a zero there would be a training
    loss nobody measured.
    """
    journal.record(
        {
            "epoch": epoch,
            "training_loss": loss,
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
) -> float:
    """One pass over the training tiles. Returns the mean loss, which is the run's only sign of
    life between one held-out score and the next."""
    loader = DataLoader(
        _Tiles(training, epoch=epoch, seed=schedule.seed),
        batch_size=schedule.batch_size,
        shuffle=True,
        num_workers=schedule.workers,
        collate_fn=_as_batch,
        # Seeded on the epoch, so the tiles arrive in the same order in a resumed session as they
        # would have in the session that was interrupted.
        generator=torch.Generator().manual_seed(schedule.seed * 1000 + epoch),
    )

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
) -> Attempt:
    """Run the held-out split and count what came back.

    The split is scored entire, empty tiles included. They are where a false positive happens,
    and a precision measured only over tiles known to contain a ship is a precision the detector
    has not earned.
    """
    loader = DataLoader(
        _Tiles(held_out),
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
                attempt = attempt + measure(_detections_from(output), _ships_in(target), tolerance)

    return attempt


class _Tiles(Dataset):
    """The catalogued tiles, read one at a time, in the form torchvision's detectors take.

    `epoch` is what decides the augmentation, and passing None turns it off — which is what the
    held-out split gets. Scoring an augmented split would measure the model against eight views
    of a tile and report the average as if it were one.
    """

    def __init__(self, refs: Sequence[TileRef], epoch: int | None = None, seed: int = 0) -> None:
        self.refs = refs
        self.epoch = epoch
        self.seed = seed

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tile = self.refs[index].read()
        if self.epoch is not None:
            tile = symmetry_for(tile.name, self.epoch, self.seed)(tile)

        boxes = torch.tensor([box.to_xyxy() for box in tile.boxes], dtype=torch.float32).reshape(
            -1, 4
        )

        return as_model_input(tile.image), {
            "boxes": boxes,
            "labels": torch.full((len(tile.boxes),), SHIP, dtype=torch.int64),
        }


def _as_batch(batch: list[tuple]) -> tuple[tuple, tuple]:
    """Detectors take a list of images of their own sizes, not one stacked tensor."""
    images, targets = zip(*batch, strict=True)
    return images, targets


def _detections_from(output: dict[str, torch.Tensor]) -> list[PixelDetection]:
    """A model's boxes, as the points the rest of the chain deals in.

    Through `Box.from_xyxy` and `Box.centre` rather than by unpacking the corners here, so that
    the axis swap and the half-pixel between an edge coordinate and a pixel index are each
    applied in the one place that owns them.
    """
    return [
        PixelDetection(row=row, col=col, score=float(score))
        for box, score in zip(
            output["boxes"].cpu().tolist(), output["scores"].cpu().tolist(), strict=True
        )
        for row, col in [Box.from_xyxy(box).centre()]
    ]


def _ships_in(target: dict[str, torch.Tensor]) -> list[Box]:
    return [Box.from_xyxy(box) for box in target["boxes"].tolist()]
