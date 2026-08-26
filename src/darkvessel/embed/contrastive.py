"""Contrastive training on detection crops, without labels.

Learns a representation in which visually similar objects sit close together, so that classes
never annotated - fixed structures, small craft, large hulls - become separable after the fact.

There are no labels here at all, and that is the whole design rather than a shortcut. What
supervises the encoder is the statement that two views of one crop are the same object and two
views of different crops are not: the loss below pulls the first pair together and pushes every
other pair apart. Which transformations may stand between the two views is therefore the entire
specification of what the representation is asked to ignore, and it lives in `views.py` where it
can be read without the framework.

The encoder is small and built here rather than borrowed. A ResNet on a 64 px crop of
single-polarisation amplitude would be three hundred megabytes of ImageNet priors about colour
photographs applied to eighty objects, and the detector already carries that argument as far as
it goes. Four strided convolutions and an average pool is enough to have a representation at all,
which is what this level is asked for, and it is small enough to fit in the minutes of laptop CPU
that the archive it is fitted on actually deserves.

The stretch that turns decibels into the 0..1 the encoder sees is written into the checkpoint, not
restated in a config. A representation fitted under one window and applied under another is the
same silent fault as a checkpoint loaded with the wrong anchors — nothing raises, the vectors
come back, the neighbours are plausible — so the window travels with the weights and the embedder
reads it from there.

torch is imported at module level, which is safe for the reason `trained.py` states: nothing
imports this module unless a config asks for it. `darkvessel run` without an embedding stage needs
no framework, which is the chain's acceptance condition.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from darkvessel.detect.amplitude import DecibelStretch
from darkvessel.detect.checkpoints import Checkpoints, Journal
from darkvessel.embed.archive import Archive
from darkvessel.embed.crops import centre
from darkvessel.embed.retrieval import chance_of, twin_recall
from darkvessel.embed.views import Speckle, rng_for, view

# How many channels a crop has. One, and there is no second: the scene this chain exports is VV,
# for the reason docs/failures.md records about the dual-polarisation stem.
CHANNELS = 1

# The widths of the four strided blocks. Doubling until the last, which holds — the last block
# sees a 4 px feature map on a 64 px crop, and there is nothing left there to widen for.
_WIDTHS = (32, 64, 128, 128)

# Which view index the twin check draws. Training takes 0 and 1, so 2 is a view of the same crop
# that this epoch's encoder has not been shown — which is what makes the number a check rather
# than a restatement of the loss.
_TWIN_VIEW = 2


@dataclass(frozen=True)
class Schedule:
    """How long the run is, and how hard it pulls."""

    epochs: int
    batch_size: int
    learning_rate: float
    # How sharply the loss distinguishes the positive pair from everything else. Low values make
    # the hardest negatives dominate the gradient, which is what a small archive needs and what
    # a large one can do without.
    temperature: float
    seed: int

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"a temperature of {self.temperature} divides the scores by nothing")
        if self.batch_size < 2:
            raise ValueError(
                f"a batch of {self.batch_size} has no negatives in it; the loss is a comparison "
                "against the rest of the batch, and a batch of one compares with nothing"
            )


class Encoder(nn.Module):
    """Crops in, one vector each out, plus the head the loss is computed on.

    Two outputs and only one of them is kept. `forward` returns the representation, which is what
    the archive stores and what retrieval ranks; `project` is the small head the contrastive loss
    is actually applied to and it is discarded after training. The separation is the one finding
    from the contrastive literature this project takes on trust rather than re-deriving: the
    representation immediately before the head carries more than the head's own output does.
    """

    def __init__(self, *, dim: int, width: tuple[int, ...] = _WIDTHS) -> None:
        super().__init__()
        channels = (CHANNELS, *width)
        self.features = nn.Sequential(
            *(
                layer
                for inputs, outputs in zip(channels[:-1], channels[1:], strict=True)
                for layer in (
                    nn.Conv2d(inputs, outputs, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(outputs),
                    nn.ReLU(inplace=True),
                )
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.represent = nn.Linear(width[-1], dim)
        self.head = nn.Sequential(
            nn.Linear(dim, width[-1]), nn.ReLU(inplace=True), nn.Linear(width[-1], dim)
        )

    def forward(self, crops: Tensor) -> Tensor:
        return self.represent(self.features(crops))

    def project(self, representation: Tensor) -> Tensor:
        return self.head(representation)


def encoder(*, dim: int, seed: int) -> Encoder:
    """A fresh encoder, named by its seed.

    Seeded here rather than by the caller, for the reason `detector_model` is: every weight in
    this model is initialised from the global generator, and a run that does not name that number
    is a run whose two executions are two different models with nothing in the config to say so.
    See docs/failures.md, 2026-08-14.
    """
    torch.manual_seed(seed)
    return Encoder(dim=dim)


def nt_xent(first: Tensor, second: Tensor, temperature: float) -> Tensor:
    """The normalised temperature-scaled cross entropy of two views of one batch.

    Each row of `first` is asked to pick its partner in `second` out of all 2n - 1 other views in
    the batch, and vice versa. Cosine similarity, so the vectors are normalised before they are
    compared — which is why retrieval ranks by cosine too: ranking by anything else would rank by
    a quantity nothing here ever optimised.
    """
    if len(first) != len(second):
        raise ValueError(f"{len(first)} views against {len(second)}; the pairs must line up")

    count = len(first)
    views = torch.nn.functional.normalize(torch.cat([first, second]), dim=1)
    similarity = views @ views.T / temperature

    # A view is not its own negative. Masked with -inf rather than with a large negative number,
    # because the softmax below is exact at -inf and merely nearly right at -1e9.
    similarity.fill_diagonal_(float("-inf"))

    # The partner of view i is view i + n, and of view i + n is view i.
    partners = torch.cat([torch.arange(count, 2 * count), torch.arange(count)]).to(
        similarity.device
    )
    return torch.nn.functional.cross_entropy(similarity, partners)


def train(
    *,
    archive: Archive,
    stretch: DecibelStretch,
    speckle: Speckle | None,
    schedule: Schedule,
    dim: int,
    tolerance_m: float,
    checkpoints: Checkpoints,
    journal: Journal,
    device: torch.device,
    say: Callable[[str], None] = print,
) -> None:
    """Fit a representation to the archive, and record what it is worth every epoch.

    Picks up wherever the last session stopped, and reproduces what an uninterrupted run would
    have done: the views of a crop are derived from the seed, the epoch and the crop's position
    in the archive rather than drawn from a stream, so a resumed epoch augments exactly as the
    interrupted one was going to. The convention `dataset.symmetry_for` states, applied here.

    The twin recall is recorded beside the loss because the loss alone cannot fail visibly. A
    representation that collapses onto a point drives the loss down to log(2n - 1) and stays
    there looking like a run that has converged; the recall drops to chance and says so.

    `tolerance_m` is the distance at which two detections are one object, and it is the fusion's
    own — see `Archive.co_located`. It belongs to the check rather than to the training: nothing
    below optimises it, and what it changes is whether the second cut of a hull counts as a wrong
    answer when the first was asked for.
    """
    if len(archive) < schedule.batch_size:
        raise ValueError(
            f"an archive of {len(archive)} crops cannot fill a batch of {schedule.batch_size}; "
            "the loss compares a pair against the rest of the batch and there is not one"
        )

    model = encoder(dim=dim, seed=schedule.seed).to(device)
    built = {
        "crop_px": archive.crop_px,
        "margin_px": archive.margin_px,
        "dim": dim,
        "seed": schedule.seed,
        # Travels with the weights rather than being restated by whoever loads them.
        "stretch": asdict(stretch),
    }
    same_as = archive.co_located(tolerance_m)
    chance = chance_of(same_as)
    journal.describe(
        {
            "built": built,
            "schedule": asdict(schedule),
            "speckle": None if speckle is None else asdict(speckle),
            "crops": len(archive),
            "scenes": archive.scenes(),
            "same_object_tolerance_m": tolerance_m,
        }
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=schedule.learning_rate)

    resumed = checkpoints.latest()
    if resumed is not None:
        epoch, path = resumed
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        say(f"resuming after epoch {epoch}, from {path.name}")

    first = checkpoints.next_epoch()
    if first > schedule.epochs:
        say(f"the schedule of {schedule.epochs} epochs is already finished; nothing to do")
        return

    say(
        f"epochs {first} to {schedule.epochs} on {device}, {len(archive)} crops from "
        f"{len(archive.scenes())} scene(s), batch {schedule.batch_size}"
    )

    for epoch in range(first, schedule.epochs + 1):
        loss = _one_epoch(model, optimiser, archive, epoch, schedule, stretch, speckle, device)

        # Before the check, not after: an interrupted check costs a number that can be recomputed
        # from the weights, and the weights are the expensive thing to lose. The ordering
        # `train.py` argues for, on a run short enough that it barely matters and consistent
        # anyway — a second convention is a second thing to get wrong.
        with checkpoints.writing(epoch) as partial:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "built": built,
                },
                partial,
            )

        recall = _twin_recall(
            model, archive, epoch, schedule, stretch, speckle, device, same_as=same_as
        )
        journal.record({"epoch": epoch, "loss": loss, "twin_recall": recall, "chance": chance})
        say(
            f"epoch {epoch}: loss {loss:.4f}, twin recall {recall:.3f} against {chance:.3f} at "
            f"chance, checkpoint {checkpoints.directory.name}/{checkpoints.path_for(epoch).name}"
        )


def _one_epoch(
    model: Encoder,
    optimiser: torch.optim.Optimizer,
    archive: Archive,
    epoch: int,
    schedule: Schedule,
    stretch: DecibelStretch,
    speckle: Speckle | None,
    device: torch.device,
) -> float:
    """One pass over the archive in batches, returning the mean loss across them.

    The order the crops arrive in is drawn from the epoch rather than from a generator, for the
    same reason their views are: a resumed session has to see what an uninterrupted one would.
    A batch short of the full size is dropped rather than run — a final batch of one has no
    negatives in it, and the loss over it is a number that means something else.
    """
    model.train()
    order = rng_for(schedule.seed, epoch, "order").permutation(len(archive))
    losses = []

    for start in range(0, len(order) - schedule.batch_size + 1, schedule.batch_size):
        batch = order[start : start + schedule.batch_size]
        views = [
            _as_input(
                _views_of(archive, batch, epoch, schedule.seed, index, stretch, speckle), device
            )
            for index in (0, 1)
        ]
        projected = [model.project(model(view)) for view in views]

        loss = nt_xent(projected[0], projected[1], schedule.temperature)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    return float(np.mean(losses))


def _views_of(
    archive: Archive,
    batch: np.ndarray,
    epoch: int,
    seed: int,
    index: int,
    stretch: DecibelStretch,
    speckle: Speckle | None,
) -> np.ndarray:
    """One view of each crop in `batch`, stretched into the amplitude the encoder was fitted on.

    Augmented in decibels and stretched afterwards, in that order. A speckle model is a statement
    about a backscatter coefficient, and applying it to a number that has already been squashed
    into 0..1 and clipped at both ends would be applying it to something else.
    """
    return np.stack(
        [
            stretch(
                view(
                    archive.crops[crop],
                    crop_px=archive.crop_px,
                    speckle=speckle,
                    rng=rng_for(seed, epoch, int(crop), index),
                )
            )
            for crop in batch
        ]
    )


def _twin_recall(
    model: Encoder,
    archive: Archive,
    epoch: int,
    schedule: Schedule,
    stretch: DecibelStretch,
    speckle: Speckle | None,
    device: torch.device,
    same_as: np.ndarray,
) -> float:
    """How often a view this epoch never trained on lands nearest its own object."""
    unaugmented = stretch(centre(archive.crops, archive.crop_px))
    twins = _views_of(
        archive, np.arange(len(archive)), epoch, schedule.seed, _TWIN_VIEW, stretch, speckle
    )

    return twin_recall(
        _embed(model, unaugmented, device), _embed(model, twins, device), same_as=same_as
    )


def _embed(model: Encoder, stretched: np.ndarray, device: torch.device) -> np.ndarray:
    """Vectors for a stack of crops already in 0..1, without touching the training state."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        vectors = model(_as_input(stretched, device)).cpu().numpy()
    model.train(was_training)
    return vectors


def _as_input(stretched: np.ndarray, device: torch.device) -> Tensor:
    """A stack of crops in 0..1 as the single-channel batch the encoder takes."""
    return torch.from_numpy(np.ascontiguousarray(stretched)).unsqueeze(1).to(device)


class ContrastiveEmbedder:
    """A fitted encoder, behind the contract the pipeline's optional stage takes.

    Everything about how it was fitted comes off the checkpoint: the crop geometry it expects, the
    dimension it answers in, and the window between decibels and amplitude it was fitted under.
    Nothing is restated in a config, so there is nothing for a config to disagree with.
    """

    def __init__(self, *, checkpoint: Path, device: torch.device | None = None) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        built = _built_of(state, checkpoint)

        # Inside a forked generator state, because `encoder` seeds the global one and every
        # weight it initialises is about to be overwritten by the load below. Without the fork,
        # opening a checkpoint would silently reset whatever stream its caller was drawing from —
        # a side effect of reading a file, and the kind that shows up somewhere else entirely.
        with torch.random.fork_rng(devices=[]):
            model = encoder(dim=int(built["dim"]), seed=0)
        model.load_state_dict(state["model"])

        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.crop_px = int(built["crop_px"])
        self.margin_px = int(built["margin_px"])
        self.dim = int(built["dim"])
        self.stretch = DecibelStretch(**built["stretch"])

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """One vector per crop, from the middle window of each — the view with the object in it.

        An empty stack comes back as an empty array of the right width rather than as a shape
        nothing can attach: a scene that found nothing is an ordinary outcome, and the layer it
        writes still has to carry the same columns as the layer beside it.
        """
        if len(crops) == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        return _embed(self.model, self.stretch(centre(crops, self.crop_px)), self.device)


def _built_of(state: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """What built the encoder in this checkpoint, refusing one that does not say.

    Silence is not allowed here, unlike in `trained.py` where the first checkpoint predates the
    block being written. This level has no such history: every encoder ever written by this
    project records its geometry, and one that does not is not an encoder this project wrote.
    """
    built = state.get("built")
    if built is None:
        raise ValueError(
            f"{checkpoint.name} does not record what built it, so the crop size, the dimension "
            "and the decibel window it was fitted under would all have to be guessed"
        )
    return built
