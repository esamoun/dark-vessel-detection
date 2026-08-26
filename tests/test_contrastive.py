"""Fitting a representation to an archive, with no labels anywhere in it.

Nothing here asserts that the representation is any good. A model is evaluated, not asserted —
the rule `test_pipeline.py` states and `test_training_run.py` keeps. What is asserted is that the
machinery around it holds: that the loss is the loss it claims to be, that a session killed
halfway through continues rather than restarts, that what fitted an encoder travels with it, and
that a collapsed representation is visible in the number the run records rather than hidden by
the loss falling.

Skipped where torch is not installed, which includes CI: the chain's acceptance condition is that
it installs and runs without a framework, so the framework is an extra and the suite has to be
honest about running without it.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from test_archive import archive as archive_of  # noqa: E402

from darkvessel.detect.amplitude import DecibelStretch  # noqa: E402
from darkvessel.detect.checkpoints import Checkpoints, Journal  # noqa: E402
from darkvessel.embed.archive import Archive  # noqa: E402
from darkvessel.embed.contrastive import (  # noqa: E402
    ContrastiveEmbedder,
    Schedule,
    encoder,
    nt_xent,
    train,
)
from darkvessel.embed.views import Speckle  # noqa: E402

# Small enough that a run fits inside a test on a laptop CPU, and large enough to have negatives
# in every batch.
CROP_PX = 16
MARGIN_PX = 4
DIM = 8

STRETCH = DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-21.0)
SCHEDULE = Schedule(epochs=2, batch_size=4, learning_rate=1e-3, temperature=0.1, seed=11)


def archive(count: int = 12) -> Archive:
    """Crops of sea with a bright square of varying size in the middle of each."""
    rng = np.random.default_rng(0)
    side = CROP_PX + 2 * MARGIN_PX
    crops = rng.normal(loc=-21.0, scale=2.0, size=(count, side, side)).astype(np.float32)
    for index, crop in enumerate(crops):
        half = 1 + index % 4
        middle = side // 2
        crop[middle - half : middle + half, middle - half : middle + half] = 5.0

    provenance = archive_of(count=count, crop_px=CROP_PX, margin_px=MARGIN_PX).provenance
    return Archive(crops=crops, provenance=provenance, crop_px=CROP_PX, margin_px=MARGIN_PX)


def test_the_loss_is_lowest_when_the_pairs_are_the_pairs() -> None:
    """Two views that agree must cost less than two views that have been shuffled apart."""
    first = torch.nn.functional.normalize(torch.randn(8, DIM), dim=1)
    shuffled = first[torch.randperm(8)]

    assert float(nt_xent(first, first.clone(), 0.1)) < float(nt_xent(first, shuffled, 0.1))


def test_a_batch_whose_views_do_not_pair_up_is_refused() -> None:
    with pytest.raises(ValueError, match="pairs must line up"):
        nt_xent(torch.randn(4, DIM), torch.randn(3, DIM), 0.1)


def test_the_same_seed_builds_the_same_encoder_twice() -> None:
    """The failure of 2026-08-14, at this level: a run that does not name its initialisation is a
    run whose two executions are two different models with nothing in the config to say so."""
    first = encoder(dim=DIM, seed=5).represent.weight.detach().clone()
    again = encoder(dim=DIM, seed=5).represent.weight.detach()
    other = encoder(dim=DIM, seed=6).represent.weight.detach()

    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def fit(
    tmp_path: Path,
    schedule: Schedule = SCHEDULE,
    crops: int = 12,
    say: Callable[[str], None] = lambda line: None,
) -> Journal:
    journal = Journal(tmp_path / "metrics.json")
    train(
        archive=archive(crops),
        stretch=STRETCH,
        speckle=Speckle(looks=4.0),
        schedule=schedule,
        dim=DIM,
        tolerance_m=200.0,
        checkpoints=Checkpoints(tmp_path / "checkpoints", keep=2),
        journal=journal,
        device=torch.device("cpu"),
        say=say,
    )
    return journal


def test_a_run_records_a_twin_recall_and_the_chance_it_is_measured_against(
    tmp_path: Path,
) -> None:
    """The loss cannot fail visibly — a collapse drives it to a constant and holds there — so
    the number kept beside it is the one that can, and it is meaningless without its baseline."""
    entries = fit(tmp_path).entries()

    assert [entry["epoch"] for entry in entries] == [1, 2]
    for entry in entries:
        assert 0.0 <= entry["twin_recall"] <= 1.0
        # Each crop of this fixture stands a kilometre from every other, so nothing counts as
        # the same object and the chance level is the strict one.
        assert entry["chance"] == pytest.approx(1.0 / 12)


def test_a_session_killed_halfway_continues_rather_than_starting_again(tmp_path: Path) -> None:
    """A free-tier session ends when the provider says so, not when the schedule does.

    The same schedule both times, because a resume under an edited one is a different experiment
    and `Journal.describe` refuses it. What is killed here is the session, not the run.
    """
    schedule = Schedule(epochs=3, batch_size=4, learning_rate=1e-3, temperature=0.1, seed=11)

    def die_after_the_first(line: str) -> None:
        if line.startswith("epoch 1:"):
            raise KeyboardInterrupt("the session ended here")

    with pytest.raises(KeyboardInterrupt):
        fit(tmp_path, schedule, say=die_after_the_first)

    journal = fit(tmp_path, schedule)

    assert [entry["epoch"] for entry in journal.entries()] == [1, 2, 3]


def test_what_fitted_an_encoder_travels_with_it(tmp_path: Path) -> None:
    """A representation fitted under one decibel window and applied under another returns
    plausible vectors and answers a different question. The window is in the checkpoint."""
    fit(tmp_path)

    embedder = ContrastiveEmbedder(checkpoint=tmp_path / "checkpoints" / "epoch-002.pt")

    assert (embedder.crop_px, embedder.margin_px, embedder.dim) == (CROP_PX, MARGIN_PX, DIM)
    assert embedder.stretch == STRETCH


def test_an_encoder_that_does_not_say_what_built_it_is_refused(tmp_path: Path) -> None:
    torch.save({"model": encoder(dim=DIM, seed=1).state_dict()}, tmp_path / "anonymous.pt")

    with pytest.raises(ValueError, match="what built it"):
        ContrastiveEmbedder(checkpoint=tmp_path / "anonymous.pt")


def test_the_embedder_answers_one_vector_per_crop_and_nothing_for_none(tmp_path: Path) -> None:
    fit(tmp_path)
    embedder = ContrastiveEmbedder(checkpoint=tmp_path / "checkpoints" / "epoch-002.pt")

    vectors = embedder(archive(5).crops)
    none = embedder(np.empty((0, CROP_PX + 2 * MARGIN_PX, CROP_PX + 2 * MARGIN_PX)))

    assert vectors.shape == (5, DIM)
    # A scene that found nothing is an ordinary outcome, and its layer still carries the columns.
    assert none.shape == (0, DIM)


def test_an_archive_too_small_to_fill_a_batch_is_refused(tmp_path: Path) -> None:
    """The loss is a comparison against the rest of the batch, and a batch of one compares with
    nothing — a run that quietly padded it would report a loss measuring something else."""
    with pytest.raises(ValueError, match="cannot fill a batch"):
        fit(tmp_path, crops=3)


def test_loading_an_encoder_does_not_reseed_whatever_the_caller_was_drawing_from(
    tmp_path: Path,
) -> None:
    """Building the model to load weights into seeds the global generator, and every weight it
    initialises is then overwritten. Left unforked, opening a checkpoint would reset a caller's
    stream as a side effect of reading a file — which shows up somewhere else entirely."""
    fit(tmp_path)
    torch.manual_seed(1234)
    expected = torch.randn(3)

    torch.manual_seed(1234)
    ContrastiveEmbedder(checkpoint=tmp_path / "checkpoints" / "epoch-002.pt")

    assert torch.equal(torch.randn(3), expected)
