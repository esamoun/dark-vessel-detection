"""The Kaggle training notebook, checked as data.

It needs a GPU, a Kaggle session and an attached dataset to actually run, none of which are here
— but the defect this file guards against was never in what the notebook does when it runs. It
was in what a human had to remember to edit before it did: `notebooks/kaggle-train.ipynb` used to
name the run being trained in two places that had no way of noticing they disagreed. Cell 4
hardcoded `configs/train.yaml`, so switching to a rung of the ladder meant training the wrong
config unless a second edit, to cell 3, kept it in step. That second edit was easy to miss and
its failure was quiet: cell 3 hardcoded the working directory as `/kaggle/working/checkpoints`
and the glob as `*/checkpoints/epoch-*.pt`, both literally "checkpoints" rather than whichever
rung was actually running, so a mismatched cell 3 found nothing to resume from, printed a
reassuring first-session message, and an evening of paid GPU time trained from scratch.

`CONFIG` is now the one place a run is named — both cells read it, and there is nothing left to
edit twice. This file is what stops that property drifting back out: it needs no torch, no GPU
and no network, so it runs in CI on every push, which the notebook itself never will.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-train.ipynb"


def _cell_sources() -> list[str]:
    """Every cell's source, joined into one string per cell, in notebook order."""
    document = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in document["cells"]]


def test_the_notebook_names_its_run_once_rather_than_in_two_places_that_can_disagree() -> None:
    """The two literals that used to name a run a second time are refused outright, and the
    training cell has to read the run back through `CONFIG` rather than naming it again itself.

    Banning the two old literals only closes half the trap: a notebook that kept the derived
    resume cell but put the training cell back to a hardcoded
    `!darkvessel train --config /kaggle/working/repo/configs/train.yaml` still has `CONFIG`
    sitting in the source, defined by the resume cell and read by nothing, and neither banned
    literal appears — so the two checks below would both pass while the two-edit trap was back.
    The training cell's own source is what has to be checked, and it has to reference `{CONFIG}`
    rather than a path of its own.
    """
    sources = _cell_sources()
    whole = "\n".join(sources)

    assert "metrics.json" not in whole
    assert "/checkpoints" not in whole

    training = [source for source in sources if "darkvessel train" in source]
    assert len(training) == 1, "expected exactly one cell that runs `darkvessel train`"
    assert "{CONFIG}" in training[0]
