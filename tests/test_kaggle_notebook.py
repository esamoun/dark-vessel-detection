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
    """The two literals that used to name a run a second time are refused outright, and `CONFIG`
    — the one place left that names it — has to be there instead. A notebook that reintroduced
    either literal would be reintroducing exactly the two-edit trap this file exists to close.
    """
    whole = "\n".join(_cell_sources())

    assert "CONFIG" in whole
    assert "metrics.json" not in whole
    assert "/checkpoints" not in whole
