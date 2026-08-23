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


def _code_sources() -> list[str]:
    """The code cells only, in notebook order.

    Ordering and "how many cells do this" claims are about what executes, and the prose cells
    describe the code cells by name — the intro names `split_by_scene` while explaining why the
    check below it exists. Counting those as a second check reads a correct notebook as broken.
    """
    document = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code"]


def _training_cell() -> int:
    """The index, among the code cells, of the one that launches the run.

    Keyed on `subprocess` rather than on the words "darkvessel train", which the split check
    also contains — it imports `training_request_from` from `darkvessel.cli`. Keying on the
    launch mechanism has the side effect of being the thing under test: a cell that went back
    to a `!` shell line would not be found here at all, and every test below would say so.
    """
    cells = [i for i, source in enumerate(_code_sources()) if "subprocess" in source]
    assert len(cells) == 1, "expected exactly one cell that launches the training run"
    return cells[0]


def test_the_notebook_names_its_run_once_rather_than_in_two_places_that_can_disagree() -> None:
    """The two literals that used to name a run a second time are refused outright, and the
    training cell has to read the run back through `CONFIG` rather than naming it again itself.

    Banning the two old literals only closes half the trap: a notebook that kept the derived
    resume cell but put the training cell back to a hardcoded
    `--config /kaggle/working/repo/configs/train.yaml` still has `CONFIG` sitting in the source,
    defined by the resume cell and read by nothing, and neither banned literal appears — so the
    two checks below would both pass while the two-edit trap was back. The training cell's own
    source is what has to be checked: it has to reference `CONFIG` and name no config file of
    its own.
    """
    whole = "\n".join(_cell_sources())

    assert "metrics.json" not in whole
    assert "/checkpoints" not in whole

    training = _code_sources()[_training_cell()]
    assert "CONFIG" in training, "the training cell has to read the run back through CONFIG"
    assert ".yaml" not in training, "naming a config here is the second edit that can disagree"


def test_the_notebook_stops_on_an_empty_held_out_split_before_it_reaches_the_training_cell() -> (
    None
):
    """The split is read, and refused when it is empty, in a cell that runs *before* training.

    `split_by_scene` is a pure filter over whichever scenes the catalogue holds, so a
    `data.images` that no longer matches the dataset's layout does not raise — it yields an
    empty held-out split, and the run trains to completion and reports its one measured number
    over nothing. This has already happened once from outside the repository: the Kaggle mirror
    splits LS-SSDD's images into two directories and Kaggle moved its mount point, both
    unannounced (docs/decisions.md, 2026-08-20).

    The check therefore lives in the notebook rather than in the runbook's prose, where it was
    a snippet an operator had to paste, and it asserts rather than prints — a `0` printed by a
    Run All scrolls past and the training cell starts anyway. Ordering is half the guard: a
    check that runs after the training call has already cost the session it exists to save.
    """
    sources = _code_sources()

    checks = [i for i, source in enumerate(sources) if "split_by_scene" in source]
    assert len(checks) == 1, "expected exactly one cell that reads the split"
    assert checks[0] < _training_cell(), "the split check has to run before the GPU time is spent"

    assert "assert held_out" in sources[checks[0]], (
        "the check has to refuse an empty held-out split, not merely print its size"
    )


def test_the_run_is_launched_from_python_rather_than_through_the_notebook_s_shell() -> None:
    """The training cell calls the interpreter directly, and no shell line relies on braces.

    Two separate things went wrong on Kaggle, in that order, and this pins both.

    `pip install -e` puts the `darkvessel` console script wherever the installer chose, and
    whether that directory is on PATH belongs to the machine. On Kaggle it is not, so
    `!darkvessel train` answered `command not found` at the one cell with the dataset and the
    wheels behind it. `sys.executable` in an argument list names the interpreter that actually
    holds the package, and cannot be resolved against a different one.

    The repair for that was `!{sys.executable} -m darkvessel train`, and it failed too: `{...}`
    substitution inside a `!` line is a property of the frontend, not of Python, and this image
    passes the braces through to bash untouched — so the run died on the brace. Worse, it fails
    the same way whether or not the name exists, which is why `!{...}` is banned outright here
    rather than left to be noticed again.
    """
    sources = _code_sources()
    training = sources[_training_cell()]

    assert "sys.executable" in training, "name the interpreter that holds the package"
    assert '"train"' in training, "the run is the `train` subcommand, not another one"

    assert "!{" not in "\n".join(sources), "a `!` line's brace substitution is not Python's"
    assert "!darkvessel" not in "\n".join(sources), "the console script is not on Kaggle's PATH"


def test_the_clone_is_named_once_and_put_on_the_path_of_both_interpreters() -> None:
    """`SRC` is defined once, before the first import, and reaches the child through the env.

    Observed on Kaggle, 2026-08-23: the repository cloned, `pip install -e` ran against it, and
    `import darkvessel` still raised ModuleNotFoundError while the console script was not on
    PATH — the package on disk and invisible to the kernel. Where the install put it is a
    property of the machine; where the clone is, is not, and this is a src layout of pure Python
    that needs no install to be importable.

    Three things have to hold together, and each is a way the fix silently stops working. The
    path has to be set *before* anything imports `darkvessel`, or the cell is decoration. It has
    to be written once, or the kernel and the child can be pointed at different clones. And the
    child needs it through `PYTHONPATH`, because a subprocess inherits the parent's environment
    and not its `sys.path`.
    """
    sources = _code_sources()

    definitions = [i for i, source in enumerate(sources) if "SRC =" in source]
    assert len(definitions) == 1, "expected exactly one cell that names the clone's src"
    assert "sys.path.insert" in sources[definitions[0]], "naming it is not putting it on the path"

    imports = next(i for i, source in enumerate(sources) if "from darkvessel" in source)
    assert definitions[0] < imports, "the path has to be set before the first darkvessel import"

    # On the code, not on the prose: every cell here explains itself, and the words PYTHONPATH
    # and SRC both appear in this cell's comment. Asserting on those passes a cell whose `env=`
    # has been deleted -- which is how this check was first written, and it let that revert
    # through.
    training = sources[_training_cell()]
    assert '"PYTHONPATH": SRC' in training, (
        "the child needs the clone through the environment: a subprocess inherits the parent's "
        "environment, never its sys.path"
    )

    assert "\n".join(sources).count('"/kaggle/working/repo/src"') == 1, "named once, read twice"
