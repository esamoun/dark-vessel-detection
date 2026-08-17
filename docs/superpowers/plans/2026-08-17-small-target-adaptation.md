# Small-Target Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the machinery that lets three detector adaptations be measured one at a time against the configuration before them — a config inheritance mechanism, a run that names itself, a comparison that refuses to compare unlike things, a learning-rate schedule that survives a resume, and a single-channel stem that is provably identical at initialisation.

**Architecture:** Everything that can be got wrong quietly stays on the torch-free side of the seam and is tested in CI: config resolution (`config.py`), the run's identity in its journal (`checkpoints.py`), and the ladder's comparison and keep/drop rule (`ladder.py`). Only the stem and the scheduler need the framework, and both are tested on the CPU against the real model builder and the real loop. The five GPU runs themselves are not agent work — they are a human in a Kaggle session, and Task 10 is their runbook.

**Tech Stack:** Python 3.11+, numpy, PyYAML, pytest, ruff; torch/torchvision behind the `detector` extra.

## Global Constraints

- `requires-python = ">=3.11"`.
- ruff: `line-length = 100`, `select = ["E", "F", "I", "UP", "B"]`. Run `make lint` before every commit.
- **The chain must install and run with no torch, no GPU and no network.** `darkvessel run` with `detector: bright-pixel` must never import torch. Any torch import lives inside the function that needs it, as `cli._train` already does.
- Tests that need torch use `torch = pytest.importorskip("torch", reason="the detector extra is not installed: pip install -e '.[detector]'")` at module top, with `# noqa: E402` on the imports below it. This is the existing idiom in `tests/test_training_run.py:23`.
- No `.pt` file is ever committed. `*.pt` is gitignored globally.
- **Metrics are reported, never asserted.** No test may pin a precision, a recall, an F1 or a detection count produced by a real model. `ladder.py` is tested against synthetic journals.
- British spelling in prose and identifiers, matching the existing code (`polarisations`, `optimiser`, `normalisation`, `georeferenced`).
- Commit after every task. Run `make lint && make test` before each commit.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/darkvessel/config.py` | **new.** Reading a config file and the file it extends. No torch, no domain knowledge. |
| `src/darkvessel/detect/checkpoints.py` | `Journal` gains a run header, and refuses a resume that changed configuration. |
| `src/darkvessel/detect/ladder.py` | **new.** The rungs, the comparison, the refusals, the keep/drop rule, the table. No torch. |
| `src/darkvessel/detect/model.py` | `stem`, the folded single-channel `conv1`, the four sampler knobs. |
| `src/darkvessel/detect/train.py` | The scheduler, its state in the checkpoint, the learning rate in the journal. |
| `src/darkvessel/detect/trained.py` | `stem` through to the model, and into the build-block refusal. |
| `src/darkvessel/cli.py` | `load_config` at every call site, a `compare` subcommand, the new config keys. |
| `configs/train.yaml` | `model.stem`, four sampler keys, `schedule.lr_schedule`. |
| `configs/ladder/r1-cosine.yaml` … `r4-sampler.yaml` | **new.** One rung each, over `extends:`. |
| `configs/ladder.yaml` | **new.** The rungs, their metrics files, and the line each one changed. |
| `notebooks/anchor_census.py` | **new.** The census that decides rungs 2 and 4. |
| `docs/runs/` | **new.** Where each rung's `metrics.json` lands, committed. |
| `tests/test_config.py` | **new.** `extends` resolution and its refusals. Runs in CI. |
| `tests/test_ladder.py` | **new.** The comparison, the refusals, the rule at its boundary. Runs in CI. |
| `tests/test_model_stem.py` | **new.** The stem's equality at initialisation. Skipped in CI. |
| `tests/test_checkpoints.py` | The journal's new shape, and the legacy bare list. |
| `tests/test_training_run.py` | The scheduler survives a resume. Skipped in CI. |
| `tests/test_pipeline.py` | Every shipped config, rungs included, still parses. |

## Task Dependency

Tasks 1–9 are laptop work and run in order; each depends on the one before it only where the "Consumes" block says so. **Task 10 is run by a human in Kaggle sessions and produces the five `metrics.json` files.** Nothing in Tasks 1–9 waits on it.

---

### Task 1: A config can extend another config

**Files:**
- Create: `src/darkvessel/config.py`
- Create: `tests/test_config.py`
- Modify: `src/darkvessel/cli.py:94`, `:238`, `:297`, `:386`, `:436` — every `yaml.safe_load(config_path.read_text())`

**Interfaces:**
- Produces: `load_config(path: Path) -> dict[str, Any]`. Resolves `extends:` chains, deep-merges dicts, and removes the `extends` key from what it returns.

**Why this exists.** Each rung of the ladder differs from the one before it by a single line. Five standalone ninety-line configs would bury that line and would let a second one drift unnoticed, which breaks the ladder in exactly the way the ladder exists to prevent.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
"""Reading a config, and the config it stands on.

The ladder in issue #11 is five runs that differ by one line each. A rung that quietly differs by
two lines measures two things and reports one, so the mechanism that lets a rung state only its
own change is load-bearing rather than a convenience — and its failure modes are a cycle, a base
that is not there, and a merge that drops a key.
"""

from pathlib import Path

import pytest

from darkvessel.config import load_config


def test_a_config_that_extends_nothing_is_read_as_it_stands(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("schedule:\n  epochs: 12\n  learning_rate: 0.005\n")

    assert load_config(path) == {"schedule": {"epochs": 12, "learning_rate": 0.005}}


def test_a_rung_states_only_what_it_changes(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "schedule:\n  epochs: 12\n  learning_rate: 0.005\nmodel:\n  stem: repeat\n"
    )
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nschedule:\n  lr_schedule: cosine\n")

    assert load_config(rung) == {
        "schedule": {"epochs": 12, "learning_rate": 0.005, "lr_schedule": "cosine"},
        "model": {"stem": "repeat"},
    }


def test_a_rung_overrides_a_value_the_base_already_set(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  anchor_sizes: [[32], [64]]\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nmodel:\n  anchor_sizes: [[4], [8]]\n")

    assert load_config(rung)["model"]["anchor_sizes"] == [[4], [8]]


def test_a_list_is_replaced_and_never_merged(tmp_path: Path) -> None:
    """A merged list would make `anchor_sizes: [[4], [8]]` mean nine levels rather than two, and
    the run would train on something no file states."""
    (tmp_path / "base.yaml").write_text("reporting:\n  thresholds: [0.05, 0.5, 0.9]\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nreporting:\n  thresholds: [0.75]\n")

    assert load_config(rung)["reporting"]["thresholds"] == [0.75]


def test_a_chain_of_rungs_resolves_through_to_the_base(tmp_path: Path) -> None:
    """The ladder is greedy: rung 2 extends rung 1, which extends the baseline."""
    (tmp_path / "base.yaml").write_text("schedule:\n  epochs: 12\n")
    (tmp_path / "r1.yaml").write_text("extends: base.yaml\nschedule:\n  lr_schedule: cosine\n")
    r2 = tmp_path / "r2.yaml"
    r2.write_text("extends: r1.yaml\nmodel:\n  anchor_sizes: [[4]]\n")

    assert load_config(r2) == {
        "schedule": {"epochs": 12, "lr_schedule": "cosine"},
        "model": {"anchor_sizes": [[4]]},
    }


def test_the_base_is_found_beside_the_file_that_names_it(tmp_path: Path) -> None:
    """Every path in this project is read relative to the config that declares it, and the base a
    rung extends is a path like any other. Rung configs live one directory down from the base."""
    (tmp_path / "train.yaml").write_text("schedule:\n  epochs: 12\n")
    (tmp_path / "ladder").mkdir()
    rung = tmp_path / "ladder" / "r1.yaml"
    rung.write_text("extends: ../train.yaml\nschedule:\n  lr_schedule: cosine\n")

    assert load_config(rung)["schedule"] == {"epochs": 12, "lr_schedule": "cosine"}


def test_a_config_that_extends_itself_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text("extends: loop.yaml\nschedule:\n  epochs: 12\n")

    with pytest.raises(ValueError, match="extends itself"):
        load_config(path)


def test_a_cycle_between_two_configs_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")

    with pytest.raises(ValueError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


def test_a_base_that_is_not_there_is_refused_by_name(tmp_path: Path) -> None:
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: ../nowhere/train.yaml\n")

    with pytest.raises(FileNotFoundError, match="train.yaml"):
        load_config(rung)


def test_the_extends_key_does_not_survive_into_the_config(tmp_path: Path) -> None:
    """Whatever reads the result should not have to know the file was assembled from two."""
    (tmp_path / "base.yaml").write_text("schedule:\n  epochs: 12\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\n")

    assert "extends" not in load_config(rung)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'darkvessel.config'`

- [ ] **Step 3: Write the implementation**

Create `src/darkvessel/config.py`:

```python
"""Reading a config file, and the config file it stands on.

Every stage of this project is defined by one YAML file, and that is deliberate: a run is
reproducible from a file rather than from a sequence of cells executed in the right order. The
ladder in issue #11 puts a strain on it. Five training runs differ from one another by a single
line each, and five standalone copies of a ninety-line file would hide that line in a diff and
would let a second line drift without anyone noticing — which would break the one property the
ladder depends on, that each rung changes exactly one thing.

So a config may name the config it extends, and state only its own difference. Dicts are merged
key by key; anything else replaces. Lists in particular replace rather than concatenate, because
`anchor_sizes` merged with its base would silently mean nine pyramid levels rather than five, and
the run would then be training on something no file states.

Nothing here knows what a config *means*. Which keys exist and what they have to contain is the
business of the `*_request_from` functions in `cli.py`, and keeping that out of here is what lets
this module be read in one sitting.
"""

from pathlib import Path
from typing import Any

import yaml

EXTENDS = "extends"


def load_config(path: Path) -> dict[str, Any]:
    """One config, with everything it inherits already folded in.

    The `extends` key does not survive: whatever reads the result should not have to know that
    the file was assembled from two.
    """
    return _load(path.resolve(), ())


def _load(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(step.name for step in (*seen, path))
        raise ValueError(f"a config extends itself, directly or through a chain: {chain}")

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist, and a config extends it")

    config = yaml.safe_load(path.read_text()) or {}
    base = config.pop(EXTENDS, None)
    if base is None:
        return config

    # Relative to the file that names it, which is the rule this project applies to every other
    # path in every other config.
    return _merge(_load((path.parent / base).resolve(), (*seen, path)), config)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` wins, key by key, and two dicts at the same key are merged rather than replaced."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Use it at every call site in the CLI**

In `src/darkvessel/cli.py`, add the import beside the others:

```python
from darkvessel.config import load_config
```

Then replace every occurrence of `config = yaml.safe_load(config_path.read_text())` with:

```python
    config = load_config(config_path)
```

There are five, in `_run`, `_ais`, `_survey`, `_train` and `_export`. Leave the `import yaml` at the top: `_export` and the tests still use it elsewhere. If `ruff` reports `yaml` as unused after the edit, remove the import.

- [ ] **Step 6: Run the whole suite**

Run: `make lint && make test`
Expected: PASS. Every existing config has no `extends` key, so `load_config` returns exactly what `yaml.safe_load` did.

- [ ] **Step 7: Commit**

```bash
git add src/darkvessel/config.py src/darkvessel/cli.py tests/test_config.py
git commit -m "feat: a config may state only its difference from the one it extends"
```

---

### Task 2: A run names itself in its own journal

**Files:**
- Modify: `src/darkvessel/detect/checkpoints.py:109-129` — the `Journal` class
- Modify: `tests/test_checkpoints.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `Journal.describe(run: dict[str, Any]) -> None`, `Journal.run() -> dict[str, Any] | None`. `Journal.entries()` keeps its signature and returns the per-epoch list.

**Why this exists.** `metrics.json` today is a bare list of per-epoch entries and nothing in it names the run. Five such files are five anonymous tables, and nothing stops a comparison between a run scored at a 200 m tolerance and one scored at 300 m.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_checkpoints.py`:

```python
def test_a_journal_says_which_run_produced_it(tmp_path: Path) -> None:
    """Five rungs of a ladder are five metrics files, and a file that does not name its run
    compares to nothing. See docs/superpowers/specs/2026-08-17-small-target-adaptation-design.md."""
    journal = Journal(tmp_path / "run" / "metrics.json")
    journal.describe({"schedule": {"learning_rate": 0.005, "lr_schedule": "cosine"}})
    journal.record({"epoch": 1, "training_loss": 0.2})

    reread = Journal(tmp_path / "run" / "metrics.json")

    assert reread.run() == {"schedule": {"learning_rate": 0.005, "lr_schedule": "cosine"}}
    assert reread.entries() == [{"epoch": 1, "training_loss": 0.2}]


def test_a_journal_from_a_run_that_never_described_itself_says_so(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "run" / "metrics.json")
    journal.record({"epoch": 1})

    assert journal.run() is None


def test_the_metrics_of_a_run_that_predates_the_run_block_still_read_back(tmp_path: Path) -> None:
    """The first trained run wrote a bare list. A session already in flight when this shape
    changed must not be stranded by it — the resume logic reads this file every time it starts.
    What the bare list cannot do is serve as a rung, and `ladder.py` refuses it there instead."""
    path = tmp_path / "run" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text('[{"epoch": 1, "training_loss": 0.18}]')

    journal = Journal(path)

    assert journal.entries() == [{"epoch": 1, "training_loss": 0.18}]
    assert journal.run() is None


def test_describing_a_run_a_second_time_with_the_same_identity_is_allowed(tmp_path: Path) -> None:
    """Which is the ordinary case: a resumed session describes itself again."""
    journal = Journal(tmp_path / "run" / "metrics.json")
    journal.describe({"seed": 20260814})
    journal.describe({"seed": 20260814})

    assert journal.run() == {"seed": 20260814}


def test_resuming_a_run_under_a_different_configuration_is_refused(tmp_path: Path) -> None:
    """The failure this closes is the quiet one: a config edited between two Kaggle sessions
    produces a single metrics file whose first six epochs and last six epochs came from two
    different experiments, and nothing in the file says so."""
    journal = Journal(tmp_path / "run" / "metrics.json")
    journal.describe({"seed": 20260814, "schedule": {"learning_rate": 0.005}})

    with pytest.raises(ValueError, match="learning_rate"):
        journal.describe({"seed": 20260814, "schedule": {"learning_rate": 0.001}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_checkpoints.py -q`
Expected: FAIL — `AttributeError: 'Journal' object has no attribute 'describe'`

- [ ] **Step 3: Rewrite the `Journal` class**

Replace `src/darkvessel/detect/checkpoints.py:109-129` in full:

```python
class Journal:
    """The numbers a run reported, and which run reported them, in a file that needs nothing to
    read.

    Written beside the weights and not inside them: the point of a training level is a precision
    and a recall, and a reader should not need torch, a GPU or an unpickle to see them. Rewritten
    whole each time rather than appended to, so that a session killed mid-write cannot leave half
    a line that the next session reads back as a number.

    The run block was added for the ladder in issue #11, where five metrics files have to be
    compared against one another. Without it they are five anonymous tables, and nothing stops a
    comparison between a run scored at a 200 m tolerance and one scored at 300 m.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, Any]]:
        """One entry per epoch that was scored, in the order they ran."""
        return self._document()["epochs"]

    def run(self) -> dict[str, Any] | None:
        """What configuration produced these numbers. None for a run that never said."""
        return self._document()["run"]

    def describe(self, run: dict[str, Any]) -> None:
        """Name the run, once, before the first epoch is recorded.

        A resumed session calls this again with the same identity, which is allowed. A resumed
        session calling it with a *different* identity means a config was edited between two
        sessions, and the file would then hold two experiments under one name with nothing saying
        so — that is refused rather than merged.
        """
        document = self._document()
        if document["run"] is not None and document["run"] != run:
            differing = _differences(document["run"], run)
            raise ValueError(
                f"{self.path.name} was written by a run described differently: {differing}. "
                "Resuming under an edited config would put two experiments in one file"
            )

        document["run"] = run
        self._write(document)

    def record(self, entry: dict[str, Any]) -> None:
        document = self._document()
        document["epochs"] = [*document["epochs"], entry]
        self._write(document)

    def _document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"run": None, "epochs": []}

        loaded = json.loads(self.path.read_text())
        # A bare list is what runs before 2026-08-17 wrote. Read rather than refused, because the
        # resume path reads this file at the start of every session and a format change must not
        # strand a run already in flight. `ladder.py` is where an unnamed run is refused.
        if isinstance(loaded, list):
            return {"run": None, "epochs": list(loaded)}

        return {"run": loaded.get("run"), "epochs": list(loaded.get("epochs", []))}

    def _write(self, document: dict[str, Any]) -> None:
        with atomically(self.path) as partial:
            partial.write_text(json.dumps(document, indent=2))


def _differences(before: dict[str, Any], after: dict[str, Any], prefix: str = "") -> str:
    """The keys that disagree, named, so the refusal above says what was edited."""
    changed = []
    for key in sorted(set(before) | set(after)):
        here, there = before.get(key), after.get(key)
        if here == there:
            continue
        if isinstance(here, dict) and isinstance(there, dict):
            changed.append(_differences(here, there, f"{prefix}{key}."))
        else:
            changed.append(f"{prefix}{key}: {here!r} -> {there!r}")
    return ", ".join(changed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_checkpoints.py -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `make lint && make test`
Expected: PASS. `tests/test_training_run.py` is skipped without torch; if torch is installed locally it must still pass, because `entries()` kept its signature.

- [ ] **Step 6: Commit**

```bash
git add src/darkvessel/detect/checkpoints.py tests/test_checkpoints.py
git commit -m "feat: a metrics file says which run wrote it, and refuses a resume that changed"
```

---

### Task 3: The ladder — comparison, refusals, and the rule

**Files:**
- Create: `src/darkvessel/detect/ladder.py`
- Create: `tests/test_ladder.py`

**Interfaces:**
- Consumes: `Journal.run()` and `Journal.entries()` from Task 2.
- Produces:
  - `Rung(label: str, changed: str, run: dict | None, epochs: list[dict])`
  - `best_f1(entry: dict) -> float`
  - `band(epochs: list[dict], window: int = 4) -> float`
  - `Verdict(label, changed, statistic, against, band, gain, kept)` — `against`, `band` and `gain` are `None` for the first rung
  - `judge(rungs: list[Rung], window: int = 4) -> list[Verdict]`
  - `table(verdicts: list[Verdict]) -> str`

**The rule, restated so the implementer does not have to fetch the spec.** The statistic is the best F1 across the reported score thresholds, at the final epoch. The band is the range — maximum minus minimum — of that same statistic over the last four epochs of the previous rung. A rung is kept if its statistic exceeds the previous kept rung's by **more** than the band; a gain exactly equal to the band is a rejection. The next rung is compared against the last rung that was kept.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ladder.py`:

```python
"""Comparing five runs that differ by one line each.

Nothing here asserts a precision, a recall or an F1 that a model produced. The journals below are
written by hand precisely so that the arithmetic and the refusals can be pinned without pinning a
measurement — the numbers a real run reports are the output of the level, and a test that fixed
one would turn a measurement into a target.

What is pinned is the boundary. A rung whose gain exactly equals the noise band is rejected, and
that single `>` rather than `>=` is the difference between a ladder and a narration of noise.
"""

from pathlib import Path

import pytest

from darkvessel.cli import ladder_request_from
from darkvessel.config import load_config
from darkvessel.detect.ladder import Rung, band, best_f1, judge, table

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

REPORTING = {"tolerance_m": 200.0, "resolution_m": 10.0, "thresholds": [0.5, 0.75]}

# Every F1 in this file is a dyadic rational — a half, a quarter, an eighth — and that is not
# decoration. The boundary case below turns on a gain being *exactly* equal to a band, and a
# statistic of 0.52 would come back from the harmonic mean a fraction of an ulp away from 0.52.
# The test would then pass or fail on the rounding of a number nobody chose.


def an_epoch(epoch: int, precision: float, recall: float) -> dict:
    """One journal entry, at two thresholds, the second of which is the better one."""
    return {
        "epoch": epoch,
        "training_loss": 0.15,
        "learning_rate": 0.005,
        "held_out_tiles": 3000,
        "held_out_ships": 2378,
        "at": [
            {"score": 0.5, "precision": 0.1, "recall": 0.1, "found": 1, "false": 1, "missed": 1},
            {
                "score": 0.75,
                "precision": precision,
                "recall": recall,
                "found": 1,
                "false": 1,
                "missed": 1,
            },
        ],
    }


def a_rung(label: str, f1s: list[float], reporting: dict | None = None) -> Rung:
    """A rung whose final epochs have the F1s given. Precision and recall are set equal, so the
    harmonic mean of the pair is the number asked for."""
    return Rung(
        label=label,
        changed="one line",
        run={"reporting": reporting or REPORTING},
        epochs=[an_epoch(index + 1, f1, f1) for index, f1 in enumerate(f1s)],
    )


def test_the_statistic_is_the_best_f1_across_the_reported_thresholds() -> None:
    entry = an_epoch(1, precision=0.9, recall=0.7)

    assert best_f1(entry) == pytest.approx(2 * 0.9 * 0.7 / (0.9 + 0.7))


def test_a_threshold_that_reported_nothing_scores_zero_rather_than_perfectly() -> None:
    """`Counts.precision` is NaN when nothing was reported, on the argument that a run which
    returned nothing was neither right nor wrong. Carried into JSON, NaN must not sort to the top
    of a maximum."""
    entry = an_epoch(1, precision=float("nan"), recall=float("nan"))
    entry["at"][0] = {"score": 0.5, "precision": 0.6, "recall": 0.6}

    assert best_f1(entry) == pytest.approx(0.6)


def test_the_band_is_the_range_of_the_statistic_over_the_last_four_epochs() -> None:
    epochs = [
        an_epoch(index + 1, f1, f1) for index, f1 in enumerate([0.25, 0.875, 0.5, 0.625, 0.75])
    ]

    # The first epoch is outside the window; 0.875 - 0.5 is the range of the last four.
    assert band(epochs) == pytest.approx(0.375)


def test_the_first_rung_is_kept_because_there_is_nothing_to_compare_it_against() -> None:
    verdicts = judge([a_rung("R0", [0.5, 0.5, 0.5, 0.5])])

    assert verdicts[0].kept is True
    assert verdicts[0].against is None
    assert verdicts[0].band is None


def test_a_rung_whose_gain_exactly_equals_the_band_is_rejected() -> None:
    """The boundary, pinned. A gain that only just reaches the noise the previous rung was
    already showing is noise, and `>=` here would let every rung through on a bad week."""
    previous = a_rung("R0", [0.5, 0.75, 0.5, 0.5])  # band 0.25, final 0.5
    later = a_rung("R1", [0.75, 0.75, 0.75, 0.75])  # final 0.75, gain exactly 0.25

    verdicts = judge([previous, later])

    assert verdicts[1].kept is False


def test_a_rung_that_clears_the_band_by_a_margin_is_kept() -> None:
    previous = a_rung("R0", [0.5, 0.75, 0.5, 0.5])  # band 0.25, final 0.5
    later = a_rung("R1", [0.875, 0.875, 0.875, 0.875])  # final 0.875, gain 0.375

    verdicts = judge([previous, later])

    assert verdicts[1].kept is True
    assert verdicts[1].gain == pytest.approx(0.375)


def test_a_rejected_rung_is_not_what_the_next_one_is_measured_against() -> None:
    """The ladder is greedy: the next rung starts from the last rung that was kept, so a rejected
    rung must not become the baseline for the one after it."""
    r0 = a_rung("R0", [0.5, 0.75, 0.5, 0.5])  # band 0.25, final 0.5
    r1 = a_rung("R1", [0.25, 0.25, 0.25, 0.25])  # worse; rejected
    r2 = a_rung("R2", [0.875, 0.875, 0.875, 0.875])  # measured against R0, not R1

    verdicts = judge([r0, r1, r2])

    assert verdicts[1].kept is False
    assert verdicts[2].against == "R0"
    assert verdicts[2].gain == pytest.approx(0.375)


def test_two_rungs_scored_at_different_tolerances_are_refused() -> None:
    generous = a_rung("R0", [0.5, 0.5, 0.5, 0.5])
    strict = a_rung("R1", [0.6, 0.6, 0.6, 0.6], reporting={**REPORTING, "tolerance_m": 300.0})

    with pytest.raises(ValueError, match="tolerance_m"):
        judge([generous, strict])


def test_two_rungs_scored_at_different_thresholds_are_refused() -> None:
    r0 = a_rung("R0", [0.5, 0.5, 0.5, 0.5])
    r1 = a_rung("R1", [0.6, 0.6, 0.6, 0.6], reporting={**REPORTING, "thresholds": [0.9]})

    with pytest.raises(ValueError, match="thresholds"):
        judge([r0, r1])


def test_two_rungs_scored_over_different_held_out_splits_are_refused() -> None:
    """The one thing every rung of this ladder has in common is the split, and a rung scored over
    a subset of it would report a precision the detector had not earned."""
    r0 = a_rung("R0", [0.5, 0.5, 0.5, 0.5])
    r1 = a_rung("R1", [0.6, 0.6, 0.6, 0.6])
    for entry in r1.epochs:
        entry["held_out_tiles"] = 1500

    with pytest.raises(ValueError, match="held_out_tiles"):
        judge([r0, r1])


def test_a_rung_that_does_not_name_its_run_is_refused() -> None:
    """A bare-list metrics file from before the run block existed. It reads back for a resume and
    it does not serve as a rung, because nothing in it says how it was scored."""
    unnamed = Rung(label="R0", changed="nothing", run=None, epochs=[an_epoch(1, 0.5, 0.5)])

    with pytest.raises(ValueError, match="does not name"):
        judge([unnamed])


def test_the_table_names_every_rung_and_says_which_were_kept() -> None:
    verdicts = judge([a_rung("R0", [0.5, 0.75, 0.5, 0.5]), a_rung("R1", [0.25, 0.25, 0.25, 0.25])])

    rendered = table(verdicts)

    assert "R0" in rendered and "R1" in rendered
    assert "kept" in rendered and "rejected" in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ladder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'darkvessel.detect.ladder'`

- [ ] **Step 3: Write the implementation**

Create `src/darkvessel/detect/ladder.py`:

```python
"""Five runs that differ by one line each, and what separates a result from a draw.

Issue #11 asks for three adaptations, each measured against the configuration before it. The
measurement is the hard part. The baseline oscillated by more than any of the three changes is
likely to be worth — precision at a fixed threshold went 0.55, 0.74, 0.75, 0.41, ... 0.28, 0.80
across twelve epochs, under two different draws of the initial weights — so comparing one final
number against another would describe the draw and not the change, and would describe it with
three decimal places.

The rule this module applies is therefore written down in `docs/decisions.md` before any of the
runs it judges, and it is mechanical rather than discretionary: a rung is kept only if it beats
the previous kept rung by more than the noise that rung was already showing. A threshold chosen
after seeing the numbers is not a threshold, it is a narration of them.

No torch. What can go wrong here is a comparison between two runs that were not scored the same
way, which would produce a published claim that is false, and that belongs on the side of the
seam a laptop tests in a second.
"""

import math
from dataclasses import dataclass
from typing import Any

# How many of a rung's last epochs the noise band is measured over. Four is what a twelve-epoch
# schedule affords while still being past the point where a decaying learning rate has settled.
WINDOW = 4

# What two rungs must agree on before their numbers may be put beside one another.
SAME_REPORTING = ("tolerance_m", "resolution_m", "thresholds")
SAME_SPLIT = ("held_out_tiles", "held_out_ships")


@dataclass(frozen=True)
class Rung:
    """One run of the ladder: what it changed, how it was scored, and what it reported."""

    label: str
    changed: str
    run: dict[str, Any] | None
    epochs: list[dict[str, Any]]


@dataclass(frozen=True)
class Verdict:
    """What the rule made of one rung. `against`, `band` and `gain` are None for the first."""

    label: str
    changed: str
    statistic: float
    against: str | None
    band: float | None
    gain: float | None
    kept: bool


def best_f1(entry: dict[str, Any]) -> float:
    """The best F1 this epoch reached across the thresholds it was reported at.

    Derived here rather than recorded by the run, because the journal holds a precision and a
    recall at each threshold and no F1 — and one derivation in one place is the only way the
    number in the README and the number the rule is applied to are the same number.
    """
    return max(_f1(point["precision"], point["recall"]) for point in entry["at"])


def band(epochs: list[dict[str, Any]], window: int = WINDOW) -> float:
    """How much the statistic moved on its own over this rung's last epochs.

    This is the noise a later rung has to beat. It is measured on the rung being compared against
    rather than assumed, so a configuration that settles buys a tighter test for the next change
    and one that does not pays for it.
    """
    recent = [best_f1(entry) for entry in epochs[-window:]]
    return max(recent) - min(recent)


def judge(rungs: list[Rung], window: int = WINDOW) -> list[Verdict]:
    """Walk the ladder, applying the rule, and say what happened to each rung.

    Greedy, which is the issue text's own phrasing — each change is measured against the previous
    configuration. A rejected rung is not what the next one is measured against: the ladder goes
    on from the last rung that was kept, so a change that did not help cannot become the baseline
    that flatters the change after it.
    """
    verdicts: list[Verdict] = []
    standing: Rung | None = None

    for rung in rungs:
        _check_named(rung)
        statistic = best_f1(rung.epochs[-1])

        if standing is None:
            verdicts.append(
                Verdict(
                    label=rung.label,
                    changed=rung.changed,
                    statistic=statistic,
                    against=None,
                    band=None,
                    gain=None,
                    kept=True,
                )
            )
            standing = rung
            continue

        _check_comparable(standing, rung)
        noise = band(standing.epochs, window)
        gain = statistic - best_f1(standing.epochs[-1])
        # Strictly greater. A gain that only reaches the noise the previous rung was already
        # showing is noise, and this is the one character that says so.
        kept = gain > noise

        verdicts.append(
            Verdict(
                label=rung.label,
                changed=rung.changed,
                statistic=statistic,
                against=standing.label,
                band=noise,
                gain=gain,
                kept=kept,
            )
        )
        if kept:
            standing = rung

    return verdicts


def table(verdicts: list[Verdict]) -> str:
    """The ladder as a markdown table, for `docs/` and the README."""
    lines = [
        "| Rung | What changed | Best F1 | Against | Band | Gain | |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for verdict in verdicts:
        lines.append(
            f"| {verdict.label} | {verdict.changed} | {verdict.statistic:.3f} "
            f"| {verdict.against or '—'} | {_maybe(verdict.band)} | {_maybe(verdict.gain, sign=True)} "
            f"| {'kept' if verdict.kept else 'rejected'} |"
        )
    return "\n".join(lines)


def _check_named(rung: Rung) -> None:
    if rung.run is None:
        raise ValueError(
            f"{rung.label} does not name the run that produced it, so how it was scored is "
            "unknown; metrics files written before 2026-08-17 are bare lists and cannot be rungs"
        )


def _check_comparable(earlier: Rung, later: Rung) -> None:
    """Refuse two rungs whose numbers do not mean the same thing.

    This is the one place in the ladder where an error would be silent and would end up published.
    A tolerance of 300 m against one of 200 m, or a split of 1500 tiles against one of 3000, gives
    two tables that look alike, subtract cleanly, and mean nothing.
    """
    for field in SAME_REPORTING:
        here, there = earlier.run["reporting"][field], later.run["reporting"][field]
        if here != there:
            raise ValueError(
                f"{earlier.label} and {later.label} were scored with different {field} "
                f"({here!r} against {there!r}), so their numbers cannot be put beside one another"
            )

    for field in SAME_SPLIT:
        here, there = earlier.epochs[-1][field], later.epochs[-1][field]
        if here != there:
            raise ValueError(
                f"{earlier.label} and {later.label} were scored over different splits: "
                f"{field} {here!r} against {there!r}"
            )


def _f1(precision: float | None, recall: float | None) -> float:
    """Zero where a threshold reported nothing.

    `Counts.precision` is NaN when nothing was reported, on the argument that a run which returned
    nothing was neither right nor wrong. That survives into JSON, and a NaN loose in a `max` would
    make an empty detector the best rung on the ladder.
    """
    if precision is None or recall is None:
        return 0.0
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _maybe(value: float | None, sign: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if sign else f"{value:.3f}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ladder.py -q`
Expected: PASS, 13 tests

- [ ] **Step 5: Run the whole suite and commit**

```bash
make lint && make test
git add src/darkvessel/detect/ladder.py tests/test_ladder.py
git commit -m "feat: the rule that separates a rung from a draw, and the refusals under it"
```

---

### Task 4: `darkvessel compare`, and the ladder as a committed file

**Files:**
- Create: `configs/ladder.yaml`
- Create: `docs/runs/.gitkeep`
- Modify: `src/darkvessel/cli.py` — `main`, plus `ladder_request_from` and `_compare`
- Modify: `tests/test_pipeline.py:528-551`
- Modify: `tests/test_ladder.py`

**Interfaces:**
- Consumes: `Rung`, `judge`, `table` from Task 3; `Journal` from Task 2; `load_config` from Task 1.
- Produces: `ladder_request_from(config: dict, relative_to: Path) -> list[dict[str, Any]]`, each dict having `label`, `changed` and `metrics` (a resolved `Path`).

- [ ] **Step 1: Write `configs/ladder.yaml`**

```yaml
# The ladder of issue #11: five runs that differ by one line each, and the rule that decides
# which of them stand. Read it with:  darkvessel compare --config configs/ladder.yaml
#
# A rung whose metrics file is not there yet has not been run. The comparison reports it as
# pending and stops, rather than comparing across the gap.
#
# The rule, and why it is what it is, is in docs/decisions.md — written and committed before the
# first of these runs, because a threshold chosen after seeing the numbers is not a threshold.

ladder:
  # How many of a rung's last epochs the noise band is measured over.
  window: 4

  rungs:
    - label: R0
      changed: nothing — configs/train.yaml as it stands, under the corrected seeding
      metrics: ../docs/runs/r0-baseline.json

    - label: R1
      changed: cosine decay of the learning rate
      metrics: ../docs/runs/r1-cosine.json

    - label: R2
      changed: anchor_sizes to [[4], [8], [16], [32], [64]]
      metrics: ../docs/runs/r2-anchors.json

    - label: R3
      changed: single-channel stem
      metrics: ../docs/runs/r3-stem.json

    - label: R4
      changed: the RPN sampler, at the value the census dictates
      metrics: ../docs/runs/r4-sampler.json
```

Create the directory the rungs land in:

```bash
mkdir -p docs/runs && touch docs/runs/.gitkeep
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_ladder.py`. Its imports and `CONFIGS` are already at the top of that file
from Task 3, so nothing new is imported here:

```python
def test_the_shipped_ladder_config_is_the_one_the_command_parses() -> None:
    """The same gap `training_request_from` exists to close. This file names five paths that will
    not all exist until five Kaggle sessions have run, and a mistyped key in it would surface only
    to whoever came back with the last of them."""
    rungs = ladder_request_from(load_config(CONFIGS / "ladder.yaml"), CONFIGS)

    assert [rung["label"] for rung in rungs] == ["R0", "R1", "R2", "R3", "R4"]
    assert all(rung["changed"] for rung in rungs)
    assert all(rung["metrics"].is_absolute() for rung in rungs)


def test_every_rung_of_the_shipped_ladder_reads_its_metrics_from_its_own_file() -> None:
    """Two rungs pointing at one file would compare a run against itself and report a gain of
    zero, which reads as a rejection and would be recorded as one."""
    rungs = ladder_request_from(load_config(CONFIGS / "ladder.yaml"), CONFIGS)

    paths = [rung["metrics"] for rung in rungs]

    assert len(set(paths)) == len(paths)
```

- [ ] **Step 3: Run to verify they fail**

Run: `pytest tests/test_ladder.py -q`
Expected: FAIL — `ImportError: cannot import name 'ladder_request_from'`

- [ ] **Step 4: Add the request reader and the command**

In `src/darkvessel/cli.py`, add to the imports:

```python
from darkvessel.detect.ladder import Rung, judge, table
```

Add the subcommand in `main`, after `train_command`:

```python
    compare_command = commands.add_parser(
        "compare", help="read the rungs of a ladder of training runs against one another"
    )
    compare_command.add_argument("--config", type=Path, required=True)
```

and the branch, before the final `return _run(args.config)`:

```python
    if args.command == "compare":
        return _compare(args.config)
```

Add these two functions, beside `training_request_from`:

```python
def ladder_request_from(config: dict[str, Any], relative_to: Path) -> list[dict[str, Any]]:
    """The rungs a ladder config names, with their metrics files resolved.

    Separate from the command for the reason `training_request_from` is: this file names five
    paths, and the last of them does not exist until five sessions on a rented GPU have finished.
    A mistyped key surfacing then would be the most expensive way to find it.
    """
    return [
        {
            "label": str(rung["label"]),
            "changed": str(rung["changed"]),
            "metrics": (relative_to / rung["metrics"]).resolve(),
        }
        for rung in config["ladder"]["rungs"]
    ]


def _compare(config_path: Path) -> int:
    """Read the ladder and say which rungs stand.

    A rung whose metrics file is not there has not been run yet, which is the ordinary state of
    this file for most of the ticket. The comparison reports it as pending and stops there rather
    than skipping it — a ladder read across a gap would measure a change against the wrong
    configuration and would not look any different.
    """
    config = load_config(config_path)
    window = int(config["ladder"].get("window", 4))

    rungs = []
    for requested in ladder_request_from(config, config_path.parent):
        if not requested["metrics"].exists():
            print(f"{requested['label']}: not run yet ({requested['metrics']})")
            break

        journal = Journal(requested["metrics"])
        rungs.append(
            Rung(
                label=requested["label"],
                changed=requested["changed"],
                run=journal.run(),
                epochs=journal.entries(),
            )
        )

    if not rungs:
        print("no rung of this ladder has been run yet")
        return 0

    print(table(judge(rungs, window=window)))
    return 0
```

- [ ] **Step 5: Let the shipped-config test see the rungs**

`tests/test_pipeline.py:528` globs `configs/*.yaml` and requires each file to describe a run, a
survey or a training. Two things change: rung configs live one directory down and would escape the
test entirely, and `configs/ladder.yaml` describes none of the three.

Replace the decorator and the head of the test at `tests/test_pipeline.py:528-546`:

```python
@pytest.mark.parametrize(
    "shipped", sorted(CONFIGS.rglob("*.yaml")), ids=lambda path: str(path.name)
)
def test_every_shipped_config_names_the_fusion_settings_a_run_needs(shipped: Path) -> None:
    """The same gap `export_request_from` exists to close, on the settings fusion reads.

    Every test above writes its own config, so the shipped files are the ones nothing in the
    suite executes. `configs/kattegat-lane.yaml` is worse than that: running it needs Earth Engine
    credentials, so a missing key there surfaces only to someone who has already authenticated
    and waited. Both go through the command's own parsing here instead.

    Every file under `configs/` is covered, rungs of the ladder included — they are one directory
    down, and `rglob` rather than `glob` is what keeps them from dropping out of this test in
    silence. Files that are not runs have to say what they are instead.
    """
    config = load_config(shipped)
    if "run" not in config:
        assert "survey" in config or "training" in config or "ladder" in config, (
            f"{shipped.name} describes neither a run, a survey, a training nor a ladder"
        )
        return

    settings = fusion_settings_from(config)

    assert settings["tolerance_m"] > 0.0
    assert settings["max_gap"] > timedelta(0)
```

Add `from darkvessel.config import load_config` to that file's imports if it is not there.

- [ ] **Step 6: Run everything**

Run: `make lint && make test`
Expected: PASS. `darkvessel compare --config configs/ladder.yaml` prints `R0: not run yet …`.

Check it by hand:

```bash
python -m darkvessel.cli compare --config configs/ladder.yaml
```

- [ ] **Step 7: Commit**

```bash
git add configs/ladder.yaml docs/runs/.gitkeep src/darkvessel/cli.py tests/test_ladder.py tests/test_pipeline.py
git commit -m "feat: read the ladder from a file, and refuse to read it across a gap"
```

---

### Task 5: A single-channel stem that is the three-channel one at initialisation

**Files:**
- Modify: `src/darkvessel/detect/model.py`
- Modify: `src/darkvessel/detect/train.py:267-296` — `_Tiles`
- Modify: `src/darkvessel/detect/trained.py`
- Create: `tests/test_model_stem.py`

**Interfaces:**
- Produces:
  - `detector_model(*, tile_px, seed, anchor_sizes=ANCHOR_SIZES, stem="repeat", pretrained=True, trainable_backbone_layers=3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25) -> FasterRCNN`
  - `as_model_input(image: np.ndarray, stem: str = "repeat") -> Tensor`
  - `STEMS: dict[str, int]` — `{"repeat": 3, "single": 1}`
  - `train(..., stem: str = "repeat")` — Task 6 adds the scheduler to the same signature

**The arithmetic, and why the order of construction matters.** The current path repeats `x` across three channels, the transform normalises per channel, and `conv1` sums: `y = Σ_c W_c·(x − m_c)/s_c`. A one-channel `conv1` with `W'[k,0,i,j] = Σ_c W[k,c,i,j]/s_c` and `b'[k] = −Σ_c (m_c/s_c)·Σ_ij W[k,c,i,j]`, with the transform set to `image_mean=[0.0], image_std=[1.0]`, produces the same `y` exactly.

The fold must happen **after** the box predictor is replaced. `detector_model` seeds the global generator and then builds; constructing an extra `Conv2d` consumes from that generator, so folding before the predictor is built would give the two stems different heads and rung 3 would measure an initialisation.

The bias is not optional. With `trainable_backbone_layers: 3`, `bn1` is a `FrozenBatchNorm2d` applying fixed statistics rather than recentring a batch, so a constant offset propagates through the whole backbone instead of being absorbed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_stem.py`:

```python
"""The input stage, and the one property that makes rung 3 of the ladder a measurement.

LS-SSDD is VV and the scene this chain exports is VV, so the dual-polarisation stem issue #11
asks for has no data on either side — see docs/failures.md. What is delivered instead is an
honest single-channel stem, and what is asserted here is that it starts life as exactly the model
the three-channel repeat starts life as. Without that, the rung would be comparing two different
initialisations and reporting the difference as an adaptation.

Skipped where torch is not installed, which includes CI.
"""

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from darkvessel.detect.model import as_model_input, detector_model  # noqa: E402

TILE_PX = 64


def a_model(stem: str):
    """Untrained, because a test that fetched 160 MB of COCO weights is not a test anyone runs.
    The fold is exact whatever the weights are."""
    return detector_model(
        tile_px=TILE_PX, seed=1, pretrained=False, trainable_backbone_layers=5, stem=stem
    ).eval()


def features(stem: str, image: np.ndarray):
    """The top of the feature pyramid, through the model's own transform.

    Through the transform rather than around it because the normalisation is half of what the
    fold absorbs, and a test that skipped it would pass with the bias missing.
    """
    model = a_model(stem)
    with torch.no_grad():
        images, _ = model.transform([as_model_input(image, stem)], None)
        return model.backbone(images.tensors)["0"]


def test_the_single_channel_stem_is_the_three_channel_one_at_initialisation() -> None:
    """The property rung 3 rests on. Same weights, same normalisation, same output — so what the
    rung measures is what training does with one bank of 7x7x1 kernels instead of three."""
    image = np.random.default_rng(0).random((TILE_PX, TILE_PX)).astype(np.float32)

    assert torch.allclose(features("repeat", image), features("single", image), atol=1e-5)


def test_the_single_channel_stem_takes_one_channel() -> None:
    model = a_model("single")

    assert model.backbone.body.conv1.in_channels == 1
    assert model.transform.image_mean == [0.0]
    assert model.transform.image_std == [1.0]


def test_the_bias_the_fold_needs_is_there() -> None:
    """`bn1` is frozen at these settings, so it applies fixed statistics instead of recentring the
    batch. Drop the bias and the two paths differ by a constant through the whole backbone."""
    assert a_model("single").backbone.body.conv1.bias is not None


def test_one_tile_reaches_the_model_with_the_channels_its_stem_expects() -> None:
    image = np.zeros((TILE_PX, TILE_PX), dtype=np.float32)

    assert as_model_input(image, stem="repeat").shape == (3, TILE_PX, TILE_PX)
    assert as_model_input(image, stem="single").shape == (1, TILE_PX, TILE_PX)


def test_a_stem_this_project_does_not_have_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="dual"):
        detector_model(tile_px=TILE_PX, seed=1, pretrained=False, stem="dual")


def test_a_checkpoint_is_refused_by_a_run_asking_for_the_other_stem() -> None:
    """Anchors leave no trace in a state dict and neither does a stem — except that a folded
    `conv1` has a different shape, so this one would at least fail loudly. It is checked anyway,
    beside the anchors, so the refusal reads as one rule rather than two accidents."""
    from darkvessel.detect.trained import _check_built

    built = {"tile_px": TILE_PX, "anchor_sizes": ((32,),), "stem": "single"}

    with pytest.raises(ValueError, match="stem"):
        _check_built(built, tile_px=TILE_PX, anchor_sizes=((32,),), stem="repeat")


def test_a_checkpoint_written_before_stems_existed_is_read_as_the_repeat_it_was() -> None:
    """`models/epoch-012.pt` and everything before 2026-08-17 has no stem in its build block, and
    every one of them was trained on three repeated channels."""
    from darkvessel.detect.trained import _check_built

    _check_built(
        {"tile_px": TILE_PX, "anchor_sizes": ((32,),)},
        tile_px=TILE_PX,
        anchor_sizes=((32,),),
        stem="repeat",
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_model_stem.py -q`
Expected: FAIL — `TypeError: detector_model() got an unexpected keyword argument 'stem'`

- [ ] **Step 3: Change `model.py`**

Add beside `ASPECT_RATIOS` in `src/darkvessel/detect/model.py`:

```python
# How many channels each input stage takes. "repeat" is the baseline the ladder starts from: one
# polarisation copied three times, which is the minimum a three-channel ImageNet backbone accepts
# and is not an adaptation to anything. "single" is the adaptation — one channel of radar
# amplitude, with the pretrained stem folded down onto it.
#
# There is no "dual". LS-SSDD is VV and the scene this chain exports is VV, so a dual-polarisation
# stem has no data to be fitted on and none to be run on. See docs/failures.md.
STEMS = {"repeat": 3, "single": 1}
```

Replace the signature and body of `detector_model`, keeping its existing docstring and adding to it:

```python
def detector_model(
    *,
    tile_px: int,
    seed: int,
    anchor_sizes: tuple[tuple[int, ...], ...] = ANCHOR_SIZES,
    stem: str = "repeat",
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    rpn_batch_size_per_image: int = 256,
    rpn_positive_fraction: float = 0.5,
    box_batch_size_per_image: int = 512,
    box_positive_fraction: float = 0.25,
) -> FasterRCNN:
    """A Faster R-CNN sized for ships of a few pixels on Sentinel-1.

    Args:
        tile_px: The side of the tiles this model is trained and run on. Fixed rather than
            inferred, so that the transform inside the model resamples nothing: rescaling radar
            amplitude changes what the detector sees, and that is a decision about a run rather
            than a convenience — the same argument `cli.py` makes about reprojecting a scene.
        seed: The run's seed. Names the weights as well as the data, which it did not until a
            run of the same config twice produced two different models.
        anchor_sizes: One tuple per pyramid level. Configurable because this is the number most
            likely to want moving once there are real numbers to move it against.
        stem: `"repeat"` or `"single"`. The single-channel stem is built to produce exactly what
            the repeat produces at initialisation, so the rung that introduces it measures what
            training does with it rather than a different starting point.
        pretrained: Start from COCO weights. A free tier gives too few epochs to train a
            ResNet-50 from scratch; set it false only where the session has no network to fetch
            them with.
        trainable_backbone_layers: How much of the backbone is unfrozen, from the top. Three of
            five is torchvision's default and is what the budget here affords.
        rpn_batch_size_per_image: How many anchors the region proposal network computes its loss
            over. Torchvision's default is 256. Lowering it is the lever on this data: with a
            handful of ships to a tile there are nowhere near 128 positive anchors to be had, so
            the remaining slots fill with background whatever `rpn_positive_fraction` asks for.
        rpn_positive_fraction: A **ceiling** on how much of that batch may be positive, not a
            target — the sampler takes `min(available, requested)`. Stated here because the
            distinction is what rung 4 of the ladder turns on.
        box_batch_size_per_image: The same, for the second-stage head.
        box_positive_fraction: The same, for the second-stage head.
    """
    if stem not in STEMS:
        raise ValueError(f"unknown stem {stem!r}; this project has {sorted(STEMS)} and no dual")

    # Applied here, before anything is constructed, because the head below is initialised from
    # scratch — two classes where COCO had 91 — and it draws from torch's global generator. Left
    # unseeded, two sessions of the same config start from two different models and report
    # different numbers, for a reason nothing in the config records. Found by running the same
    # configuration twice: see docs/failures.md.
    torch.manual_seed(seed)

    channels = STEMS[stem]
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        # Freezing is a claim about weights that were fitted on something else. Without them
        # there is nothing to preserve, and torchvision says so rather than silently obeying.
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        rpn_anchor_generator=AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=(ASPECT_RATIOS,) * len(anchor_sizes),
        ),
        # The single-channel stem absorbs the normalisation into its own weights, so the transform
        # in front of it has nothing left to do.
        image_mean=IMAGENET_MEAN if channels == 3 else [0.0],
        image_std=IMAGENET_STD if channels == 3 else [1.0],
        min_size=tile_px,
        max_size=tile_px,
        rpn_batch_size_per_image=rpn_batch_size_per_image,
        rpn_positive_fraction=rpn_positive_fraction,
        box_batch_size_per_image=box_batch_size_per_image,
        box_positive_fraction=box_positive_fraction,
    )

    # The COCO head predicts 91 classes. Ships are one of them, and reusing that column would be
    # a defensible shortcut on optical imagery; on radar amplitude the features underneath it are
    # different enough that it is not worth the confusion of explaining. Fresh head, two classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, CLASSES
    )

    # After the head, and that ordering is load-bearing. Building a `Conv2d` draws from the global
    # generator, so folding the stem before the head above would give the two stems different
    # heads — and the rung that introduces the stem would be measuring an initialisation.
    if channels == 1:
        _fold_stem(model)

    return model


def _fold_stem(model: FasterRCNN) -> None:
    """Replace the three-channel `conv1` with the one-channel convolution that computes the same
    thing.

    The repeat path is `y = Σ_c W_c · (x − m_c) / s_c`, where `m` and `s` are ImageNet's per-channel
    statistics and `x` is one polarisation of amplitude copied three times. Summing the kernels
    over the channel axis, each divided by its own standard deviation, gives a one-channel weight
    that reproduces the first term; the second is a constant per output channel, which is a bias.

    The bias would be redundant in a stock ResNet, where the batch norm behind `conv1` recentres
    whatever arrives. It is not redundant here: at the trainable-layer counts this project uses,
    `bn1` is a `FrozenBatchNorm2d` applying fixed statistics, so a constant offset propagates
    through the entire backbone instead of being absorbed.
    """
    conv1 = model.backbone.body.conv1
    weight = conv1.weight.data
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    folded = torch.nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=True,
    )
    folded.weight.data = (weight / std).sum(dim=1, keepdim=True)
    folded.bias.data = -(weight * mean / std).sum(dim=(1, 2, 3))

    model.backbone.body.conv1 = folded
```

Replace `as_model_input`:

```python
def as_model_input(image: np.ndarray, stem: str = "repeat") -> Tensor:
    """One tile of amplitude in 0..1, as the image this model's stem expects.

    Repeated rather than averaged or padded with zeros, under the repeat stem: a grey image is
    something the pretrained filters have seen, and two channels of zeros is not. Under the single
    stem there is nothing to repeat — the fold has already put the three kernels into one.
    """
    tile = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
    return tile.repeat(STEMS[stem], 1, 1)
```

- [ ] **Step 4: Carry the stem into the training loop**

In `src/darkvessel/detect/train.py`, add `stem` to `train`'s signature after `built`:

```python
    built: dict[str, Any],
    stem: str = "repeat",
    say: Callable[[str], None] = print,
```

Pass it to both `_Tiles` constructions — in `_one_epoch` and in `_score` — which means threading it through those two functions' signatures as a keyword argument. In `_Tiles.__init__` add `stem: str = "repeat"`, store it, and use it in `__getitem__`:

```python
        return as_model_input(tile.image, self.stem), {
```

`_score` is called from two places in `train`; give both the `stem=stem` keyword.

- [ ] **Step 5: Carry the stem into inference**

In `src/darkvessel/detect/trained.py`, add `stem: str = "repeat"` to `TrainedDetector.__init__`'s keyword arguments, pass it to `_check_built(...)` and to `detector_model(...)`, store it as `self.stem`, and use it in `__call__`:

```python
            tile = as_model_input(self.stretch(image), self.stem).to(self.device)
```

Extend `_check_built` with a third parameter and a third refusal:

```python
def _check_built(
    built: dict[str, Any] | None,
    *,
    tile_px: int,
    anchor_sizes: tuple[tuple[int, ...], ...],
    stem: str = "repeat",
) -> None:
```

and, after the anchor check:

```python
    # Absent from every checkpoint written before 2026-08-17, all of which were trained on three
    # repeated channels — so silence means "repeat" rather than "unknown".
    recorded_stem = built.get("stem", "repeat")
    if recorded_stem != stem:
        raise ValueError(
            f"the checkpoint was built with the {recorded_stem!r} stem and this run asks for "
            f"{stem!r}; the two take a different number of channels"
        )
```

In `src/darkvessel/cli.py`, `trained_request_from` gains `"stem": run_config["detector"].get("stem", "repeat")` — matching how that function already reads the rest of the detector block — and `_detector_from` passes it through to `TrainedDetector`.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_model_stem.py tests/test_trained_detector.py tests/test_training_run.py -q`
Expected: PASS where torch is installed; skipped otherwise.

Run: `make lint && make test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/darkvessel/detect/model.py src/darkvessel/detect/train.py src/darkvessel/detect/trained.py src/darkvessel/cli.py tests/test_model_stem.py
git commit -m "feat: a stem that takes one polarisation, and starts as the model it replaces"
```

---

### Task 6: A learning-rate schedule that survives a resume

**Files:**
- Modify: `src/darkvessel/detect/train.py`
- Modify: `tests/test_training_run.py`

**Interfaces:**
- Consumes: `train(..., stem=...)` from Task 5.
- Produces: `Schedule(..., lr_schedule: str = "constant")`. Journal entries gain a `"learning_rate"` key.

**Where the step goes, and why.** The scheduler is stepped after the epoch's training and **before** the checkpoint is written. The state that lands on the disk with epoch N is therefore the state a session resuming at epoch N+1 needs, and the resumed session loads it and starts without stepping. The learning rate recorded in the journal is the one the epoch was actually trained at, captured before the step.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_run.py`. It needs `a_run` to take a schedule, so change that helper's signature to `def a_run(tmp_path: Path, epochs: int, lr_schedule: str = "constant") -> dict:` and pass `lr_schedule=lr_schedule` into the `Schedule(...)` it builds.

```python
def test_a_resumed_session_continues_the_learning_rate_schedule(tmp_path: Path) -> None:
    """The failure a schedule introduces, and the one this design cannot afford.

    Everything about a run is derived from the seed and the epoch number rather than carried in a
    generator's position, so a session resumed at epoch 3 does what an uninterrupted run would
    have done there. A learning-rate scheduler is the first piece of state in this loop that does
    not work that way: left out of the checkpoint it restarts from the top, the resumed session
    trains its remaining epochs at the wrong rate, and nothing anywhere says so.

    So an interrupted run and an uninterrupted one are required to report the same rates.
    """
    straight = tmp_path / "straight"
    train(**(a_run(straight, epochs=4, lr_schedule="cosine")))
    uninterrupted = [
        entry["learning_rate"] for entry in Journal(straight / "run" / "metrics.json").entries()
    ]

    killed = tmp_path / "killed"
    train(**(a_run(killed, epochs=2, lr_schedule="cosine")))
    train(**(a_run(killed, epochs=4, lr_schedule="cosine")))
    resumed = [
        entry["learning_rate"] for entry in Journal(killed / "run" / "metrics.json").entries()
    ]

    assert len(uninterrupted) == 4
    assert resumed == pytest.approx(uninterrupted)


def test_a_constant_schedule_reports_the_one_rate_it_trained_at(tmp_path: Path) -> None:
    """The baseline of the ladder, and the shape the failure log's diagnosis rests on: the rate
    never moved, which is why twelve epochs bounced instead of settling."""
    train(**(a_run(tmp_path, epochs=2)))

    rates = [entry["learning_rate"] for entry in Journal(tmp_path / "run" / "metrics.json").entries()]

    assert rates == [0.001, 0.001]


def test_a_schedule_this_project_does_not_have_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lr_schedule"):
        train(**(a_run(tmp_path, epochs=1, lr_schedule="exponential")))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_training_run.py -q`
Expected: FAIL — `TypeError: Schedule.__init__() got an unexpected keyword argument 'lr_schedule'`

- [ ] **Step 3: Add the field and the scheduler**

In `src/darkvessel/detect/train.py`, add to `Schedule`:

```python
    # "constant" is what the first run used and what the ladder's baseline keeps. The failure log
    # records what it cost: twelve epochs that reached the neighbourhood of a minimum in three and
    # bounced around it for nine, while the training loss stayed nearly flat and said nothing.
    lr_schedule: str = "constant"
```

Add, below the `Schedule` dataclass:

```python
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
    if schedule.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=schedule.epochs)

    raise ValueError(
        f"unknown lr_schedule {schedule.lr_schedule!r}; this project has 'constant' and 'cosine'"
    )
```

In `train`, after the optimiser is built:

```python
    scheduler = _scheduler(optimiser, schedule)
```

In the resume block, after `optimiser.load_state_dict(...)`:

```python
        # A scheduler left out of the checkpoint restarts from the top, and the resumed session
        # trains its remaining epochs at rates an uninterrupted run would never have used. None
        # in a checkpoint written by a constant-rate run, which is why this is guarded twice.
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
```

In the epoch loop, replace the body from `loss = _one_epoch(...)` down to the `_report(...)` call:

```python
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
        landed = checkpoints.path_for(epoch)
        say(
            f"epoch {epoch}: loss {loss:.4f}, rate {rate:.5f}, "
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
```

Give `_report` the new keyword and put it in the entry, beside `training_loss`:

```python
def _report(
    epoch: int,
    *,
    loss: float | None,
    learning_rate: float | None,
    attempt: Attempt,
    ...
```

```python
            "training_loss": loss,
            "learning_rate": learning_rate,
```

The resume-scoring call near the top of `train` passes `learning_rate=None` for the same reason it passes `loss=None`: the number was lost with the session that used it, and a value there would be a rate nobody trained at.

- [ ] **Step 4: Name the run in its journal**

Still in `train`, immediately after `model.to(device)`:

```python
    # Written before the first epoch, so a metrics file says which configuration produced it. Five
    # rungs of a ladder are five of these files, and one that does not name its run compares to
    # nothing. `describe` refuses a resume under an edited config rather than merging two
    # experiments into one file.
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
```

Add `asdict` to the dataclasses import at the top of the module:

```python
from dataclasses import asdict, dataclass
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_training_run.py -q`
Expected: PASS where torch is installed.

Run: `make lint && make test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/darkvessel/detect/train.py tests/test_training_run.py
git commit -m "feat: the rate decays, the schedule survives a resume, and the journal records both"
```

---

### Task 7: The config keys, and the four rungs as files

**Files:**
- Modify: `configs/train.yaml`
- Modify: `src/darkvessel/cli.py:312-370` — `training_request_from`; `:401-411` — `_train`
- Create: `configs/ladder/r1-cosine.yaml`, `r2-anchors.yaml`, `r3-stem.yaml`, `r4-sampler.yaml`
- Modify: `tests/test_training_run.py`

**Interfaces:**
- Consumes: `load_config` (Task 1), `detector_model`'s new keywords (Task 5), `Schedule.lr_schedule` (Task 6).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training_run.py`:

```python
LADDER = CONFIG.parent / "ladder"


def test_the_ladder_has_the_four_rungs_the_plan_names() -> None:
    """First, because the three tests below are parametrised over this directory and would all
    pass vacuously on an empty one — which is exactly the state the repository is in before this
    task, and exactly the way a missing rung would go unnoticed after it."""
    assert sorted(path.name for path in LADDER.glob("*.yaml")) == [
        "r1-cosine.yaml",
        "r2-anchors.yaml",
        "r3-stem.yaml",
        "r4-sampler.yaml",
    ]


@pytest.mark.parametrize("rung", sorted(LADDER.glob("*.yaml")), ids=lambda path: path.name)
def test_every_rung_of_the_ladder_is_a_training_config_the_command_parses(rung: Path) -> None:
    """Each of these is run on a machine rented by the hour, days apart. A mistyped key in the
    fourth would surface after three evenings had already been spent."""
    request = training_request_from(load_config(rung), rung.parent)

    assert request["schedule"]["epochs"] > 0
    assert request["model"]["stem"] in {"repeat", "single"}


@pytest.mark.parametrize("rung", sorted(LADDER.glob("*.yaml")), ids=lambda path: path.name)
def test_every_rung_writes_its_checkpoints_and_its_metrics_somewhere_of_its_own(
    rung: Path,
) -> None:
    """The trap this closes is quiet and expensive. Rungs share a working directory on Kaggle, and
    a rung that inherited the previous one's checkpoint directory would find a finished schedule
    there and do nothing at all — reporting the previous rung's numbers as its own."""
    request = training_request_from(load_config(rung), rung.parent)
    baseline = training_request_from(load_config(CONFIG), CONFIG.parent)

    assert request["checkpoints"].directory != baseline["checkpoints"].directory
    assert request["journal"].path != baseline["journal"].path


def test_the_rungs_of_the_ladder_do_not_share_a_working_directory() -> None:
    requests = [
        training_request_from(load_config(rung), rung.parent) for rung in LADDER.glob("*.yaml")
    ]

    directories = [request["checkpoints"].directory for request in requests]
    metrics = [request["journal"].path for request in requests]

    assert len(set(directories)) == len(directories)
    assert len(set(metrics)) == len(metrics)
```

Add `from darkvessel.config import load_config  # noqa: E402` to that file's imports.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_training_run.py -q`
Expected: FAIL on `test_the_ladder_has_the_four_rungs_the_plan_names` — the `ladder` directory does
not exist yet, so the assertion compares against an empty list. The three parametrised tests below
it collect nothing and report as passing, which is why that first test is there.

- [ ] **Step 3: Add the keys to `configs/train.yaml`**

In the `model:` block, after `anchor_sizes`:

```yaml
  # The input stage. "repeat" copies one polarisation across the three channels an ImageNet
  # backbone accepts, which is the minimum that runs at all and is not an adaptation to anything.
  # "single" takes one channel and folds the pretrained stem onto it. There is no dual-polarisation
  # stem, because LS-SSDD is VV and the scene this chain exports is VV — see docs/failures.md.
  stem: repeat
  # How many anchors the region proposal network computes its loss over, and the ceiling on how
  # much of that may be positive. A ceiling, not a target: the sampler takes min(available,
  # requested), and on a tile holding three ships of four pixels there are nowhere near 128
  # positive anchors to be had. Torchvision's own defaults, kept as the baseline the ladder
  # measures against; notebooks/anchor_census.py is what says where to move them.
  rpn_batch_size_per_image: 256
  rpn_positive_fraction: 0.5
  box_batch_size_per_image: 512
  box_positive_fraction: 0.25
```

In the `schedule:` block, after `weight_decay`:

```yaml
  # "constant" is what the first run used, and docs/failures.md records what it cost: the model
  # reached the neighbourhood of a minimum in three epochs and bounced around it for nine more,
  # while the loss stayed flat and said nothing. "cosine" is the ladder's first rung.
  lr_schedule: constant
```

- [ ] **Step 4: Read them in `training_request_from`**

In the `"model"` dict:

```python
            "anchor_sizes": tuple(tuple(level) for level in config["model"]["anchor_sizes"]),
            "stem": str(config["model"]["stem"]),
            "rpn_batch_size_per_image": int(config["model"]["rpn_batch_size_per_image"]),
            "rpn_positive_fraction": float(config["model"]["rpn_positive_fraction"]),
            "box_batch_size_per_image": int(config["model"]["box_batch_size_per_image"]),
            "box_positive_fraction": float(config["model"]["box_positive_fraction"]),
```

In the `"schedule"` dict:

```python
            "lr_schedule": str(config["schedule"]["lr_schedule"]),
```

In `_train`, pass the stem to the loop as well as to the builder:

```python
    train(
        model=detector_model(tile_px=request["tile_px"], **request["model"]),
        ...
        built={"tile_px": request["tile_px"], **request["model"]},
        stem=request["model"]["stem"],
    )
```

- [ ] **Step 5: Write the four rungs**

`configs/ladder/r1-cosine.yaml`:

```yaml
# Rung 1 of the ladder in issue #11: the learning rate decays.
#
# Not one of the ticket's three adaptations, and here for the reason docs/failures.md gives: the
# baseline oscillated by more than any of the three is likely to be worth, so without this every
# later comparison measures the draw. Measured against R0 like any other rung.

extends: ../train.yaml

schedule:
  lr_schedule: cosine

out:
  checkpoints: /kaggle/working/checkpoints-r1
  metrics: /kaggle/working/metrics-r1-cosine.json
```

`configs/ladder/r2-anchors.yaml`:

```yaml
# Rung 2: anchors for ships of a few pixels.
#
# The stock smallest anchor is 32 px, a 320 m vessel at 10 m, longer than nearly everything in the
# training set. These are 40 m upwards. Extends rung 1 because the ladder is greedy: each rung
# starts from the last one that was kept. If R1 is rejected, this line is repointed at
# ../train.yaml and the edit goes into docs/failures.md with the rejection.

extends: r1-cosine.yaml

model:
  anchor_sizes: [[4], [8], [16], [32], [64]]

out:
  checkpoints: /kaggle/working/checkpoints-r2
  metrics: /kaggle/working/metrics-r2-anchors.json
```

`configs/ladder/r3-stem.yaml`:

```yaml
# Rung 3: one channel of radar amplitude instead of three copies of it.
#
# The stem is built to produce exactly what the repeat produces at initialisation, so what this
# rung measures is what training does with one bank of kernels rather than a different starting
# point. See docs/decisions.md.

extends: r2-anchors.yaml

model:
  stem: single

out:
  checkpoints: /kaggle/working/checkpoints-r3
  metrics: /kaggle/working/metrics-r3-stem.json
```

`configs/ladder/r4-sampler.yaml`:

```yaml
# Rung 4: the foreground/background imbalance, at the loss.
#
# The number below is a placeholder until notebooks/anchor_census.py has run — the census counts
# how many positive anchors a tile actually offers, and this is set to what it reports rather than
# to a guess. Do not run this rung before the census, and record the census figure in
# docs/decisions.md beside the value chosen here.

extends: r3-stem.yaml

model:
  rpn_batch_size_per_image: 32

out:
  checkpoints: /kaggle/working/checkpoints-r4
  metrics: /kaggle/working/metrics-r4-sampler.json
```

- [ ] **Step 6: Run everything**

Run: `make lint && make test`
Expected: PASS, including the shipped-config test from Task 4, which now walks the rungs too.

- [ ] **Step 7: Commit**

```bash
git add configs/train.yaml configs/ladder src/darkvessel/cli.py tests/test_training_run.py
git commit -m "feat: four rungs, each a file stating the one line it changes"
```

---

### Task 8: The census that decides rungs 2 and 4

**Files:**
- Create: `notebooks/anchor_census.py`

**Interfaces:**
- Consumes: `detector_model` (Task 5), `catalogue`, `split_by_scene`, `Layout` from `dataset.py`.
- Produces: printed counts, pasted by a human into `docs/decisions.md`.

There is no test for this. It is a measurement script in the shape of `notebooks/sweep_window.py`, it reads a dataset that exists only inside a Kaggle session, and what it produces is numbers for a decision log rather than behaviour anything depends on.

- [ ] **Step 1: Write the script**

Create `notebooks/anchor_census.py`:

```python
"""How many anchors ever match a ship, and how many the sampler can therefore find.

Two rungs of the ladder in issue #11 turn on numbers nobody has measured. The first is the anchor
sizing: the stock set starts at 32 px, a 320 m vessel at 10 m, and the argument for moving it is
currently an argument from arithmetic rather than from a count. The second is the imbalance, and
it is the one that is easy to get wrong. Faster R-CNN already subsamples — 256 anchors at a
ceiling of 50% positive — so the 1000:1 imbalance of a dense detector does not exist here. What
may exist is subtler: on a tile holding three ships of four pixels there may be nowhere near 128
positive anchors to be had, in which case `rpn_positive_fraction` is not the lever at all. It is a
ceiling, not a target, and the sampler takes min(available, requested). The lever would then be
`rpn_batch_size_per_image`, moved *down*, so the few positives are not drowned.

The prediction, written before this was run: a realised positive fraction near 1%, not 50%. If the
census contradicts it, that is recorded as a contradiction — see README.md on the recall
prediction this project got wrong before.

Costs no GPU quota. It reads boxes, not images, and runs on a CPU session in minutes.

    python3 notebooks/anchor_census.py
"""

from collections import Counter
from pathlib import Path

import torch
# A private module of torchvision's, and named here rather than reimplemented: the point of this
# census is what torchvision's own matcher does with these boxes, not what a copy of it would.
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

from darkvessel.detect.dataset import Layout, catalogue, split_by_scene
from darkvessel.detect.model import ANCHOR_SIZES, detector_model

ROOT = Path("/kaggle/input/ls-ssdd-v10/LS-SSDD-v1.0-OPEN")
LAYOUT = Layout(images="JPEGImages", annotations="Annotations", image_suffix=".jpg")
TILE_PX = 800

# What rung 2 proposes, against what the baseline ships.
CANDIDATES = {
    "stock (the baseline)": ANCHOR_SIZES,
    "small (rung 2)": ((4,), (8,), (16,), (32,), (64,)),
}


def anchors_for(sizes: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, list[int]]:
    """Every anchor one tile offers, and how many of them belong to each pyramid level.

    The feature-map shapes come from running the backbone once on a dummy tile rather than from
    the stride arithmetic, because the arithmetic is a claim about torchvision's FPN and this is
    the measurement of it.
    """
    model = detector_model(
        tile_px=TILE_PX, seed=0, anchor_sizes=sizes, pretrained=False
    ).eval()

    blank = torch.zeros(1, 3, TILE_PX, TILE_PX)
    with torch.no_grad():
        features = list(model.backbone(blank).values())

    images = ImageList(blank, [(TILE_PX, TILE_PX)])
    per_level = [
        level.shape[-2] * level.shape[-1] * len(model.rpn.anchor_generator.aspect_ratios[0])
        for level in features
    ]
    return model.rpn.anchor_generator(images, features)[0], per_level


def census(sizes: tuple[tuple[int, ...], ...], refs: list) -> None:
    anchors, per_level = anchors_for(sizes)
    # Torchvision's own thresholds, and its own guarantee that every box gets at least one anchor
    # however poor the overlap. That guarantee is why "zero positives" never happens and why the
    # count, not the presence, is the thing worth measuring.
    matcher = Matcher(0.7, 0.3, allow_low_quality_matches=True)

    boundaries = torch.tensor(per_level).cumsum(0)
    positives_per_tile = []
    by_level: Counter[int] = Counter()
    rescued = 0

    for ref in refs:
        boxes = torch.tensor([box.to_xyxy() for box in ref.boxes], dtype=torch.float32)
        if not len(boxes):
            continue

        quality = box_iou(boxes, anchors)
        matched = matcher(quality)
        positive = (matched >= 0).nonzero().flatten()
        positives_per_tile.append(len(positive))

        for index in positive.tolist():
            by_level[int((boundaries <= index).sum())] += 1

        # Boxes whose best anchor never reached 0.7 and were matched only by the low-quality rule.
        rescued += int((quality.max(dim=1).values < 0.7).sum())

    total = sum(positives_per_tile)
    tiles = len(positives_per_tile)
    print(f"  ship-bearing tiles: {tiles}")
    print(f"  positive anchors per tile: mean {total / max(tiles, 1):.1f}, "
          f"min {min(positives_per_tile)}, max {max(positives_per_tile)}")
    print(f"  boxes matched only by allow_low_quality_matches: {rescued}")
    print(f"  by pyramid level: {dict(sorted(by_level.items()))}")
    # 256 anchors are sampled at a ceiling of 50% positive, so 128 are asked for.
    print(f"  realised positive fraction against a batch of 256: "
          f"{min(total / max(tiles, 1), 128.0) / 256:.3f}")


def main() -> None:
    refs = catalogue(ROOT, LAYOUT)
    training, _ = split_by_scene(refs)
    ship_bearing = [ref for ref in training if ref.boxes]

    sizes = sorted(
        max(box.to_xyxy()[2] - box.to_xyxy()[0], box.to_xyxy()[3] - box.to_xyxy()[1])
        for ref in ship_bearing
        for box in ref.boxes
    )
    print(f"{len(sizes)} ships over {len(ship_bearing)} training tiles")
    print(f"longest side in pixels: p05 {sizes[len(sizes) // 20]:.1f}, "
          f"median {sizes[len(sizes) // 2]:.1f}, p95 {sizes[-len(sizes) // 20]:.1f}")

    for label, candidate in CANDIDATES.items():
        print(f"\n{label}: {candidate}")
        census(candidate, ship_bearing)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check it imports and lints**

Run: `make lint`
Expected: PASS.

Run: `python -c "import ast, pathlib; ast.parse(pathlib.Path('notebooks/anchor_census.py').read_text())"`
Expected: no output. It cannot be executed here — `ROOT` exists only inside a Kaggle session.

- [ ] **Step 3: Commit**

```bash
git add notebooks/anchor_census.py
git commit -m "feat: count the positive anchors before paying a GPU evening to discover them"
```

---

### Task 9: The rule, in the decision log, before any of it runs

**Files:**
- Modify: `docs/decisions.md`
- Modify: `docs/failures.md`

This task has no code and it is not optional. A keep/drop threshold committed after the first `metrics.json` arrives is not a threshold.

- [ ] **Step 1: Add the decision**

Append to `docs/decisions.md`, following the existing `## YYYY-MM-DD — title` format:

```markdown
## 2026-08-17 — What counts as a rung helping, decided before anything ran

**Decision.** A rung of the small-target ladder is kept if its best F1 across the reported score
thresholds, at its final epoch, exceeds the previous kept rung's by **more** than the range of
that same statistic over the previous rung's last four epochs. A gain exactly equal to that range
is a rejection. `ladder.py` applies it; `configs/ladder.yaml` names the rungs it is applied to.

**Why a rule rather than a reading.** The baseline did not converge — see the failure log — and
its precision at a fixed threshold moved by a factor of three between adjacent epochs of a single
run. The gain a better anchor size buys is plausibly two or three points of F1, which is smaller
than that. Under those conditions "this change helped" is a sentence that can be written about
almost any pair of numbers, and the only defence is to fix what would count as help before seeing
any of it.

**Why the band is measured rather than assumed.** It is the previous rung's own dispersion over
its last four epochs, so a configuration that settles buys a tighter test for the next change and
one that does not pays for it. Nothing here is a claim about how much noise there ought to be.

**Why strictly greater.** A gain that only reaches the noise the previous rung was already showing
is noise. One character, and it is the difference between a ladder and a narration.

**The fallback, decided now rather than when it is needed.** If the cosine decay does not settle
the run — R1's band over its last four epochs stays the same order as R0's — the statistic becomes
the median over the last four epochs rather than the final one, the band stays the range, and the
finding that a decaying rate did not settle this configuration is recorded in the failure log.
Deciding this in advance is what stops it being an escape hatch.

**What this ticket cannot do about the deeper problem.** One run per rung, one seed. A rung whose
gain clears the band could still be a lucky draw, and only repeated seeds would settle that. It
would double a thirteen-hour budget on a free tier, and it is recorded here as the honest limit of
what these five numbers support rather than papered over.
```

- [ ] **Step 2: Add the entry the polarisation criterion needs**

Append to `docs/failures.md`:

```markdown
## 2026-08-17 — The dual-polarisation stem has no data, on either side of the chain

**What was asked for.** Issue #11's first acceptance criterion is an input stage adapted to radar
polarisation channels, and `model.py` sharpened it before the work began: "a dual-polarisation
stem trained as one".

**Why it is not here.** There is no second polarisation anywhere in this project. LS-SSDD-v1.0 is
VV, all 9000 sub-images of it. The scene the chain runs on is VV, and `configs/kattegat-lane.yaml`
records why: Earth Engine answers a single download up to 48 MiB, the box came back at 57 MB in VV
and VH, and the area was the one thing that had been measured and argued for, so the polarisation
was what gave way. Building a dual stem now would mean fitting its second channel on a copy of the
first, which is not an adaptation — it is the null adaptation the repository already ships.

**What was done instead.** A single-channel stem: `conv1` takes one channel of radar amplitude
rather than three copies of it, its weights folded down from the pretrained RGB kernels so that
the model is numerically identical to the repeat at initialisation. That is an input stage adapted
to what the data actually is, and it is measured as rung 3 of the ladder.

**What it would take to do the thing that was asked.** A VV+VH export, which means either a
smaller study area or a coarser resolution — both already argued over once — and a dual-polarisation
training set, which means either finding one at Sentinel-1's resolution or accepting a set whose
physics is not this chain's physics. Both are decisions above this ticket's level. Recorded here
so that the day a dual-polarisation export exists, this is a task rather than a rediscovery.
```

- [ ] **Step 3: Commit, and check the ordering**

```bash
git add docs/decisions.md docs/failures.md
git commit -m "docs: fix what counts as a rung helping, before there is a number to look at"
```

Verify the commit predates every metrics file — this is the guarantee the rule rests on:

```bash
git log --oneline -1 -- docs/decisions.md && git log --oneline -- docs/runs/
```

Expected: the decisions commit exists and `docs/runs/` has no commits yet.

---

### Task 10: The Kaggle runbook

**Files:**
- Create: `docs/superpowers/plans/2026-08-17-ladder-runbook.md`

**Run by:** a human, in Kaggle sessions. Not an agent task. Tasks 1–9 do not wait on it.

- [ ] **Step 1: Write the runbook**

Create `docs/superpowers/plans/2026-08-17-ladder-runbook.md`:

```markdown
# The ladder, session by session

Five GPU runs of twelve epochs, about 2.6 hours each on a T4, plus one CPU session for the census.
Thirteen hours against a free tier's thirty a week, which leaves room for one session to be lost.

Every run is `darkvessel train --config <rung>`, interrupted and restarted as many times as the
provider requires — that is what the loop is built for. Kaggle's *Save Version* re-executes the
whole notebook in a fresh machine, so the artefact you download is a **second run** of the same
code, not the session you watched. Take the metrics from the saved version, not from the console.

## Session 0 — the census, on CPU

Attach `ls-ssdd-v10`. No GPU, no internet needed.

    !pip install -e '.[detector]'
    !python3 notebooks/anchor_census.py

Paste its output into `docs/decisions.md` as the evidence for rung 2's anchor sizes and rung 4's
sampler value, and set `rpn_batch_size_per_image` in `configs/ladder/r4-sampler.yaml` from the
realised positive fraction it reports. If it contradicts the prediction written in the script's
docstring, record the contradiction rather than the prediction.

## Sessions 1 to 5 — the rungs, in order

| Session | Config | Metrics land as |
| --- | --- | --- |
| 1 | `configs/train.yaml` | `docs/runs/r0-baseline.json` |
| 2 | `configs/ladder/r1-cosine.yaml` | `docs/runs/r1-cosine.json` |
| 3 | `configs/ladder/r2-anchors.yaml` | `docs/runs/r2-anchors.json` |
| 4 | `configs/ladder/r3-stem.yaml` | `docs/runs/r3-stem.json` |
| 5 | `configs/ladder/r4-sampler.yaml` | `docs/runs/r4-sampler.json` |

Each rung writes to its own `checkpoints-rN` directory, so a rung cannot resume the previous
rung's finished schedule and report its numbers as its own.

After each one, bring the metrics file back into the repository under the name above and run:

    darkvessel compare --config configs/ladder.yaml

Then commit the metrics file together with the entry the verdict calls for — a keep into
`docs/decisions.md`, a rejection into `docs/failures.md` with its numbers.

**If a rung is rejected**, repoint the next rung's `extends:` at the last rung that was kept, and
commit that edit with the rejection. The ladder is greedy and the file has to say so.

## When the five are in

`README.md` gains a section after "The first run — 2026-08-14" holding the table
`darkvessel compare` prints, and `docs/decisions.md` gains the reasoning for the anchor sizes and
the pyramid levels that criterion 2 asks for. Then the five acceptance criteria on issue #11 can
be ticked, or the ones that were not met explained where they were not.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-08-17-ladder-runbook.md
git commit -m "docs: the five sessions, in the order they have to happen"
```

---

## Self-Review

**Spec coverage.** Every section of the design maps to a task: `extends` → 1; the run's identity →
2; the ladder, its refusals and the rule → 3; the `compare` command and `configs/ladder.yaml` → 4;
the single-channel stem and the sampler knobs → 5; the cosine schedule and its resume → 6; the
config keys and the four rung files → 7; the census → 8; the rule committed before the runs and
the dual-polarisation entry → 9; the division of labour → 10.

**Two things the spec named that the plan places explicitly.** The spec said rungs must not share a
working directory only by implication; Task 7 tests it, because a rung inheriting the previous
rung's checkpoint directory would find a finished schedule, do nothing, and report the previous
rung's numbers as its own. And the spec did not say where the stem fold goes relative to the head;
Task 5 does, because building a `Conv2d` draws from the seeded generator and folding too early
would give the two stems different heads — which is the exact failure `docs/failures.md` already
records once.

**Out of scope, as the spec states.** Re-running the chain on the Kattegat scene, any change to
another pipeline stage, and trimming the coarse pyramid levels unless the census says they match
nothing and the budget survives.
