# The ladder, session by session

Five GPU runs of twelve epochs, about 2.6 hours each on a T4, plus one CPU session for the census.
Thirteen hours against a free tier's thirty a week, which leaves room for one session to be lost.

Every run is `darkvessel train --config <rung>`, interrupted and restarted as many times as the
provider requires — that is what the loop is built for. Kaggle's *Save Version* re-executes the
whole notebook in a fresh machine, so the artefact you download is a **second run** of the same
code, not the session you watched. Take the metrics from the saved version, not from the console.

## Session 0 — the census, on CPU. Done on 2026-08-20.

Its numbers, what they settled and what they contradicted are in `docs/decisions.md`. Rung 4's
`rpn_batch_size_per_image` is fixed at 32 as a result and is no longer provisional. Kept here
because every session below starts the same way, and because a rerun is how you check the numbers
rather than trust them.

Attach the dataset **`petrarodriguez/ls-ssdd-v1-0`**. Note two things about it. Kaggle now mounts
inputs at `/kaggle/input/datasets/<owner>/<slug>` rather than `/kaggle/input/<slug>`, so the path
in any older note here is dead. And this mirror is not laid out the way LS-SSDD was published: its
images are already split into `JPEGImages_sub_train` and `JPEGImages_sub_test`, each doubly
nested, with all 9000 annotations in one directory. That is why `data.images` in
`configs/train.yaml` names two directories — the held-out scenes, 11 to 15, live in the second,
and naming only the first gives an empty held-out split that nothing would report.

No GPU, no internet needed:

    !git clone -q https://github.com/esamoun/dark-vessel-detection.git /kaggle/working/repo
    !cd /kaggle/working/repo && pip install -e '.[detector]'
    !cd /kaggle/working/repo && python3 notebooks/anchor_census.py

This CPU session does not load the training notebook, so confirm the dataset reads as it should
before trusting the census's numbers. In a Python cell — not a `!` shell one:

```python
import pathlib, sys
sys.path.insert(0, "/kaggle/working/repo/src")
from darkvessel.config import load_config
from darkvessel.cli import training_request_from
from darkvessel.detect.dataset import catalogue, split_by_scene

configs = pathlib.Path("/kaggle/working/repo/configs")
request = training_request_from(load_config(configs / "train.yaml"), configs)
training, held_out = split_by_scene(catalogue(request["root"], request["layout"]))
print(len(training), len(held_out))
```

It must print `6000 3000`. A held-out count of zero means the mirror moved again; stop rather than
train against a split that will be scored over nothing.

**Sessions 1 to 5 do not paste this.** `notebooks/kaggle-train.ipynb` carries the same check as a
cell of its own, between the resume and the training call, and it asserts rather than prints — a
zero there stops the notebook instead of scrolling past a Run All. It was prose here first, and
prose an operator has to retype is a step that gets skipped or mistyped on the evening it matters.

## Sessions 1 to 5 — the rungs, in order

Each config's own `out:` block, read directly out of the four files in `configs/ladder/` and out
of `configs/train.yaml`, against the paths `configs/ladder.yaml` reads each rung from. These two
sides do not meet on their own — a Kaggle session's `/kaggle/working` evaporates when the session
ends, and `configs/ladder.yaml`'s `metrics:` lines point into `../docs/runs/`, inside the
repository. Nothing copies one to the other; that is this table.

| Session | Config | Kaggle writes checkpoints to | Kaggle writes metrics to | Commit the metrics file as |
| --- | --- | --- | --- | --- |
| 1 | `configs/train.yaml` | `/kaggle/working/checkpoints` | `/kaggle/working/metrics.json` | `docs/runs/r0-baseline.json` |
| 2 | `configs/ladder/r1-cosine.yaml` | `/kaggle/working/checkpoints-r1` | `/kaggle/working/metrics-r1-cosine.json` | `docs/runs/r1-cosine.json` |
| 3 | `configs/ladder/r2-anchors.yaml` | `/kaggle/working/checkpoints-r2` | `/kaggle/working/metrics-r2-anchors.json` | `docs/runs/r2-anchors.json` |
| 4 | `configs/ladder/r3-stem.yaml` | `/kaggle/working/checkpoints-r3` | `/kaggle/working/metrics-r3-stem.json` | `docs/runs/r3-stem.json` |
| 5 | `configs/ladder/r4-sampler.yaml` | `/kaggle/working/checkpoints-r4` | `/kaggle/working/metrics-r4-sampler.json` | `docs/runs/r4-sampler.json` |

Each rung writes to its own `checkpoints-rN` directory, so a rung cannot resume the previous
rung's finished schedule and report its numbers as its own — the five sessions share one Kaggle
working directory, and only the config tells them apart.

**Adapting `notebooks/kaggle-train.ipynb`.** The resume cell (cell 3) sets one constant, `CONFIG`,
and derives the checkpoint directory, the glob and the metrics filename from it via
`load_config`; the training cell reads the same constant. For sessions 2 to 5, set `CONFIG` to
that session's ladder config and nothing else — there is no second edit left to miss. If a rung
needs a second session to finish its twelve epochs, `CONFIG` unchanged is the whole of what a
resume needs: the glob is already scoped to that rung's own `checkpoints-rN` directory, so it
cannot pick up a checkpoint left over from a different rung's attached output. This replaced an
earlier two-edit procedure — the training cell's `--config` and the resume cell's hardcoded
`/kaggle/working/checkpoints` and `metrics.json` had to be changed together, and missing the
second one found nothing under `*/checkpoints/epoch-*.pt`, reported no checkpoint attached, and
restarted the rung from epoch 0 rather than resuming — silently, and at the cost of the epochs
already paid for.

After each session, bring the metrics file back into the repository under the name the table
gives, and run:

    darkvessel compare --config configs/ladder.yaml

Then commit the metrics file together with the entry the verdict calls for — a keep into
`docs/decisions.md`, a rejection into `docs/failures.md` with its numbers.

**If a rung is rejected**, repoint the next rung's `extends:` at the last rung that was kept, and
commit that edit with the rejection. The ladder is greedy and the file has to say so.

## When the five are in

`README.md` gains a section after "The first run — 2026-08-14" holding the table
`darkvessel compare` prints, and `docs/decisions.md` gains the reasoning for the anchor sizes and
the pyramid levels that criterion 2 of issue #11 asks for. Then the five acceptance criteria on
issue #11 can be ticked, or the ones that were not met explained where they were not — criterion 1
is one of those: `docs/failures.md` already records the dual-polarisation stem asked for as
blocked on data, with the single-channel stem shipped as rung 3 in its place.
