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
realised positive fraction it reports — the value shipped there now, `32`, is explicitly marked
provisional pending this run. If the census contradicts the prediction in the script's own
docstring (a realised positive fraction near 1%, not 50%), record the contradiction rather than
the prediction: this project has already reported a prediction that turned out wrong once, in the
first training run, and said so in the README rather than quietly fixing the number.

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

**Adapting `notebooks/kaggle-train.ipynb`.** The notebook's last cell names `configs/train.yaml`
explicitly, and its resume cell (cell 3) hardcodes `/kaggle/working/checkpoints` and
`metrics.json` because those are R0's own paths — the ones in the table's first row. For sessions
2 to 5, change the last cell's `--config` to that session's ladder config, and if a rung needs a
second session to finish its twelve epochs, change cell 3's glob and filenames to that rung's own
`checkpoints-rN` and `metrics-rN-*.json` too. Left pointing at R0's names, cell 3 finds nothing
under `*/checkpoints/epoch-*.pt`, reports no checkpoint attached, and the rung restarts from epoch
0 rather than resuming — silently, and at the cost of the epochs already paid for.

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
