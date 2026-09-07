# The ladder, session by session

Five GPU runs of twelve epochs, about 2.6 hours each on a T4, plus one CPU session for the census.
Thirteen hours against a free tier's thirty a week, which leaves room for one session to be lost.

Every run is `darkvessel train --config <rung>`, interrupted and restarted as many times as the
provider requires — that is what the loop is built for. On Kaggle the notebook calls that from
Python — `subprocess` on `[sys.executable, "-m", "darkvessel", "train", ...]` — and not from a
`!` shell line, because `pip install -e` puts the console script somewhere the session's shell
does not look, and a `!` line's `{...}` substitution belongs to the frontend rather than to
Python and is not performed on that image. Same command, reached without either.

The notebook also puts the clone's `src/` on `sys.path`, and passes it to that subprocess as
`PYTHONPATH`, rather than trusting the install to have made the package importable: on
2026-08-23 the clone succeeded, `pip install -e` ran, and `import darkvessel` still raised
`ModuleNotFoundError`. Where an install lands is a property of the machine; where the clone is,
is not. Kaggle's *Save Version* re-executes the
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
| 1 ✅ | `configs/train.yaml` | `/kaggle/working/checkpoints` | `/kaggle/working/metrics.json` | `docs/runs/r0-baseline.json` |
| 2 ✅ | `configs/ladder/r1-cosine.yaml` | `/kaggle/working/checkpoints-r1` | `/kaggle/working/metrics-r1-cosine.json` | `docs/runs/r1-cosine.json` |
| 3 ❌ | `configs/ladder/r2-anchors.yaml` | `/kaggle/working/checkpoints-r2` | `/kaggle/working/metrics-r2-anchors.json` | `docs/runs/r2-anchors.json` |
| 4 ❌ | `configs/ladder/r3-stem.yaml` | `/kaggle/working/checkpoints-r3` | `/kaggle/working/metrics-r3-stem.json` | `docs/runs/r3-stem.json` |
| 5 ❌ | `configs/ladder/r4-sampler.yaml` | `/kaggle/working/checkpoints-r4` | `/kaggle/working/metrics-r4-sampler.json` | `docs/runs/r4-sampler.json` |

**Sessions 1 and 2 both ran on 2026-08-23.** R0 scored F1 **0.807** with a band of **0.026**,
putting R1's bar at 0.833. R1 scored **0.836** — kept, by 0.0021 — and collapsed the band to
**0.0099**, which is what the verdict actually rests on rather than the two-thousandth margin.
Because the rule measures the band on the rung being compared against, **R2 is kept only above
0.8454**, and rejected at or below it, the `>` being strict. A settling rung buys a tighter test
for the next one; this is that, arriving early. The numbers and the reasoning are in
`docs/decisions.md`, 2026-08-23. Three sessions remain, about eight GPU hours.

R1 being kept, `r2-anchors.yaml` keeps `extends: r1-cosine.yaml` — no edit. That inheritance is
now held by a test rather than by this sentence: `test_config.py` asserts every rung resolves to
`lr_schedule: cosine`, so a rung repointed at the baseline fails on a laptop rather than after a
GPU evening.

**Session 5 ran on 2026-08-25. R4 was rejected and the ladder is complete.** F1 **0.827**, so
−0.0087 against R1 — *smaller than R1's noise band of 0.0099*, which makes it a draw rather than a
harm, unlike R2's −0.048. All five runs are in `docs/runs/`; one change of five was kept, and it
was the one that is not among the ticket's three adaptations. What remains is the section below,
"When the five are in" — no GPU time is left to spend.

**Session 4 ran on 2026-08-24 and R3 was rejected too.** F1 **0.83556** against R1's
**0.83557** — a draw to five decimal places, and the near-null the folded stem's own design
predicted. Criterion 1 of issue #11 is answered by that number rather than evaded: the three
copies were not costing anything. `r4-sampler.yaml` is repointed at `r1-cosine.yaml` — its comment
named `r2-anchors.yaml` as the fallback, on the assumption that only one of R2 and R3 could fall,
and both did. **The bar for R4 is still 0.8454.** One session remains.

Read `docs/failures.md` before starting it: R4's stated reason for `rpn_batch_size_per_image: 32`
is measured under R2's small anchors, which are not on this branch, and under the stock anchors it
now inherits the sampler is not idle at all. The value stays at 32 — it was fixed before any run —
but what session 5 measures is not the question the config's comment poses.

**Session 3 ran on 2026-08-23 and R2 was rejected.** F1 **0.788** against the 0.8454 above, and
below R0's own 0.807 — the small anchors are worse than the stock ones outright. The anchor census
predicted this in writing on 2026-08-19, before any rung had run; the numbers and the mechanism
are in `docs/failures.md`. `r3-stem.yaml` is repointed at `r1-cosine.yaml` accordingly, so R3
stands on R1 and is measured against it. **The bar for R3 is unchanged at 0.8454** — a rejected
rung moves neither the standing statistic nor the band. Two sessions remain, about five GPU hours.

Each rung writes to its own `checkpoints-rN` directory, so a rung cannot resume the previous
rung's finished schedule and report its numbers as its own — the five sessions share one Kaggle
working directory, and only the config tells them apart.

**Adapting `notebooks/kaggle-train.ipynb`.** Re-import the notebook from the repository rather
than editing an old copy in a browser tab — sessions 2 to 5 need the corrections of 2026-08-23,
which a notebook imported before them does not have. Then turn **Internet on** before running
anything: without it the clone fails, nothing installs, and every cell below reports a
`ModuleNotFoundError` that is an echo of that one line rather than a fault of its own. A newly
imported notebook has Internet off by default, and toggling it restarts the session.

The resume cell sets one constant, `CONFIG`,
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

## Session 6 — the threshold sweep, on CPU. Done on 2026-08-30.

The ladder closed with five rungs and one of them kept. Issue #24 adds a sixth, and it is the one
the census's own reservation of 2026-08-19 named: the RPN's foreground IoU threshold, which both
rejected RPN rungs were defined by and neither tested.

Its value is not in any config yet, deliberately. It comes out of this session, the way rung 4's
`rpn_batch_size_per_image: 32` came out of session 0 — and this one costs no GPU quota either.
Same setup as session 0 above, same dataset, same three lines:

    !git clone -q https://github.com/esamoun/dark-vessel-detection.git /kaggle/working/repo
    !cd /kaggle/working/repo && pip install -e '.[detector]'
    !cd /kaggle/working/repo && python3 notebooks/anchor_census.py

The script now prints a fourth block after the two anchor-set censuses: every ship's best overlap
with any anchor, read at five percentiles, and a table of nine candidate foreground thresholds
from 0.7 down to 0.05 with the rescue-only share and the realised positive fraction at each. Only
the stock anchors are swept — the small set was rejected on 2026-08-23 and nothing will run on it.

Bring that table back, and the threshold is set from it in `docs/decisions.md` **before** the
session below is started. `docs/decisions.md`, 2026-08-29, holds a prediction about what the sweep
will say; if the sweep contradicts it, the contradiction is what gets written, not a narrowed
version of the prediction.

**It ran on 2026-08-30 and the threshold is set at 0.3.** The table is in `docs/decisions.md`,
2026-08-30, with the prediction judged there: half right — the median ship's best overlap is
**0.207**, below the 0.25 two entries of that log quote, but twice the 0.10 the prediction named.
The sweep reproduced the census of 2026-08-19 on every figure but one, and that one was a defect in
the sweep rather than in the census: see `docs/failures.md`, 2026-08-30.

## Session 7 — the sixth rung. Done on 2026-08-30. Rejected.

**F1 0.82282 against R1's 0.83557** — a loss of 0.0128 against a band of 0.0099, so outside it and
therefore a loss rather than R4's draw, though R5's own band of 0.0253 is twice the difference.
Precision was unchanged at 0.850 and the whole loss was recall; 72 of the 2378 held-out ships left
the detector's reach altogether. `docs/failures.md`, 2026-08-30, has the mechanism, which inverts
the census's reading of the rescue rule. **R1 remains the kept rung and the chain is unchanged.**

Two operational notes from this session, both paid for. The first attempt ran interactively
overnight, reached epoch 10, and was lost when the browser connection dropped — with Persistence
off, `/kaggle/working` went with it. Use **Save Version (Save & Run All)**, which runs headless and
does not care about the browser, and turn Persistence on regardless so an interrupted attempt is
resumable. The committed run was also faster than the interactive one: 4.93 it/s on the held-out
pass against 3.9, about 12 min 45 an epoch rather than 15.


`configs/ladder/r5-fg-iou.yaml` now exists, extends `r1-cosine.yaml` — the last rung kept — and
moves **one** key: `model.rpn_fg_iou_thresh` from 0.7 to 0.3. One and not two, because 0.3 is the
last value reachable without also moving `model.rpn_bg_iou_thresh`, `Matcher` refusing a background
threshold above the foreground one. Everything below 0.3 is a rung of its own and is deferred in
`docs/decisions.md`, 2026-08-30.

What the change buys, from the sweep: 1019 more ships gain a genuine match instead of a rescued one
(rescue-only 3257 → 2238 of 3637), and the positives the sampler draws roughly double, 43 per image
→ 96. The rung's own prediction is in that entry, written before this session: **not a draw** —
the statistic moves by more than R1's band of 0.0099, direction positive — with the risk that would
make it wrong named beside it.

| Session | Config | Kaggle writes checkpoints to | Kaggle writes metrics to | Commit the metrics file as |
| --- | --- | --- | --- | --- |
| 7 | `configs/ladder/r5-fg-iou.yaml` | `/kaggle/working/checkpoints-r5` | `/kaggle/working/metrics-r5-fg-iou.json` | `docs/runs/r5-fg-iou.json` |

**The bar is R1's, unchanged at 0.8454** — 0.83557 plus the 0.0099 band R1 was already showing.
R2, R3 and R4 were all rejected, so none of them moved either the standing statistic or the band,
and the rule is the same one fixed on 2026-08-17: kept only on a strict `>`.

Twelve epochs, about 2.6 hours on a T4. Then `darkvessel compare --config configs/ladder.yaml`,
and the entry the verdict calls for — `docs/decisions.md` if kept, `docs/failures.md` if not —
with the sweep's counts under the threshold actually run set against the census of 2026-08-19,
which is what criterion 4 of the ticket asks for.
