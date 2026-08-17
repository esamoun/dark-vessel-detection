# Adapting the detector for small targets and extreme imbalance

Design for issue #11, *Adapt the detector for small targets and extreme imbalance*.

Written in English, like the rest of `docs/`, so it sits beside `decisions.md` rather than beside
the conversation that produced it.

## What this ticket actually has to close

Five acceptance criteria, and only three of them are about the model:

1. The input stage adapted to radar polarisation channels.
2. Feature levels and anchor sizing chosen for small targets, with the reasoning in the decision
   log.
3. Foreground/background imbalance addressed at the loss.
4. Each change measured against the previous configuration on the same held-out split.
5. Changes that did not help recorded in the failure log.

The last two are the ones that decide whether this ticket is worth anything. Three architectural
changes with no measurement behind them is a weekend of guessing; the same three with a ladder of
comparisons behind them is the level. So the design below spends most of its argument on the
measurement and comparatively little on the architecture, which is the opposite of where the
issue text spends its words, and deliberately so.

## Where the ticket's text and the data disagree

**There is no VH.** The first criterion asks for an input stage taking radar polarisation
channels, and `model.py:22` sharpens that to "a dual-polarisation stem trained as one". Neither
half of the chain has a second polarisation to give it:

- LS-SSDD-v1.0 is VV only. Every one of its 9000 sub-images is single-channel.
- The scene the chain runs on is VV only. `configs/kattegat-lane.yaml:26-36` records why: Earth
  Engine answers a single download up to 48 MiB, this box came back at 57 MB in VV and VH, and
  the area was the one thing that had been measured and argued for. The polarisation is what gave
  way, and the comment says in as many words that VH would separate a ship from the sea better.

A dual-polarisation stem is therefore not deferred for lack of time. It has no data to be fitted
on and no data to be run on, and building one now would mean training a second channel on a copy
of the first — which is not an adaptation, it is the null adaptation the repository already
ships.

**What is delivered instead.** A genuine single-channel stem: `conv1` takes one channel of radar
amplitude rather than three copies of it, with its weights folded down from the pretrained RGB
kernels. That is an input stage adapted to what the data is — one polarisation of radar
amplitude, not three colours — and it is measured like every other rung. The dual-polarisation
stem goes into `failures.md` as blocked on data, with the path to build it stated, so that the
day a VV+VH export exists the work is a task rather than a rediscovery.

## The measurement problem, and why the ladder starts at the learning rate

`failures.md:260-286` already established that the baseline did not converge. Precision at a
fixed score threshold of 0.50, across the twelve epochs:

```
0.55  0.74  0.75  0.41  0.64  0.84  0.65  0.28  0.80  0.63  0.53  0.81
```

Adjacent epochs differ by a factor of three, and the same shape appeared under two different
initial weights, so it is a property of the configuration rather than of a draw. The diagnosis is
in the failure log: `learning_rate: 0.005`, constant, no decay anywhere. The model reaches the
neighbourhood of a minimum in about three epochs and bounces around it for nine more.

This is fatal to criterion 4 as written. The improvement a better anchor size buys is plausibly
two or three points of F1. The gap between epoch 8 and epoch 9 of the *same run* is larger than
that. Comparing the final epoch of configuration A against the final epoch of configuration B,
under a schedule that oscillates by more than the effect being measured, produces a number that
describes the draw and not the change — and it produces it with two decimal places and a table,
which is worse than producing nothing.

`failures.md:283-286` parked the schedule deliberately: "it changes what the numbers mean, so it
belongs to a run that can be compared against this one — and this baseline exists to be that
comparison". This ticket is that run. The learning-rate decay is the ladder's first rung, measured
against the baseline exactly like the three adaptations, and everything after it is compared
between converged runs.

## The baseline is not reproducible, so it is run again

The table published in `README.md:355-360` — precision 0.941 at recall 0.706, score threshold
0.75 — comes from a run whose detection head was drawn from an unseeded generator. The fix landed
afterwards, in `199ed3e fix: the seed names the weights too`, and it changed how the head is
initialised. Running `configs/train.yaml` unmodified today would not reproduce those numbers.

`failures.md:246-248` states the cost of exactly this: "a number in the README with no run behind
it that can be re-created is a number nobody can check — including the next ticket, which is
supposed to measure its changes against exactly these". This is the next ticket, and it declines
to build a five-rung ladder on a number it cannot re-create.

So rung 0 is `configs/train.yaml` run again, unchanged, under today's code. It costs one session
and it buys two things: a reproducible foot for the ladder, and the gap between it and the
published table, which is a measurement in its own right of what a change of head initialisation
was worth.

## The ladder

Five runs of twelve epochs. On a Kaggle T4 an epoch takes about thirteen minutes, so a run is
about 2.6 hours and the ladder is about thirteen — one free-tier week is thirty, which leaves room
for one session to be lost.

| Rung | What changes | Why here |
| --- | --- | --- |
| `R0` | nothing — `configs/train.yaml` as it stands | a reproducible foot, under the corrected seeding |
| `R1` | cosine decay of the learning rate | makes every later comparison legible |
| `R2` | `anchor_sizes` → `[[4],[8],[16],[32],[64]]` | the headline small-target change, already named by `decisions.md` 2026-08-14 |
| `R3` | single-channel stem | the input, after the anchors: better anchors change which anchors are positive |
| `R4` | the RPN sampler, at the value the census dictates | the imbalance, once it is known where it is |

**Cumulative, not one-factor-at-a-time.** Each rung starts from the last rung that was kept, which
is the issue text's own phrasing — "each change is measured separately against the previous
configuration". The cost is that the path is greedy: an adaptation that would have helped on top
of a different base is lost. The alternative, measuring all three against a fixed base, answers a
question about composition that the ticket does not ask, and needs a sixth run to end with an
actual detector rather than three isolated measurements.

**Why the anchors precede the stem, and both precede the sampler.** The anchor sizes decide which
anchors clear the RPN's foreground threshold, which decides how many positives exist to be
sampled, which is the entire subject of rung 4. Measuring the sampler before the anchors would
measure it against an anchor set already known to be the wrong range, and the answer would have
to be thrown away.

**Cosine, not steps, and no warmup.** Twelve epochs and a `StepLR` introduces two free parameters
— where the step falls and how far it drops — that would have to be justified from nothing.
`CosineAnnealingLR` over the schedule has none. Warmup is the other obvious addition and is
deliberately not in rung 1: it is a third knob, and it becomes a candidate only if `R1` shows
instability inside its first three epochs, in which case it is its own rung with its own
measurement.

## The rule for keeping a rung, written before any of them runs

**The statistic** is the best F1 across the reported score thresholds, at the final epoch. The
journal records a precision and a recall at each threshold and no F1, so `ladder.py` derives it as
their harmonic mean and the derivation lives in one place.

**The noise band** is the range — maximum minus minimum — of that same statistic over the last
four epochs of the previous rung.

**The rule.** A rung is kept if its statistic exceeds the previous rung's by more than the noise
band. A rung that does not clear it is rejected, recorded in `failures.md` with its numbers, and
the next rung starts from the last rung that was kept. A gain exactly equal to the band is a
rejection, not a keep.

This entry is written into `decisions.md` and committed **before the first session runs**. A
threshold chosen after seeing the numbers is not a threshold, it is a narration of them, and it is
the quietest way there is to manufacture a result out of noise.

**The fallback, also decided now.** If `R1` does not stabilise the run — its band over the last
four epochs stays the same order as `R0`'s — the statistic becomes the median over the last four
epochs rather than the final one, the band stays the range, and the finding that a cosine decay
did not settle this configuration becomes an entry in the failure log. Deciding the fallback in
advance is what stops it from being an escape hatch.

## The census, before any GPU hour is spent

`notebooks/anchor_census.py`, run in a Kaggle **CPU** session — it needs the LS-SSDD annotations,
which live on `/kaggle/input` and which this project keeps off the local disk. Minutes, and no
GPU quota.

It runs torchvision's own `AnchorGenerator` and the RPN's `Matcher` over the real boxes of the
training split, and reports, per pyramid level:

- how many anchors clear the foreground IoU threshold of 0.7, under the stock sizes and under the
  candidate `[[4],[8],[16],[32],[64]]`;
- how many ground-truth boxes are matched only by `allow_low_quality_matches`, which is
  torchvision guaranteeing every box at least one anchor no matter how poor;
- the realised positive fraction against the 128 that `rpn_batch_size_per_image: 256` with
  `rpn_positive_fraction: 0.5` asks for;
- the distribution of ship box sizes in pixels.

**The prediction, written before it runs**, in the same spirit as the recall prediction the README
records and then reports wrong. With about three ships to a ship-bearing tile, the RPN will find a
handful of positives and fill the remaining ~250 slots with background: a realised positive
fraction near 1%, not 50%. If that holds, `rpn_positive_fraction` is not the lever — it is a
**ceiling**, not a target, and torchvision samples `min(available, requested)`. The lever is
`rpn_batch_size_per_image`, moved *down*, so that the few positives are not drowned. Rung 4 moves
whichever knob the census identifies, and if the census contradicts the prediction it is recorded
as contradicted.

The census also supplies the evidence criterion 2 asks for. "Feature levels chosen for small
targets, with the reasoning in the decision log" is satisfied by per-level positive-anchor counts
— which levels ever match a ship, and which never do — rather than by an argument from the stride
arithmetic. If it shows that the coarse levels match nothing across 3637 ships, trimming them is a
named candidate for a later rung and is recorded in `decisions.md` as reasoned and deferred. It
does not join rung 2, because a rung that changes two things measures neither.

## The single-channel stem, identical at initialisation

`detector_model` gains `stem: "repeat" | "single"`. The single-channel path is constructed so that
the model at initialisation is numerically identical to the three-channel repeat, and that is
load-bearing rather than decorative: it is what makes rung 3 a measurement of one thing.

The current path repeats `x` across three channels, the transform normalises per channel, and
`conv1` sums:

```
y = Σ_c W_c · (x − m_c) / s_c
```

A one-channel `conv1` with

```
W'[k, 0, i, j] = Σ_c W[k, c, i, j] / s_c
b'[k]          = −Σ_c (m_c / s_c) · Σ_ij W[k, c, i, j]
```

and the transform set to `image_mean=[0.0], image_std=[1.0]` produces the same `y` exactly.

The bias matters here and would not in a stock ResNet. With `trainable_backbone_layers: 3`, `bn1`
is a `FrozenBatchNorm2d` applying fixed statistics rather than recentring the batch, so a constant
offset propagates instead of being absorbed. Dropping `b'` would leave the two paths differing by
a constant through the whole backbone.

What rung 3 therefore measures is not a different starting point. It is what training does with
one bank of 7×7×1 kernels instead of three, which is the question, and nothing else. A CPU test
asserts the equality on random input.

The stem goes into the `built` block, so it travels in the checkpoint, so `trained.py` and the
guard in `checkpoints.py` follow it with no new code — the refusal written for anchor sizes covers
the stem for free.

## What the code changes

| File | Responsibility |
| --- | --- |
| `src/darkvessel/detect/model.py` | `stem`, the folded single-channel `conv1`, the sampler knobs passed through to `FasterRCNN` |
| `src/darkvessel/detect/train.py` | the scheduler, constructed beside the optimiser and stepped per epoch; its state into the checkpoint |
| `src/darkvessel/detect/checkpoints.py` | `Journal` gains a run header; a bare list still reads back |
| `src/darkvessel/detect/ladder.py` | **new.** torch-free. Reads the rungs, refuses mismatched ones, emits the table, applies the rule |
| `src/darkvessel/cli.py` | `extends:` in config loading; a `compare` subcommand |
| `configs/train.yaml` | `model.stem`, the four sampler keys, `schedule.lr_schedule` |
| `configs/ladder/r1-cosine.yaml` … `r4-sampler.yaml` | **new.** one rung each, minimal, over `extends: ../train.yaml`. `R0` has no file of its own: it *is* `configs/train.yaml` |
| `configs/ladder.yaml` | **new.** the rungs, their metrics files, and the line each one changed |
| `notebooks/anchor_census.py` | **new.** the census |
| `docs/runs/r0-baseline.json` … `r4-sampler.json` | **new.** the metrics of each rung, committed |
| `tests/test_ladder.py`, `tests/test_model_stem.py` | **new.** |

**The scheduler and resume.** Its state is saved beside the optimiser's. Without that, a session
killed at epoch 7 restarts the schedule from the top and the resumed run is no longer the same
experiment as the interrupted one — which is the single property `train.py` was built to
guarantee, and the one thing a learning-rate schedule is most likely to break quietly.

**`extends:` in config loading.** Each rung differs from the one before by a single line. Five
standalone ninety-line files would make that line invisible in a diff and would let a second line
drift without anyone noticing, which breaks the ladder in exactly the way the ladder exists to
prevent. A rung config is therefore ten lines over `extends: ../train.yaml`. It resolves the base
relative to the file that declares it — the rule this repository already applies to every path in
every config — refuses a cycle, and refuses a missing base.

**The run's identity, and the refusals.** `metrics.json` today is a bare list of per-epoch
entries: epoch, training loss, held-out counts, the table by threshold. Nothing in it names the
run. Five such files are five anonymous tables, and nothing stops a comparison between a run
scored at a 200 m tolerance and one scored at 300 m.

The journal becomes `{"run": {...}, "epochs": [...]}`, where `run` carries the existing `built`
block plus the learning rate and its schedule, the subset ratio, the tolerance and the thresholds.
A bare list is still read back as epochs, so a session already in flight is not stranded by a
format change, but `ladder.py` refuses it as a rung: a run that does not name itself compares to
nothing.

`ladder.py` compares two rungs only if they agree on `held_out_tiles`, `held_out_ships`, the
tolerance and the thresholds, and its refusal names the field that differs. This is the one place
in the ticket where a silent error would produce a published claim that is false.

## Testing

**In CI, without torch.**

- `test_ladder.py` — the table; the three refusals; the rule at its boundary, where a gain exactly
  equal to the band is rejected; a bare-list journal refused as a rung.
- `test_checkpoints.py` — the new journal shape round-trips; a bare list still reads back for
  resume.
- `test_pipeline.py` — `extends` resolves the base plus its single override, refuses a cycle,
  refuses a missing base, and resolves relative to the declaring file.

**With torch, skipped in CI, run on the laptop.**

- `test_model_stem.py` — the two stems produce the same output at initialisation; `built` carries
  the stem; a checkpoint built with one stem is refused by the other.
- `test_training_run.py`, one test added — a session killed after epoch 1 under cosine, resumed,
  and the learning rate at epoch 2 equal to what an uninterrupted run would have had.

No test pins a precision, a recall or a detection count. That constraint is inherited from the
swap ticket and holds here for a stronger reason: this ticket's whole output is numbers that move.
`ladder.py` is tested against synthetic journals.

## Who runs what

**Agent, on the laptop, before any Kaggle session.** Every code change above, the rung configs,
the tests, the census script, and the `decisions.md` entry fixing the keep/drop rule. That entry
is committed before the first run.

**Human, on Kaggle.** The census in a CPU session, then the five runs in order, bringing each
`metrics.json` back into the repository under `docs/runs/`.

**Agent, after each run.** `darkvessel compare --config configs/ladder.yaml`, the rule applied,
and the entry written — a keep into `decisions.md`, a rejection into `failures.md` with its
numbers.

## Out of scope

**Re-running the chain on the Kattegat scene with the winning model.** Tempting, since the swap is
already done and the config exists. It is declined because it would produce a fresh claim about a
scene whose azimuth-displacement bias is documented in the failure log, and because issue #12 —
*Evaluation report and failure analysis* — exists for exactly that. This ticket closes on the
LS-SSDD held-out split, which is what all five of its criteria ask about.

**Any change to another stage of the pipeline.** Same constraint the swap ticket worked under.

**Trimming the coarse pyramid levels**, unless the census says they match nothing and the budget
survives the five rungs. Reasoned in `decisions.md` either way.

## What would make this ticket fail

Not a rung that does not help — that outcome is provided for, has a rule, and has a place to be
written down. The failure modes are:

- **A comparison between runs that were not scored the same way.** Guarded by `ladder.py`'s
  refusals, which is why they are tested rather than trusted.
- **A keep/drop threshold adjusted after the numbers arrive.** Guarded by committing the rule
  first, in a commit that predates the first `metrics.json`.
- **A resumed session that is not the run it resumed**, because the scheduler restarted. Guarded
  by the resume test.
- **Rung 3 measuring an initialisation rather than a stem.** Guarded by the equality test.
