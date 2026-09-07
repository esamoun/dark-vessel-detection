# Swapping the trained detector into the chain

Design for issue #10, *Swap the trained detector into the chain — Level 2 publishable*.

Written in English, like the rest of `docs/`, so it sits beside `decisions.md` rather than beside
the conversation that produced it.

## What this ticket actually has to close

The ticket asks for one thing: the trained model satisfies the same `detector` parameter the
threshold stand-in satisfies, and nothing else in the pipeline changes. That is the seam paying
for itself.

Three incompatibilities stand between the checkpoint and the chain. Only the first is named in
the issue text; the second is named in `configs/train.yaml` as "the next ticket's problem"; the
third is named nowhere and was found by reading `scene.py` against `model.py`.

1. **Unit.** The model was fitted on 8-bit amplitude divided by 255 (`dataset.py:164`). The chain
   exports Sentinel-1 GRD in decibels, where this scene's sea sits at −21.84 dB. `decisions.md`
   already records the gap and assigns its resolution here.
2. **Tile size.** The model is built with `min_size = max_size = 800` (`model.py:101`); the chain
   cuts at 512. Left alone, torchvision resamples every tile from 512 to 800 — and this project
   refuses to resample radar amplitude elsewhere, in as many words (`cli.py:434`).
3. **Holes.** `Scene._amplitude` turns nodata into NaN precisely because every comparison against
   NaN is false, which immunises a threshold detector. A convolutional network has no such
   immunity: one NaN propagates through the FPN and poisons the whole tile. On
   `data/real/kattegat-lane.tif` that is 189 731 pixels, 6.0% of the scene, and the failure is
   silent — the affected tiles simply return nothing.

## The measured facts this design rests on

`data/real/kattegat-lane.tif`, the acquisition the chain already runs on:

| | |
| --- | --- |
| Size, dtype | 1727 × 1845 px, float64, decibels |
| nodata | `0.0`, 189 731 px (6.0%) |
| Sea | median −21.84 dB, robust σ (MAD × 1.4826) 2.30 dB |
| Tail | p99 −17.25 dB, p99.99 +2.13 dB, max +27.08 dB |
| Above 0 dB | 438 px |

The threshold baseline, already on disk at `outputs/kattegat-lane.gpkg`: **16 detections, 2
matched, 14 dark, against 12 declared vessels** — and 9 of the 16 stand inside one 200 m square
in the north-west, which is one bright object counted nine times.

The trained detector, from `metrics.json` of the final epoch, over the entire held-out split:
precision 0.941, recall 0.706 at a score threshold of 0.75.

## Decision: the mapping is derived by measurement and shipped as a constant

The stretch LS-SSDD's authors applied is not recorded and cannot be recovered from the JPEGs. So
the mapping has to be *chosen*. It is chosen by matching the sea: the affine from decibels to
0..1 is fixed so that this scene's sea lands where the sea the model was actually fitted on sat.
Two moments, two parameters.

**The matching happens once, and its answer is frozen into the config.** This is the load-bearing
half of the decision. An affine refitted per scene would be a percentile stretch wearing a
different hat: the same hull would take a different value under a different sea state, and a
score threshold of 0.75 would stop meaning the same thing from one acquisition to the next. What
ships is a fixed window in decibels — two numbers a reader can interpret directly — whose
provenance is a measurement recorded in `decisions.md`.

The arithmetic, given a reference sea of mean `μ_ref` and standard deviation `σ_ref` in 0..1
units, and this scene's sea at median `m` with robust spread `s`:

```
span    = s / σ_ref
floor   = m − μ_ref · span
ceiling = floor + span
```

What the measurement buys is visible before it is taken. Under three plausible references:

| Assumed LS-SSDD sea | Resulting window |
| --- | --- |
| 0.10 ± 0.03 | −29.5 … +47.1 dB |
| 0.15 ± 0.05 | −28.7 … +17.2 dB |
| 0.25 ± 0.08 | −29.0 … −0.3 dB |

The floor lands near −29 dB whichever reference is assumed, because it is the sea less a few
sigma and the data settles it alone. The ceiling ranges over 47 dB. Choosing the window by eye
would have got the floor right and the ceiling anywhere, and the ceiling is what decides whether
a hull stands out or saturates.

The same measurement settles a second question the arithmetic above quietly assumes: whether the
affine belongs on decibels or on linear amplitude. The shape of the LS-SSDD sea histogram answers
it — roughly symmetric about a mid-grey means the source stretch was log-like and the affine
belongs on decibels; crushed against zero with a long tail means it was linear, and the mapping
inverts to σ⁰ first. The design ships whichever the histogram supports, and records the other as
rejected.

## Decision: nodata is filled at sea level, and guarded downstream

A hole gets the value the sea gets, so the contrast at its boundary is near zero. The alternative
— flooring holes at 0.0 — is simpler to state but paints 6% of the scene as a perfectly black
patch with hard edges, and a hard edge is a strong feature for a detector. The risk moves from
the hole to its outline rather than going away.

Filling is not sufficient on its own, so a second, independent mechanism sits behind it: any
detection whose centre falls inside a hole is discarded. The fill prevents the boundary artefact;
the guard makes it impossible for a hole to be reported as a target whatever the fill does. Each
is testable alone.

Synthetic speckle was considered and rejected. It would put the fill fully inside the training
distribution, but it invents amplitude — which this project already refuses on the augmentation
side, where a contrast jitter is described as producing a ship made of a different material.

## Decision: the chain cuts at 800, not the model at 512

The chain's tiling moves to `size_px: 800, overlap_px: 64`, giving 3 × 3 = 9 tiles over this
scene instead of 16 — and `Tiling` returns all nine at exactly 800 × 800, with no short tile at
the far edge, which is the condition that makes this work at all: a 736 px edge tile would be
resized by the very transform this decision exists to keep idle. Nothing is resampled and the
model runs at exactly the scale it was scored at, so the precision and recall in the README still
describe the configuration that runs.

Building the inference model at 512 instead would also resample nothing — the network is fully
convolutional and its anchors are in input pixels, so hulls keep their native size. It was
rejected for a narrower reason: the chain would then run at a scale the model has never been
scored at, and this ticket exists to measure the model's contribution rather than assume it.
Introducing an unmeasured variable into exactly that comparison is the one thing it cannot
afford.

Tiling is a property of a run, not a stage of the pipeline, so the ticket's "no other stage is
modified" holds.

## Decision: the checkpoint says what built it

`torch.save` currently writes `{"epoch", "model", "optimiser"}` and nothing else
(`train.py:113`). `AnchorGenerator` holds no parameters, and `min_size`/`max_size` are transform
attributes rather than weights — so loading `epoch-012.pt` into a model built with different
anchor sizes succeeds without a word and yields a silently wrong detector. It is the failure mode
`decisions.md` describes for the unit gap, in a second place: no crash, no warning, plausible
detections.

Two changes close it. `train.py` writes its build block — `anchor_sizes`, `tile_px`, `seed` —
into the checkpoint, which costs one line and removes the problem for every future run.
`TrainedDetector` reads that block where it exists and refuses a disagreement with the config;
for `epoch-012.pt`, which predates it, the run config restates the values with a comment naming
`configs/train.yaml` and seed `20260814`.

The first of these touches the training side rather than the pipeline. The ticket's constraint is
about pipeline stages, so it is in scope, and it is recorded here rather than discovered in
review.

## Components

### `src/darkvessel/detect/amplitude.py` — new, no torch

Everything about the conversion that can be got wrong quietly, on the side of the seam a laptop
tests in a second.

```python
@dataclass(frozen=True)
class DecibelStretch:
    floor_db: float      # maps to 0.0
    ceiling_db: float    # maps to 1.0
    sea_db: float        # what a hole receives, before the stretch

    def __call__(self, image: np.ndarray) -> np.ndarray: ...
```

NaN is filled with `sea_db` *before* clipping, so the fill follows the window automatically and
two numbers cannot drift apart. The module also holds `fit_window(...)`, which solves the
arithmetic above — used to derive the shipped constants, not called on any run.

### `src/darkvessel/detect/trained.py` — new, imports torch

`TrainedDetector` satisfies the `Detector` protocol. Per tile: stretch, `as_model_input`, model,
filter by score threshold, boxes to centres. It holds the hole guard, because it is the one place
that sees the original NaN tile alongside the model's output.

torch is imported the way `cli._train` imports it — inside the branch that needs it — so
`darkvessel run` with the stand-in still installs and runs with no framework, no GPU and no
network. That is the chain's acceptance condition and it does not move.

### `src/darkvessel/detect/model.py` — `_detections_from` moves here

It currently lives in `train.py:294` and does exactly the box-to-`PixelDetection` conversion
inference needs. It moves next to `as_model_input`, the other half of the tensor boundary, and
`train.py` imports it. No behaviour changes; the half-pixel conversion keeps one implementation
rather than gaining a second.

### `src/darkvessel/cli.py`

`_detector_from` gains a `"trained"` branch reading the checkpoint path, the score threshold, the
stretch constants and the build block. A refusal mirroring `_check_working_crs` rejects a run
where `tiling.size_px` disagrees with the model's `tile_px`, naming both numbers, rather than
letting torchvision resample.

### `configs/kattegat-lane.yaml`

`tiling` moves to 800/64. The `run` block names the trained detector, its checkpoint, its
operating point and its stretch, each with the comment that says where the number came from.

The operating point ships at **0.75**, a config key rather than a constant. It is the row of the
held-out table where precision is 0.941 against a recall of 0.706, and it is the right default
for what this chain does with an answer: every detection that is not matched against AIS becomes
a dark vessel, and a dark vessel is a claim that someone may be asked to send a boat at. Recall
costs a miss; precision costs a false accusation and a wasted inspection. Whoever pays for the
inspections can move the key, which is why the training run reports the whole table instead of
one number.

## What is refused rather than guessed

| Situation | Reaction |
| --- | --- |
| `tiling.size_px` ≠ the model's `tile_px` | refuse, naming both numbers |
| Checkpoint build block contradicts the config | refuse |
| Scene is not in decibels (8-bit dtype) | refuse, mirroring `dataset.py:154` |
| Detection centred inside a nodata hole | discarded, and counted in the report |

## The Kaggle pass

One session, four things:

1. Bring down `epoch-012.pt`, about 330 MB. `keep: 2` left only epochs 11 and 12; epoch 9, which
   scored better (F1 0.817 against 0.807), is already deleted. Epoch 12 it is.
2. Measure the LS-SSDD sea: median, robust σ, and the histogram shape. Measured over the held-out
   tiles only, and over pixels outside the annotated boxes, so that it is sea rather than hulls.
3. Solve the window, and copy the two numbers into the config.
4. Record the run identifier, the epoch, and the SHA-256 of the file.

Weights live outside git, under a gitignored `models/`. The repository carries the path, the
provenance and the digest, not 330 MB.

## Testing

On the torch-free side, and therefore in CI: the stretch places the sea where the reference says
it should sit; a pixel below the floor returns 0.0 and one above the ceiling returns 1.0; a NaN
comes back at the sea value and never at the top of the range; `fit_window` recovers the moments
it was given.

On the torch side, skipped in CI as `test_training_run.py` already is: `TrainedDetector` satisfies
the protocol, and the hole guard discards a detection placed inside a hole.

The existing seam test keeps running with the stand-in detector. That is an acceptance criterion
of the ticket, not an incidental.

## Comparison against the baseline

No new tooling. The baseline is already written — `outputs/kattegat-lane.gpkg`, 16 detections, 2
matched, 14 dark, against 12 declarations, with 9 of the 16 stacked in one 200 m square. The
trained run writes its own layer and `_verdict` already prints the two lines that matter. What
gets written is a table in the README and an entry in `decisions.md`: how many declarations each
detector recovered, how many dark detections each returned, and whether the stack of nine
survives.

This is the only honest evidence available here, and it is weak: 12 vessels on one scene. It will
be written as that, not as a validation. A mapping selected against 12 declared positions on the
same scene it is then reported on is tuned on its own evaluation, and the README says so.

## Out of scope

Small-target anchors, a dual-polarisation stem, and a learning-rate schedule all belong to #11,
which has to measure each change against the configuration before it. This ticket *is* that
configuration.

## Acceptance, against the issue's own list

- The trained model satisfies the same `detector` parameter — `_detector_from` branch, same
  protocol, pipeline untouched.
- The chain runs end to end on the real scene with the trained detector — 9 tiles at 800/64.
- Results are compared against the threshold baseline on the same scene — README table.
- The seam test still passes with the stub detector — unchanged and still in CI.
- README updated to reflect Level 2 complete.
