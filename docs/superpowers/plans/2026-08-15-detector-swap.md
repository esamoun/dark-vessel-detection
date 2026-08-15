# Detector Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the trained Faster R-CNN behind the pipeline's `detector` parameter, so the chain runs end to end on the real Sentinel-1 scene and its result can be compared against the threshold baseline.

**Architecture:** A new torch-free module (`amplitude.py`) owns the decibel-to-0..1 conversion, the robust sea estimate and the nodata guard — everything that can be wrong silently, on the side of the seam a laptop tests in a second. A new torch module (`trained.py`) wraps the checkpoint behind the existing `Detector` protocol. The pipeline itself is not touched; the CLI gains one branch and one refusal.

**Tech Stack:** Python 3.11+, numpy, rasterio, pytest, ruff; torch/torchvision behind the `detector` extra.

## Global Constraints

- `requires-python = ">=3.11"`.
- ruff: `line-length = 100`, `select = ["E", "F", "I", "UP", "B"]`. Run `make lint` before every commit.
- **The chain must install and run with no torch, no GPU and no network.** `darkvessel run` with `detector: bright-pixel` must never import torch. Any torch import lives inside the function that needs it, as `cli._train` already does.
- Tests that need torch use `torch = pytest.importorskip("torch", reason="the detector extra is not installed: pip install -e '.[detector]'")` at module top, with `# noqa: E402` on the imports below it. This is the existing idiom in `tests/test_training_run.py:23`.
- No `.pt` file is ever committed. `.gitignore` already ignores `*.pt` globally, so `models/` needs no new rule.
- Metrics are reported, never asserted. No test may pin a precision, a recall or a detection count from the trained model.
- British spelling in prose and identifiers, matching the existing code (`polarisations`, `optimiser`, `georeferenced`).

## File Structure

| File | Responsibility |
| --- | --- |
| `src/darkvessel/detect/amplitude.py` | **new.** The decibel window, the robust sea estimate, the nodata guard. No torch. |
| `src/darkvessel/detect/trained.py` | **new.** `TrainedDetector`, satisfying the `Detector` protocol. Imports torch. |
| `src/darkvessel/detect/model.py` | gains `detections_from`, moved up from `train.py`. |
| `src/darkvessel/detect/train.py` | loses `_detections_from`; writes its build block into the checkpoint. |
| `src/darkvessel/cli.py` | `trained_request_from`, a `"trained"` branch in `_detector_from`, and the tile-size refusal. |
| `configs/kattegat-lane.yaml` | tiling to 800/64; the `trained` run block. |
| `tests/test_amplitude.py` | **new.** Everything in `amplitude.py`. Runs in CI. |
| `tests/test_trained_detector.py` | **new.** `TrainedDetector` against a real model on the CPU. Skipped in CI. |
| `tests/test_training_run.py` | one test added: the checkpoint carries its build block. |
| `tests/test_pipeline.py` | two tests added: the trained config parses, and a tile-size disagreement is refused. |
| `docs/decisions.md`, `docs/failures.md`, `README.md` | the window's provenance, and the comparison. |

## Task Dependency

**Task 1 is run by a human in a Kaggle session and produces two numbers and a file.** Tasks 2–8 do not consume its output and can begin immediately. Only Task 9 needs it. If the Kaggle session is not available yet, start at Task 2 and come back.

---

### Task 1: The Kaggle pass — measure the sea, bring down the weights

**Run by:** a human, in a Kaggle notebook. Not an agent task.

**Produces:**
- `models/epoch-012.pt` in the working tree (gitignored).
- Three numbers to paste into Task 9: `floor_db`, `ceiling_db`, and the LS-SSDD sea reference they came from.
- A histogram sketch that answers whether the affine belongs on decibels or on linear amplitude.

**Setup:** start a new Kaggle notebook. Attach two data sources: the LS-SSDD dataset (`ls-ssdd-v10`), and **the output of the notebook version that ran the training** — that is where `checkpoints/epoch-012.pt` survives. Turn the internet switch on.

- [ ] **Step 1: Locate the checkpoint**

```python
import subprocess
print(subprocess.run(["find", "/kaggle/input", "-name", "epoch-*.pt"],
                     capture_output=True, text=True).stdout)
```

Expected: a path ending `checkpoints/epoch-012.pt`, and probably `epoch-011.pt` beside it. `keep: 2` deleted everything earlier, including epoch 9. If nothing is found, the session's output was not saved and the run has to be repeated — stop here and say so.

- [ ] **Step 2: Strip the optimiser and write the file to download**

The checkpoint carries SGD momentum buffers, which are the same size as the weights and are useless for inference. Dropping them halves the download. The build block is added here because the run that wrote this file predates Task 5.

```python
import hashlib, torch
from pathlib import Path

SOURCE = Path("<the path Step 1 printed>")
OUT = Path("/kaggle/working/epoch-012.pt")

state = torch.load(SOURCE, map_location="cpu", weights_only=True)
torch.save(
    {
        "epoch": state["epoch"],
        "model": state["model"],
        # What built it, read off configs/train.yaml of the run that produced it. Recorded here
        # because this checkpoint predates train.py writing its own; see docs/decisions.md.
        "built": {
            "tile_px": 800,
            "anchor_sizes": ((32,), (64,), (128,), (256,), (512,)),
            "seed": 20260814,
            "pretrained": True,
            "trainable_backbone_layers": 3,
        },
    },
    OUT,
)
print(OUT.stat().st_size / 1e6, "MB")
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
```

Expected: about 165 MB, and a digest. **Write the digest down** — it goes into `docs/decisions.md` in Task 9.

- [ ] **Step 3: Measure the LS-SSDD sea**

Uses the repository's own catalogue and reader, so the split and the 8-bit conversion are the tested ones rather than a second implementation. Sea is every pixel outside an annotated box, with a 4 px margin so a hull's bright halo does not count as sea. Every fifth held-out tile, which is 600 tiles and some 380 million pixels — far more than two moments need.

```python
!pip install -q -e /kaggle/working/dark-vessel-detection    # or wherever the repo is cloned

import numpy as np
from pathlib import Path
from darkvessel.detect.dataset import catalogue, split_by_scene

MARGIN = 4
ROOT = Path("/kaggle/input/ls-ssdd-v10/LS-SSDD-v1.0-OPEN")

_, held_out = split_by_scene(catalogue(ROOT))
hist = np.zeros(256, dtype=np.int64)

for ref in held_out[::5]:
    tile = ref.read()                      # 0..1 float32, straight from the 8-bit JPEG
    sea = np.ones(tile.image.shape, dtype=bool)
    for box in tile.boxes:
        r0 = max(int(box.min_row) - MARGIN, 0)
        r1 = min(int(np.ceil(box.max_row)) + MARGIN, sea.shape[0])
        c0 = max(int(box.min_col) - MARGIN, 0)
        c1 = min(int(np.ceil(box.max_col)) + MARGIN, sea.shape[1])
        sea[r0:r1, c0:c1] = False
    values = np.rint(tile.image[sea] * 255.0).astype(np.uint8)
    hist += np.bincount(values, minlength=256)

print(f"{hist.sum():,} sea pixels over {len(held_out[::5])} held-out tiles")
```

Expected: a line reading roughly `380,000,000 sea pixels over 600 held-out tiles`.

- [ ] **Step 4: Read the moments off the histogram, and look at its shape**

8-bit values in a 256-bin histogram are exact, so these quantiles are exact rather than estimated.

```python
def quantile(hist, q):
    total = hist.sum()
    return int(np.searchsorted(np.cumsum(hist), q * total))

median = quantile(hist, 0.5)
# MAD, from the histogram of |value - median|, folded onto the same bins.
folded = np.zeros(256, dtype=np.int64)
for value, count in enumerate(hist):
    folded[abs(value - median)] += count
sigma = quantile(folded, 0.5) * 1.4826

mean = float((np.arange(256) * hist).sum() / hist.sum())
print(f"LS-SSDD sea: median {median}/255 = {median/255:.4f}, "
      f"robust sigma {sigma:.2f}/255 = {sigma/255:.4f}, mean {mean/255:.4f}")
print(f"p1 {quantile(hist,0.01)}  p25 {quantile(hist,0.25)}  "
      f"p75 {quantile(hist,0.75)}  p99 {quantile(hist,0.99)}")

# The shape, in 32 bars. This is what decides decibels against linear amplitude.
coarse = hist.reshape(32, 8).sum(axis=1)
for i, count in enumerate(coarse):
    print(f"{i*8:3d}-{i*8+7:3d} {'#' * int(60 * count / coarse.max()):<60} {count:,}")
```

**How to read the bars.** A hump sitting away from zero, roughly symmetric, means the source stretch was log-like and the affine belongs on decibels — proceed as planned. A spike jammed against the first bars with a long thin tail to the right means the stretch was linear, and Task 9 must map through `σ⁰ = 10 ** (dB / 10)` before the affine. **Record which one it is**; the rejected one goes into `docs/decisions.md`.

- [ ] **Step 5: Solve the window**

The same three lines `fit_window` implements in Task 3, written out here because `amplitude.py` may not exist yet. The scene's two numbers are measured, not assumed — they come from `data/real/kattegat-lane.tif` and are reproduced by `sea_level` in Task 3.

```python
SCENE_SEA_DB = -21.84      # median of kattegat-lane.tif, holes excluded
SCENE_SPREAD_DB = 2.30     # MAD * 1.4826 of the same

mu_ref, sd_ref = median / 255, sigma / 255
span = SCENE_SPREAD_DB / sd_ref
floor = SCENE_SEA_DB - mu_ref * span
print(f"floor_db: {floor:.2f}\nceiling_db: {floor + span:.2f}\nsea_db: {SCENE_SEA_DB}")
```

Expected: a floor near −29 dB — that end is settled by the scene whatever the reference says — and a ceiling somewhere between −1 and +48 dB, which is the number this whole pass exists to fix. **Write all three down.**

- [ ] **Step 6: Download and place the file**

Kaggle's notebook Output tab offers `/kaggle/working/epoch-012.pt`. Download it, then locally:

```bash
mkdir -p models && mv ~/Downloads/epoch-012.pt models/
shasum -a 256 models/epoch-012.pt
```

Expected: the digest matches what Step 2 printed. If it does not, the download truncated — fetch it again.

---

### Task 2: The decibel window

**Files:**
- Create: `src/darkvessel/detect/amplitude.py`
- Test: `tests/test_amplitude.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DecibelStretch(floor_db: float, ceiling_db: float, sea_db: float)`, callable as `(np.ndarray) -> np.ndarray` returning float32 in 0..1; property `sea -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_amplitude.py`:

```python
"""The conversion between what the chain exports and what the detector was fitted on.

Every number here is arithmetic on a window chosen in the test rather than the shipped one. The
shipped window is a measurement recorded in docs/decisions.md, and a test that pinned it would
turn that measurement into a target — the same rule this project applies to precision and recall.
"""

import numpy as np
import pytest

from darkvessel.detect.amplitude import DecibelStretch

# A window with round numbers, so every expectation below is arithmetic a reader can do by eye:
# 40 dB wide, and the sea eight of those forty above the floor.
STRETCH = DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-22.0)


def test_the_floor_is_zero_and_the_ceiling_is_one():
    converted = STRETCH(np.array([[-30.0, 10.0]], dtype=np.float32))
    assert converted.tolist() == [[0.0, 1.0]]


def test_the_sea_lands_where_the_window_puts_it():
    assert STRETCH(np.array([[-22.0]], dtype=np.float32))[0, 0] == pytest.approx(0.2)


def test_beyond_either_end_is_clipped_rather_than_wrapped():
    converted = STRETCH(np.array([[-48.0, 27.0]], dtype=np.float32))
    assert converted.tolist() == [[0.0, 1.0]]


def test_a_hole_comes_back_at_the_sea_and_never_at_the_top():
    converted = STRETCH(np.array([[np.nan, -22.0]], dtype=np.float32))
    assert converted[0, 0] == pytest.approx(0.2)
    assert converted[0, 0] == converted[0, 1]


def test_nothing_comes_back_as_nan():
    converted = STRETCH(np.array([[np.nan, np.nan]], dtype=np.float32))
    assert np.isfinite(converted).all()


def test_the_input_is_not_modified():
    image = np.array([[np.nan, -22.0]], dtype=np.float32)
    STRETCH(image)
    assert np.isnan(image[0, 0])


def test_the_result_is_float32_as_the_model_input_expects():
    assert STRETCH(np.array([[-22.0]], dtype=np.float64)).dtype == np.float32


def test_a_window_that_does_not_widen_is_refused():
    with pytest.raises(ValueError, match="ceiling"):
        DecibelStretch(floor_db=0.0, ceiling_db=-10.0, sea_db=-5.0)


def test_a_sea_outside_its_own_window_is_refused():
    with pytest.raises(ValueError, match="sea"):
        DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-40.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_amplitude.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'darkvessel.detect.amplitude'`.

- [ ] **Step 3: Write the implementation**

Create `src/darkvessel/detect/amplitude.py`:

```python
"""What the chain exports, in the unit the detector was fitted on.

The chain hands out calibrated decibels; the model was fitted on 8-bit amplitude divided by 255.
These are not the same quantity, and the stretch LS-SSDD's authors used to turn one into the
other is not recorded in the dataset and cannot be recovered from it. So the mapping is chosen,
and the choice is made by matching the sea — see `fit_window` and docs/decisions.md.

What ships is a fixed window in decibels, not a fit performed per scene. A window refitted on
every acquisition is a percentile stretch under another name: the same hull would take a
different value under a different sea state, and a score threshold would stop meaning the same
thing from one scene to the next.

Nothing here imports torch. The window, the sea estimate and the hole guard are exactly the
decisions that go wrong without a symptom, so they live on the side of the seam a laptop tests in
a second — the same division `dataset.py` and `checkpoints.py` are on.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from darkvessel.detect.detector import PixelDetection

# Turns a median absolute deviation into the standard deviation of the Gaussian that would
# produce it. Used rather than a plain standard deviation because a scene holds ships, and a ship
# is forty decibels above the sea it sits in: on kattegat-lane.tif the plain figure is 2.57 dB
# against 2.30 dB robustly, and the difference is entirely the targets.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class DecibelStretch:
    """The window of decibels the model's 0..1 covers, and where a hole sits inside it.

    `sea_db` is not redundant with the two ends. It is where this product's sea stands, and it is
    what a nodata hole receives *before* the stretch — so the fill follows the window
    automatically and two numbers that must agree cannot drift apart.
    """

    floor_db: float
    ceiling_db: float
    sea_db: float

    def __post_init__(self) -> None:
        if self.ceiling_db <= self.floor_db:
            raise ValueError(
                f"the ceiling is {self.ceiling_db} dB and the floor {self.floor_db} dB; a window "
                "has to widen upwards or it maps every pixel to the same value"
            )
        if not self.floor_db <= self.sea_db <= self.ceiling_db:
            raise ValueError(
                f"the sea is at {self.sea_db} dB, outside the window {self.floor_db} to "
                f"{self.ceiling_db} dB; a hole would then be filled at one end of the range "
                "rather than at the sea, which is the one thing this fill exists to avoid"
            )

    @property
    def sea(self) -> float:
        """Where the sea lands once the window is applied. What a hole comes back as."""
        return (self.sea_db - self.floor_db) / (self.ceiling_db - self.floor_db)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """One scene or tile of decibels, as the amplitude in 0..1 the model was fitted on.

        Holes are filled before the stretch rather than after. A NaN reaching the model is not a
        hole the network ignores — it propagates through every convolution that touches it and
        empties the whole tile, with no crash and no warning. `scene.py` writes them as NaN
        precisely because every comparison against NaN is false, which immunises a threshold; a
        network has no such immunity, and this is where that difference is paid for.
        """
        filled = np.where(np.isnan(image), np.float32(self.sea_db), image)
        scaled = (filled - self.floor_db) / (self.ceiling_db - self.floor_db)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_amplitude.py -q`
Expected: 9 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add src/darkvessel/detect/amplitude.py tests/test_amplitude.py
git commit -m "feat: state the window between the chain's decibels and the model's amplitude"
```

---

### Task 3: Fitting the window to the sea, and measuring the sea

**Files:**
- Modify: `src/darkvessel/detect/amplitude.py`
- Test: `tests/test_amplitude.py`

**Interfaces:**
- Consumes: `DecibelStretch` from Task 2.
- Produces: `SeaReference(mean: float, spread: float)`; `fit_window(*, sea_db: float, spread_db: float, reference: SeaReference) -> DecibelStretch`; `sea_level(image: np.ndarray) -> tuple[float, float]` returning `(median_db, robust_sigma_db)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_amplitude.py`, and extend the import at the top to
`from darkvessel.detect.amplitude import DecibelStretch, SeaReference, fit_window, sea_level`:

```python
# A reference sea standing at 0.15 with a spread of 0.05, which is a plausible shape for an
# 8-bit product and is *not* the measured one. What the measured one is belongs in the config.
REFERENCE = SeaReference(mean=0.15, spread=0.05)


def test_the_fitted_window_puts_the_sea_where_the_reference_says():
    stretch = fit_window(sea_db=-21.84, spread_db=2.30, reference=REFERENCE)
    assert stretch(np.array([[-21.84]], dtype=np.float32))[0, 0] == pytest.approx(0.15, abs=1e-4)


def test_one_spread_above_the_sea_is_one_reference_spread_above_it():
    stretch = fit_window(sea_db=-21.84, spread_db=2.30, reference=REFERENCE)
    converted = stretch(np.array([[-21.84 + 2.30]], dtype=np.float32))
    assert converted[0, 0] == pytest.approx(0.20, abs=1e-4)


def test_the_fitted_window_fills_holes_at_the_sea_it_was_fitted_to():
    stretch = fit_window(sea_db=-21.84, spread_db=2.30, reference=REFERENCE)
    assert stretch.sea_db == -21.84
    assert stretch.sea == pytest.approx(0.15, abs=1e-4)


def test_a_reference_with_no_spread_is_refused():
    with pytest.raises(ValueError, match="spread"):
        SeaReference(mean=0.15, spread=0.0)


def test_the_sea_is_measured_past_the_ships_in_it():
    # 200 x 200 of sea at -22 dB, and forty pixels of ship at +25. A plain standard deviation
    # would be dragged upwards by the ships; the robust one must not be.
    generator = np.random.default_rng(20260815)
    image = generator.normal(-22.0, 2.0, size=(200, 200)).astype(np.float32)
    image[:5, :8] = 25.0

    median, sigma = sea_level(image)
    assert median == pytest.approx(-22.0, abs=0.15)
    assert sigma == pytest.approx(2.0, abs=0.15)


def test_holes_do_not_count_as_sea():
    image = np.full((100, 100), -22.0, dtype=np.float32)
    image[:50] = np.nan

    median, _ = sea_level(image)
    assert median == pytest.approx(-22.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_amplitude.py -q`
Expected: `ImportError: cannot import name 'SeaReference'`.

- [ ] **Step 3: Write the implementation**

Append to `src/darkvessel/detect/amplitude.py`:

```python
@dataclass(frozen=True)
class SeaReference:
    """Where the sea stood in the images the model was fitted on, in the 0..1 it was handed.

    Measured over the held-out tiles, outside the annotated boxes, and recorded in
    docs/decisions.md with the run that measured it. It is a property of the training set rather
    than of any scene the chain later reads.
    """

    mean: float
    spread: float

    def __post_init__(self) -> None:
        if self.spread <= 0:
            raise ValueError(
                f"a reference spread of {self.spread} describes a sea with no variation in it, "
                "and the window is fitted by dividing by it"
            )


def fit_window(*, sea_db: float, spread_db: float, reference: SeaReference) -> DecibelStretch:
    """The window that puts this product's sea where the model's sea was.

    Two moments, two parameters. The stretch LS-SSDD applied is not recoverable, but the
    statistics the model was actually fitted under are — so the sea is matched rather than the
    processing chain guessed at.

    Called to *derive* the shipped constants, not on a run. What a run reads is the answer, in
    the config, where a reader can see both ends of the window at once.
    """
    span = spread_db / reference.spread
    floor = sea_db - reference.mean * span
    return DecibelStretch(floor_db=floor, ceiling_db=floor + span, sea_db=sea_db)


def sea_level(image: np.ndarray) -> tuple[float, float]:
    """This scene's sea, as a median and a spread in decibels, ignoring holes.

    Robust rather than the plain mean and standard deviation, because a scene contains ships and
    a ship stands forty decibels above the water. On kattegat-lane.tif the plain spread is
    2.57 dB against 2.30 dB here, and the whole of that difference is the targets — which would
    then widen the window that is supposed to make them stand out.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("the scene is entirely nodata; there is no sea to measure")

    median = float(np.median(finite))
    return median, float(np.median(np.abs(finite - median)) * MAD_TO_SIGMA)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_amplitude.py -q`
Expected: 15 passed.

- [ ] **Step 5: Check it against the real scene**

Run:

```bash
python3 -c "
from darkvessel.data.scene import Scene
from darkvessel.detect.amplitude import sea_level
from pathlib import Path
print('%.2f dB, sigma %.2f dB' % sea_level(Scene.from_geotiff(Path('data/real/kattegat-lane.tif')).image))
"
```

Expected: `-21.84 dB, sigma 2.30 dB`. These are the two numbers Task 1 Step 5 uses. If they differ, the scene on disk is not the one this plan was written against — stop and say so.

- [ ] **Step 6: Lint and commit**

```bash
make lint
git add src/darkvessel/detect/amplitude.py tests/test_amplitude.py
git commit -m "feat: fit the window by matching the sea, and measure a scene's sea past its ships"
```

---

### Task 4: The nodata guard

**Files:**
- Modify: `src/darkvessel/detect/amplitude.py`
- Test: `tests/test_amplitude.py`

**Interfaces:**
- Consumes: `PixelDetection` from `darkvessel.detect.detector`.
- Produces: `without_holes(detections: Sequence[PixelDetection], image: np.ndarray) -> list[PixelDetection]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_amplitude.py`, extending the imports with `without_holes` and adding
`from darkvessel.detect.detector import PixelDetection`:

```python
def _tile_with_a_hole_at(row: int, col: int) -> np.ndarray:
    image = np.full((4, 4), -22.0, dtype=np.float32)
    image[row, col] = np.nan
    return image


def test_a_detection_centred_in_a_hole_is_dropped():
    image = _tile_with_a_hole_at(1, 1)
    kept = without_holes([PixelDetection(row=1.0, col=1.0, score=0.9)], image)
    assert kept == []


def test_a_detection_on_water_survives():
    image = _tile_with_a_hole_at(1, 1)
    detection = PixelDetection(row=2.0, col=2.0, score=0.9)
    assert without_holes([detection], image) == [detection]


def test_the_pixel_a_fractional_centre_falls_in_is_the_one_that_is_checked():
    # 1.4 addresses a point inside pixel 1, which is the hole. 1.6 is inside pixel 2, which is
    # not. The half added before flooring is the same convention tiling.Core.contains uses.
    image = _tile_with_a_hole_at(1, 1)
    assert without_holes([PixelDetection(row=1.4, col=1.4, score=0.9)], image) == []
    assert len(without_holes([PixelDetection(row=1.6, col=1.6, score=0.9)], image)) == 1


def test_order_and_scores_are_left_alone():
    image = _tile_with_a_hole_at(0, 0)
    detections = [
        PixelDetection(row=3.0, col=1.0, score=0.4),
        PixelDetection(row=1.0, col=2.0, score=0.9),
    ]
    assert without_holes(detections, image) == detections
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_amplitude.py -q`
Expected: `ImportError: cannot import name 'without_holes'`.

- [ ] **Step 3: Write the implementation**

Append to `src/darkvessel/detect/amplitude.py`:

```python
def without_holes(
    detections: Sequence[PixelDetection], image: np.ndarray
) -> list[PixelDetection]:
    """Drop anything reported from a pixel the product declared as nodata.

    The second of two independent mechanisms, and it is not redundant with the first. Filling a
    hole at sea level stops its boundary from being a feature; this stops the hole itself from
    ever being reported as a target, whatever the fill does and whatever a later change to the
    fill might do. Each can be removed without silently disabling the other.

    Takes the image as the scene gave it — with the holes still NaN — so it must be applied
    before the stretch fills them, not after.

    The half added before flooring converts a fractional pixel index, which addresses a pixel
    centre, into the pixel it falls in. It is the same half `tiling.Core.contains` adds, for the
    same reason: comparing an index against an edge without it is a silent half-pixel error.
    """
    rows, cols = image.shape
    kept = []
    for detection in detections:
        row = int(np.floor(detection.row + 0.5))
        col = int(np.floor(detection.col + 0.5))
        if 0 <= row < rows and 0 <= col < cols and np.isnan(image[row, col]):
            continue
        kept.append(detection)

    return kept
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_amplitude.py -q`
Expected: 19 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add src/darkvessel/detect/amplitude.py tests/test_amplitude.py
git commit -m "feat: a hole cannot be reported as a target, whatever it was filled with"
```

---

### Task 5: `detections_from` moves to the tensor boundary

**Files:**
- Modify: `src/darkvessel/detect/model.py` (append), `src/darkvessel/detect/train.py:294-307` (delete `_detections_from`, import instead)

**Interfaces:**
- Consumes: `Box` from `darkvessel.detect.dataset`, `PixelDetection` from `darkvessel.detect.detector`.
- Produces: `detections_from(output: dict[str, Tensor]) -> list[PixelDetection]` in `model.py`.

This is a move, not a rewrite. The behaviour is already covered by `tests/test_training_run.py`, which runs the real loop and the real scoring; if the move breaks it, that suite fails.

- [ ] **Step 1: Add the function to `model.py`**

Append to `src/darkvessel/detect/model.py`, and extend its imports with
`from darkvessel.detect.dataset import Box` and `from darkvessel.detect.detector import PixelDetection`:

```python
def detections_from(output: dict[str, Tensor]) -> list[PixelDetection]:
    """A model's boxes, as the points the rest of the chain deals in.

    Through `Box.from_xyxy` and `Box.centre` rather than by unpacking the corners here, so that
    the axis swap and the half-pixel between an edge coordinate and a pixel index are each
    applied in the one place that owns them.

    Beside `as_model_input` because it is the other half of the same boundary: one turns a tile
    into what torchvision takes, the other turns what torchvision returns back into what this
    project's contract states. It sat in `train.py` while training was the only caller; inference
    is the second, and a second copy of the half-pixel is exactly the defect `Box` exists to
    prevent.
    """
    return [
        PixelDetection(row=row, col=col, score=float(score))
        for box, score in zip(
            output["boxes"].cpu().tolist(), output["scores"].cpu().tolist(), strict=True
        )
        for row, col in [Box.from_xyxy(box).centre()]
    ]
```

- [ ] **Step 2: Delete the copy in `train.py` and import the moved one**

In `src/darkvessel/detect/train.py`, delete the whole `_detections_from` function (currently
lines 294–307), change the `model` import to
`from darkvessel.detect.model import SHIP, as_model_input, detections_from`, and change the one
call site in `_score` from `measure(_detections_from(output), ...)` to
`measure(detections_from(output), ...)`. Remove `Box` from the `dataset` import line only if
nothing else in the file still uses it — `_ships_in` does, so it stays.

- [ ] **Step 3: Run the training suite to verify nothing moved but the code**

Run: `pytest tests/test_training_run.py -q`
Expected: passed (not skipped — torch 2.13.0 is installed on this machine). If it is skipped, run `pip install -e ".[detector]"` first.

- [ ] **Step 4: Verify no dangling reference remains**

Run: `grep -rn "_detections_from" src tests`
Expected: no output.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add src/darkvessel/detect/model.py src/darkvessel/detect/train.py
git commit -m "refactor: put both halves of the tensor boundary in one module"
```

---

### Task 6: The checkpoint records what built it

**Files:**
- Modify: `src/darkvessel/detect/train.py:47-58` (signature), `:113-121` (the save), `src/darkvessel/cli.py:372-381` (the call)
- Test: `tests/test_training_run.py`

**Interfaces:**
- Consumes: nothing.
- Produces: checkpoints carrying `state["built"]`, a dict with keys `tile_px`, `anchor_sizes`, `seed`, `pretrained`, `trainable_backbone_layers`.

`AnchorGenerator` holds no parameters and `min_size`/`max_size` are transform attributes, so a
checkpoint loaded into a model built with different anchors loads cleanly and is silently wrong.
This closes that for every future run.

- [ ] **Step 1: Extend the shared fixture, then write the failing test**

`built` is a required keyword, so `a_run` — the helper every test in this file passes to
`train(**...)` — has to supply it, or all of them break. Change the import line
`from darkvessel.detect.model import detector_model` to
`from darkvessel.detect.model import ANCHOR_SIZES, detector_model  # noqa: E402`, and add this
entry to the dict `a_run` returns, beside `"device"`:

```python
        # What built the model above. `a_run` builds it with the stock anchors, so this says so;
        # a fixture whose build block described a different model would be testing nothing.
        "built": {
            "tile_px": TILE_PX,
            "anchor_sizes": ANCHOR_SIZES,
            "seed": 1,
            "pretrained": False,
            "trainable_backbone_layers": 5,
        },
```

Then append the test:

```python
def test_the_checkpoint_records_what_built_it(tmp_path: Path) -> None:
    """A checkpoint that does not say what built it can be loaded into the wrong model.

    Anchor sizes are not weights — `AnchorGenerator` holds no parameters, and min_size/max_size
    are attributes of the transform — so a state dict fitted under one set loads without
    complaint under another and then looks for ships of the wrong size, quietly. The build block
    is what lets the side that loads it refuse.
    """
    run = a_run(tmp_path, epochs=1)
    train(**run)

    _, path = Checkpoints(tmp_path / "run").latest()
    state = torch.load(path, map_location="cpu", weights_only=True)

    assert state["built"] == run["built"]
    assert state["built"]["anchor_sizes"] == ANCHOR_SIZES
```

- [ ] **Step 2: Run the whole file to verify the new test fails and the others still would**

Run: `pytest tests/test_training_run.py -q`
Expected: every test in the file fails with
`TypeError: train() got an unexpected keyword argument 'built'`. That is the point of putting
`built` in `a_run`: the parameter is required, so nothing can quietly keep passing without it.

- [ ] **Step 3: Add the parameter and write it**

In `src/darkvessel/detect/train.py`, add `built: dict[str, Any],` to `train`'s keyword-only
parameters (after `device`), import `Any` from `typing`, document it in the docstring as *"What
built the model, written into every checkpoint so that the side that loads it can refuse a model
built differently"*, and extend the save at line 113:

```python
        with checkpoints.writing(epoch) as partial:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    # Not weights, and that is the point: anchor sizes leave no trace in a state
                    # dict, so a checkpoint that does not name them can be loaded into a model
                    # that looks for ships of a different size and will not say so.
                    "built": built,
                },
                partial,
            )
```

- [ ] **Step 4: Pass it from the CLI**

In `src/darkvessel/cli.py`, inside `_train`, change the `train(...)` call to pass
`built={"tile_px": request["tile_px"], **request["model"]}` alongside the existing arguments.

- [ ] **Step 5: Run the training suite**

Run: `pytest tests/test_training_run.py -q`
Expected: all passed, including the new test.

- [ ] **Step 6: Lint and commit**

```bash
make lint
git add src/darkvessel/detect/train.py src/darkvessel/cli.py tests/test_training_run.py
git commit -m "fix: a checkpoint that does not name its anchors can be loaded into the wrong model"
```

---

### Task 7: `TrainedDetector`

**Files:**
- Create: `src/darkvessel/detect/trained.py`
- Test: `tests/test_trained_detector.py`

**Interfaces:**
- Consumes: `DecibelStretch`, `without_holes` (Task 2, 4); `as_model_input`, `detections_from`, `detector_model` (Task 5).
- Produces: `TrainedDetector(*, checkpoint: Path, stretch: DecibelStretch, score_threshold: float, tile_px: int, anchor_sizes: tuple[tuple[int, ...], ...], device: torch.device | None = None)`, callable as `(np.ndarray) -> list[PixelDetection]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trained_detector.py`:

```python
"""The trained model behind the contract the stand-in satisfies.

Nothing here asserts how well it detects. A model is evaluated, not asserted — the same rule
test_pipeline.py states. What is asserted is that it satisfies the protocol, that it refuses a
checkpoint built for a different detector, and that a hole cannot come back as a target.

Skipped where torch is not installed, which includes CI: the chain's acceptance condition is that
it installs and runs without a framework.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="the detector extra is not installed: pip install -e '.[detector]'"
)

from darkvessel.detect.amplitude import DecibelStretch  # noqa: E402
from darkvessel.detect.detector import PixelDetection  # noqa: E402
from darkvessel.detect.model import detector_model  # noqa: E402
from darkvessel.detect.trained import TrainedDetector  # noqa: E402

# Small enough that a ResNet-50 FPN runs on a laptop CPU in a test. The same size
# test_training_run.py trains at.
TILE_PX = 64
ANCHORS = ((8,), (16,), (32,), (64,), (128,))
STRETCH = DecibelStretch(floor_db=-30.0, ceiling_db=10.0, sea_db=-22.0)

BUILT = {
    "tile_px": TILE_PX,
    "anchor_sizes": ANCHORS,
    "seed": 7,
    "pretrained": False,
    "trainable_backbone_layers": 3,
}


def write_checkpoint(directory: Path, built: dict | None = BUILT) -> Path:
    """An untrained model saved as a checkpoint. Its weights are meaningless and unused here."""
    model = detector_model(tile_px=TILE_PX, seed=7, anchor_sizes=ANCHORS, pretrained=False)
    state = {"epoch": 1, "model": model.state_dict()}
    if built is not None:
        state["built"] = built

    path = directory / "epoch-001.pt"
    torch.save(state, path)
    return path


def sea(rows: int = TILE_PX, cols: int = TILE_PX) -> np.ndarray:
    generator = np.random.default_rng(20260815)
    return generator.normal(-22.0, 2.3, size=(rows, cols)).astype(np.float32)


def detector_at(path: Path, score_threshold: float = 0.75) -> TrainedDetector:
    return TrainedDetector(
        checkpoint=path,
        stretch=STRETCH,
        score_threshold=score_threshold,
        tile_px=TILE_PX,
        anchor_sizes=ANCHORS,
        device=torch.device("cpu"),
    )


def test_it_satisfies_the_detector_contract(tmp_path):
    found = detector_at(write_checkpoint(tmp_path))(sea())
    assert isinstance(found, list)
    assert all(isinstance(detection, PixelDetection) for detection in found)


def test_a_tile_of_holes_does_not_empty_the_answer_with_nan(tmp_path):
    """A NaN reaching the network propagates through every convolution that touches it."""
    image = sea()
    image[:8, :8] = np.nan

    found = detector_at(write_checkpoint(tmp_path), score_threshold=0.0)(image)
    assert all(np.isfinite(detection.score) for detection in found)


def test_nothing_is_reported_from_inside_a_hole(tmp_path):
    image = np.full((TILE_PX, TILE_PX), np.nan, dtype=np.float32)

    assert detector_at(write_checkpoint(tmp_path), score_threshold=0.0)(image) == []


def test_a_checkpoint_built_for_other_anchors_is_refused(tmp_path):
    path = write_checkpoint(tmp_path, built={**BUILT, "anchor_sizes": ((32,), (64,), (128,), (256,), (512,))})

    with pytest.raises(ValueError, match="anchor_sizes"):
        detector_at(path)


def test_a_checkpoint_built_for_another_tile_size_is_refused(tmp_path):
    path = write_checkpoint(tmp_path, built={**BUILT, "tile_px": 800})

    with pytest.raises(ValueError, match="tile_px"):
        detector_at(path)


def test_a_checkpoint_from_before_the_build_block_still_loads(tmp_path):
    """epoch-012.pt predates train.py recording it; the run config restates the values."""
    assert detector_at(write_checkpoint(tmp_path, built=None)) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trained_detector.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'darkvessel.detect.trained'`.

- [ ] **Step 3: Write the implementation**

Create `src/darkvessel/detect/trained.py`:

```python
"""The trained detector, behind the contract the stand-in satisfies.

This is what the seam was built for. The pipeline takes a detector as a parameter and never
learns which one it got, so swapping the model in is one branch in the command that builds it and
nothing else — no stage of the chain changes, and the deterministic substitute keeps working
beside it.

Three things stand between a checkpoint and a scene, and all three are handled here or in
`amplitude.py` rather than anywhere the pipeline can see:

* the chain deals in calibrated decibels and the model was fitted on amplitude in 0..1;
* a product's nodata arrives as NaN, which a threshold ignores and a network does not;
* a checkpoint does not record its own anchors, so it can be loaded into a detector looking for
  ships of the wrong size without a word.

torch is imported at module level here, which is safe because nothing imports this module unless
a config asks for it — `cli._detector_from` imports it inside its branch, the way `cli._train`
does. `darkvessel run` with the stand-in still needs no framework.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkvessel.detect.amplitude import DecibelStretch, without_holes
from darkvessel.detect.detector import PixelDetection
from darkvessel.detect.model import as_model_input, detections_from, detector_model


class TrainedDetector:
    """A checkpoint, run over one tile at a time, reporting what the chain's contract states."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        stretch: DecibelStretch,
        score_threshold: float,
        tile_px: int,
        anchor_sizes: tuple[tuple[int, ...], ...],
        device: torch.device | None = None,
    ) -> None:
        """Load the weights and put the model in the state it answers from.

        Args:
            checkpoint: The file a training run wrote. Only its weights and its build block are
                read; the optimiser state, where present, is not needed for inference.
            stretch: How decibels become the amplitude this model was fitted on.
            score_threshold: The confidence below which a detection is not reported. A detector
                has a precision *at* a confidence, and this is where that choice is made; the
                training run reports the whole table so it can be made against numbers.
            tile_px: The side of the tiles this model runs on. It has to be the side the chain
                cuts, or torchvision resamples between the two — `cli` refuses that case before
                anything is loaded.
            anchor_sizes: One tuple per pyramid level. Checked against the checkpoint's own
                record, because it leaves no trace in a state dict.
            device: Where to run. Defaults to the GPU if there is one.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _check_built(state.get("built"), tile_px=tile_px, anchor_sizes=anchor_sizes)

        # `pretrained=False` because every weight is about to be overwritten, and because
        # fetching COCO weights would put a network on the path of a command that must not need
        # one. The seed is for the same reason irrelevant: the head it initialises is replaced by
        # the load below.
        model = detector_model(
            tile_px=tile_px, seed=0, anchor_sizes=anchor_sizes, pretrained=False
        )
        model.load_state_dict(state["model"])

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.stretch = stretch
        self.score_threshold = score_threshold

    def __call__(self, image: np.ndarray) -> list[PixelDetection]:
        """One tile of decibels, as targets in that tile's pixel coordinates.

        The guard runs against `image` as it arrived — holes still NaN — rather than against the
        stretched copy, which no longer has any. It is the reason the stretch is applied to a
        copy rather than in place.
        """
        with torch.no_grad():
            tile = as_model_input(self.stretch(image)).to(self.device)
            output = self.model([tile])[0]

        found = [
            detection
            for detection in detections_from(output)
            if detection.score >= self.score_threshold
        ]
        return without_holes(found, image)


def _check_built(
    built: dict[str, Any] | None,
    *,
    tile_px: int,
    anchor_sizes: tuple[tuple[int, ...], ...],
) -> None:
    """Refuse a checkpoint built for a detector other than the one being constructed.

    Silence is allowed for one reason: the first trained checkpoint predates `train.py` writing
    this block, and its build parameters are restated in the run config with the training config
    named beside them. Every checkpoint written since carries its own, and a disagreement is an
    error rather than a warning — a model looking for the wrong size of ship produces detections,
    in plausible places, with scores.
    """
    if built is None:
        return

    if int(built["tile_px"]) != tile_px:
        raise ValueError(
            f"the checkpoint was built with tile_px {built['tile_px']} and this run asks for "
            f"{tile_px}; the model would resample every tile between the two"
        )

    recorded = _as_sizes(built["anchor_sizes"])
    if recorded != _as_sizes(anchor_sizes):
        raise ValueError(
            f"the checkpoint was built with anchor_sizes {recorded} and this run asks for "
            f"{_as_sizes(anchor_sizes)}; anchors are not weights, so this would load cleanly and "
            "look for ships of the wrong size without saying so"
        )


def _as_sizes(sizes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """One shape for anchor sizes, so a list from YAML and a tuple from a checkpoint compare."""
    return tuple(tuple(int(size) for size in level) for level in sizes)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trained_detector.py -q`
Expected: 6 passed. Allow a minute — each test builds a ResNet-50 FPN on the CPU.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add src/darkvessel/detect/trained.py tests/test_trained_detector.py
git commit -m "feat: the trained model, behind the contract the stand-in satisfies"
```

---

### Task 8: Wiring it to a config, and refusing a tiling it cannot run at

**Files:**
- Modify: `src/darkvessel/cli.py:91-128` (`_run`), `:462-467` (`_detector_from`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `TrainedDetector` (Task 7), `DecibelStretch` (Task 2).
- Produces: `trained_request_from(run_config: dict[str, Any], relative_to: Path) -> dict[str, Any]` and `check_tile_size(run_config: dict[str, Any], tiling: Tiling) -> None`, both importable with no torch installed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`, extending its `cli` import with `check_tile_size` and
`trained_request_from`:

```python
TRAINED_RUN = {
    "detector": "trained",
    "trained": {
        "checkpoint": "../models/epoch-012.pt",
        "tile_px": 800,
        "anchor_sizes": [[32], [64], [128], [256], [512]],
        "score_threshold": 0.75,
        "stretch": {"floor_db": -28.74, "ceiling_db": 17.26, "sea_db": -21.84},
    },
}


def test_a_trained_run_is_read_without_the_framework_installed(tmp_path):
    """Every key of a shipped config goes through a function a test can call, or it becomes the
    one key nothing in the suite ever parses — the argument `export_request_from` already makes.
    Here it is sharper: the framework this config names is an optional extra."""
    request = trained_request_from(TRAINED_RUN, tmp_path)

    assert request["checkpoint"] == (tmp_path / "../models/epoch-012.pt").resolve()
    assert request["tile_px"] == 800
    assert request["anchor_sizes"] == ((32,), (64,), (128,), (256,), (512,))
    assert request["score_threshold"] == 0.75
    assert request["stretch"].floor_db == -28.74
    assert request["stretch"].sea == pytest.approx((-21.84 + 28.74) / (17.26 + 28.74))


def test_a_tiling_the_model_was_not_built_for_is_refused():
    with pytest.raises(ValueError, match="800"):
        check_tile_size(TRAINED_RUN, Tiling(size_px=512, overlap_px=64))


def test_the_tiling_the_model_was_built_for_is_accepted():
    check_tile_size(TRAINED_RUN, Tiling(size_px=800, overlap_px=64))


def test_the_stand_in_is_not_asked_what_tile_size_it_wants():
    check_tile_size({"detector": "bright-pixel", "threshold": 0.5}, Tiling(size_px=144, overlap_px=32))


def test_the_shipped_real_config_names_a_tiling_its_detector_can_run_at():
    """The one config in this package that nothing else in the suite runs — it needs Earth Engine
    credentials to reach any of its other stages. This is the check that it is internally
    consistent, which is the argument `fusion_settings_from` already makes for the fusion keys."""
    path = Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml"
    config = yaml.safe_load(path.read_text())

    check_tile_size(
        config["run"],
        Tiling(
            size_px=int(config["tiling"]["size_px"]),
            overlap_px=int(config["tiling"]["overlap_px"]),
        ),
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline.py -q`
Expected: `ImportError: cannot import name 'check_tile_size'`.

- [ ] **Step 3: Write the implementation**

In `src/darkvessel/cli.py`, add `from darkvessel.detect.amplitude import DecibelStretch` to the
imports (no torch in it), and add these two functions beside the other `*_request_from` ones:

```python
def trained_request_from(run_config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What the trained detector takes from a config file.

    Separate from the command that builds it for the reason `training_request_from` is, with the
    constraint drawn the other way round: that one is the stage that needs a GPU, this one is the
    stage that needs the framework to be installed at all. A mistyped key here would otherwise
    surface only on a machine with the detector extra on it.

    `anchor_sizes` and `tile_px` are restated here rather than read off the checkpoint because
    the first trained checkpoint predates `train.py` recording them. Where a checkpoint does
    carry its build block, `TrainedDetector` refuses a disagreement rather than preferring one.
    """
    trained = run_config["trained"]
    stretch = trained["stretch"]

    return {
        "checkpoint": (relative_to / trained["checkpoint"]).resolve(),
        "tile_px": int(trained["tile_px"]),
        "anchor_sizes": tuple(tuple(int(size) for size in level) for level in trained["anchor_sizes"]),
        "score_threshold": float(trained["score_threshold"]),
        "stretch": DecibelStretch(
            floor_db=float(stretch["floor_db"]),
            ceiling_db=float(stretch["ceiling_db"]),
            sea_db=float(stretch["sea_db"]),
        ),
    }


def check_tile_size(run_config: dict[str, Any], tiling: Tiling) -> None:
    """Refuse a run whose tiles are not the size its detector was built for.

    The same shape of refusal `_check_working_crs` makes, and for the same reason. Torchvision
    would resize each tile to the size the model declares, silently, and resampling radar
    amplitude is a decision rather than a convenience: it changes what the detector sees, and the
    precision and recall reported for this model were measured at one scale and not the other.

    The stand-in has no opinion about tile size, so it is not asked.
    """
    if run_config["detector"] != "trained":
        return

    tile_px = int(run_config["trained"]["tile_px"])
    if tiling.size_px != tile_px:
        raise ValueError(
            f"the chain cuts {tiling.size_px} px tiles and the detector was built for {tile_px} "
            "px; the model would resample between the two, which changes what it sees. Set "
            "tiling.size_px to the model's, or run a model built for this tiling"
        )
```

Change `_detector_from` to take the config directory and gain its branch:

```python
def _detector_from(run_config: dict[str, Any], relative_to: Path) -> Detector:
    """Build the detector named by the config. This is the injection point."""
    name = run_config["detector"]
    if name == "bright-pixel":
        return BrightPixelDetector(threshold=float(run_config["threshold"]))
    if name == "trained":
        # Imported here rather than at the top of the module, the way `_train` imports torch: the
        # chain's acceptance condition is that it installs and runs with no framework, and a run
        # with the stand-in must not pull two gigabytes of CUDA wheels to threshold pixels.
        from darkvessel.detect.trained import TrainedDetector

        return TrainedDetector(**trained_request_from(run_config, relative_to))
    raise ValueError(f"unknown detector {name!r}; known detectors: 'bright-pixel', 'trained'")
```

And in `_run`, build the tiling before the detector, check it, then pass `relative_to`:

```python
    tiling = _tiling_from(config["tiling"])
    check_tile_size(run_config, tiling)

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=_detector_from(run_config, relative_to),
        tiling=tiling,
        **fusion,
    )
```

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q`
Expected: all passed. `test_the_shipped_real_config_names_a_tiling_its_detector_can_run_at` will
fail until Task 9 updates the config — at this point `configs/kattegat-lane.yaml` still says
`detector: bright-pixel`, so `check_tile_size` returns early and the test passes. It becomes a
real check in Task 9.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add src/darkvessel/cli.py tests/test_pipeline.py
git commit -m "feat: name the trained detector in a config, and refuse a tiling it cannot run at"
```

---

### Task 9: The real run, and what it found

**Requires Task 1's three numbers and `models/epoch-012.pt`.**

**Files:**
- Modify: `configs/kattegat-lane.yaml`, `docs/decisions.md`, `README.md`

- [ ] **Step 1: Write the run block**

In `configs/kattegat-lane.yaml`, change the tiling and the run block. Substitute Task 1 Step 5's
three numbers for `floor_db`, `ceiling_db` and `sea_db`, and its Step 4 figures in the comment.

```yaml
# 800 px, because that is the size the model was trained and scored at, and Tiling returns all
# nine of them at exactly 800 x 800 with no short tile at the far edge. Cut at 512 instead and
# torchvision resizes every tile to 800 on the way in — see docs/decisions.md. `cli.check_tile_size`
# refuses the mismatch rather than letting it happen quietly.
tiling:
  size_px: 800
  overlap_px: 64

run:
  scene: ../data/real/kattegat-lane.tif
  ais: ../data/real/kattegat-lane-ais.csv
  output: ../outputs/kattegat-lane-trained.gpkg
  detector: trained
  trained:
    # Not in the repository: 165 MB of weights, and *.pt is ignored. Brought down from the
    # training run's Kaggle output; the run, the epoch and the file's digest are in
    # docs/decisions.md.
    checkpoint: ../models/epoch-012.pt
    # What built the model these weights came from, restated because this checkpoint predates
    # train.py recording it. They are configs/train.yaml's, at seed 20260814.
    tile_px: 800
    anchor_sizes: [[32], [64], [128], [256], [512]]
    # Precision 0.941 against recall 0.706 on the held-out split. Every detection this chain does
    # not match against AIS becomes a dark vessel, which is a claim someone may be sent out on:
    # a miss costs a ship nobody looked at, a false alarm costs an inspection and an accusation.
    # The training run reports the whole table so this can be moved against numbers.
    score_threshold: 0.75
    # The window between the decibels this scene is in and the amplitude the model was fitted on,
    # fitted by matching the sea. Derived once, in the Kaggle pass recorded in docs/decisions.md,
    # and fixed here: refitted per scene it would be a percentile stretch under another name.
    stretch:
      floor_db: <Task 1 Step 5>
      ceiling_db: <Task 1 Step 5>
      sea_db: -21.84
```

- [ ] **Step 2: Verify the config parses and the tiling is accepted**

Run: `pytest tests/test_pipeline.py -q`
Expected: all passed, and `test_the_shipped_real_config_names_a_tiling_its_detector_can_run_at`
is now exercising a real comparison (800 against 800) rather than returning early.

- [ ] **Step 3: Run the chain**

Run: `darkvessel run --config configs/kattegat-lane.yaml`

Expected: a detection count, the CRS, the output path, then the two `_verdict` lines — how many
matched and how many dark against 12 declared positions, and how many of the matches rest on an
interpolated position. Nine tiles of 800 px through a ResNet-50 FPN on this laptop's CPU will
take a few minutes.

**Do not tune anything to improve this number.** If it looks wrong, the finding is the finding;
record it. A window adjusted until the answer looks better is a window fitted on its own
evaluation.

- [ ] **Step 4: Record the numbers against the baseline**

Add a section to `README.md` after *The first run*, with a table comparing the two runs on this
one scene:

| | Threshold at 0 dB | Trained at 0.75 |
| --- | --- | --- |
| Detections | 16 | *from Step 3* |
| Matched to a declaration | 2 | *from Step 3* |
| Dark | 14 | *from Step 3* |
| Declarations searched | 12 | 12 |

State plainly what the evidence is worth: twelve declared vessels on one scene, and a mapping
whose two constants were chosen against the sea of the training set rather than against this
scene's answers — but reported on the same scene, so nothing here is a held-out number. The
held-out numbers are the LS-SSDD table above it, and they are the ones that carry weight.

Also note whether the stack of nine detections in one 200 m square in the north-west survives.
It is one bright object counted nine times by the stand-in; whether the model still returns it,
and how many times, is the most legible single difference between the two runs.

- [ ] **Step 5: Record the decisions**

Add three entries to `docs/decisions.md`, dated, in the file's existing voice:

1. *The window between decibels and amplitude is fitted to the sea, once, and shipped as a
   constant* — the reference measured in Task 1, the arithmetic, the floor being settled by the
   scene while the ceiling was not, and why a per-scene fit was rejected. Include the histogram
   verdict from Task 1 Step 4 and name the reading that was rejected.
2. *The chain cuts at the size the model was scored at* — 800 rather than 512, all nine tiles
   full-size, and why building the model at 512 was rejected even though it also resamples
   nothing.
3. *A hole is filled at sea level and guarded afterwards* — the two mechanisms, why the fill
   alone is not enough, why the floor at 0.0 was rejected, and why synthetic speckle was.

Also record in the same entry, or in a fourth: the Kaggle run the checkpoint came from, epoch 12,
its SHA-256, and that epoch 9 scored better and had already been deleted by `keep: 2`.

- [ ] **Step 6: Update the README's status line**

Mark Level 2 complete where the README tracks the levels, and update the closing paragraph of
*Training the detector*, which currently describes the decibel/amplitude gap as still open. It is
now closed, and it should point at the decision that closed it instead.

- [ ] **Step 7: Run everything and commit**

```bash
make lint && make test
git add configs/kattegat-lane.yaml docs/decisions.md README.md
git commit -m "feat: run the chain on the real scene with the trained detector"
```

- [ ] **Step 8: Close the ticket**

Check the five acceptance criteria on issue #10 and note in a comment which commit satisfies
each. #11 is unblocked by this.

---

## Self-Review

**Spec coverage.** Every section of the design maps to a task: the mapping and its derivation to
Tasks 2, 3 and 9; the hole decision to Tasks 4 and 7; the tile-size decision to Tasks 8 and 9;
the checkpoint build block to Tasks 6 and 7; `detections_from` to Task 5; the Kaggle pass to
Task 1; the baseline comparison and both documents to Task 9. The operating point of 0.75 is set
in Task 9's config with the argument the spec gives for it.

**Types.** `DecibelStretch(floor_db, ceiling_db, sea_db)`, `SeaReference(mean, spread)`,
`fit_window(*, sea_db, spread_db, reference)`, `sea_level(image) -> (median, sigma)`,
`without_holes(detections, image)`, `detections_from(output)`, `TrainedDetector(*, checkpoint,
stretch, score_threshold, tile_px, anchor_sizes, device)`, `trained_request_from(run_config,
relative_to)`, `check_tile_size(run_config, tiling)` — each name is used identically wherever it
appears.

**Known gap, deliberate.** Task 9 Step 1 leaves `floor_db` and `ceiling_db` as
`<Task 1 Step 5>`. This is not a placeholder standing in for a decision nobody has made; it is a
measurement that a human takes in Task 1 and pastes here, and the step says exactly where it
comes from. Everything else in the plan runs without it.
