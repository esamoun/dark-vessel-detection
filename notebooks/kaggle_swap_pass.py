"""The one Kaggle pass the swap ticket needs: measure the sea, and package the weights.

Companion to `kaggle-train.ipynb`, and run once. It answers the question `docs/decisions.md` left
open on 2026-08-14 — the model is fitted on 8-bit amplitude and the chain feeds it decibels, and
the stretch between the two is not recorded in the dataset and cannot be recovered from it. What
*can* be recovered is where the sea sat in the images the model actually saw, and that is two
numbers, which is exactly enough to fix an affine.

Run it in a session with two data sources attached:

  * the LS-SSDD dataset (`ls-ssdd-v10`);
  * the output of the notebook version that trained, which is where `epoch-012.pt` survives.

Paste the whole file into one cell, or `%run` it. It prints three things:

  1. a build-block-carrying, optimiser-stripped copy of the checkpoint, and its digest;
  2. the LS-SSDD sea as a median and a robust spread, with the histogram that says whether the
     window belongs on decibels or on linear amplitude;
  3. the window itself — floor and ceiling in decibels — to paste into
     `configs/kattegat-lane.yaml`.

Nothing here is imported by the chain and nothing here runs in CI. It is a measurement script,
and its output is a decision recorded in docs/decisions.md.
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------
# What this run is measured against. Both numbers come from data/real/kattegat-lane.tif and are
# reproduced by `darkvessel.detect.amplitude.sea_level`, which is the tested implementation:
#
#   python3 -c "from darkvessel.data.scene import Scene; \
#               from darkvessel.detect.amplitude import sea_level; from pathlib import Path; \
#               print(sea_level(Scene.from_geotiff(Path('data/real/kattegat-lane.tif')).image))"
#
# Robust rather than plain, because the scene holds ships and a ship stands forty decibels above
# the water: the plain spread is 2.57 dB and the whole of the difference is the targets.
SCENE_SEA_DB = -21.84
SCENE_SPREAD_DB = 2.30

# Every fifth held-out tile — 600 of them, some 380 million sea pixels. Two moments need far
# less; the whole split is not read because the reading is the slow part and nothing is gained.
STRIDE = 5

# How far outside an annotated box a pixel still counts as ship rather than sea. A hull's bright
# return spills past the box the annotator drew, and counting that spill as sea would widen the
# reference spread with exactly the signal the window exists to preserve.
MARGIN_PX = 4

DATASET = Path("/kaggle/input/ls-ssdd-v10/LS-SSDD-v1.0-OPEN")
PACKAGED = Path("/kaggle/working/epoch-012.pt")

# What built the weights this pass packages, read off configs/train.yaml at seed 20260814. The
# training run that produced them predates train.py writing its own build block, so it is stated
# here — once, in the file that does the packaging. Anchors leave no trace in a state dict, so a
# checkpoint without this can be loaded into a detector looking for the wrong size of ship.
BUILT = {
    "tile_px": 800,
    "anchor_sizes": ((32,), (64,), (128,), (256,), (512,)),
    "seed": 20260814,
    "pretrained": True,
    "trainable_backbone_layers": 3,
}


def find_checkpoint(epoch: int = 12) -> Path:
    """Where the training session's output landed this time.

    Searched rather than hard-coded: the attachment path carries the notebook slug, which differs
    between one person's copy of the run and another's.
    """
    found = subprocess.run(
        ["find", "/kaggle/input", "-name", f"epoch-{epoch:03d}.pt"],
        capture_output=True,
        text=True,
    ).stdout.split()

    if not found:
        raise FileNotFoundError(
            f"no epoch-{epoch:03d}.pt under /kaggle/input. Attach the output of the notebook "
            "version that trained; if it was never saved as a version, the weights went with the "
            "session and the run has to be repeated"
        )
    return Path(found[0])


def package(source: Path, out: Path = PACKAGED) -> str:
    """Strip the optimiser, add the build block, and return the digest of what to download.

    The momentum buffers are the same size as the weights and are useless for inference, so
    dropping them halves the download. `TrainedDetector` reads `model` and `built` and nothing
    else, so the stripped file loads exactly as the original would.
    """
    import torch

    state = torch.load(source, map_location="cpu", weights_only=True)
    torch.save({"epoch": state["epoch"], "model": state["model"], "built": BUILT}, out)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out} — {out.stat().st_size / 1e6:.0f} MB, from {source}")
    print(f"  sha256 {digest}")
    print("  download it from the notebook's Output tab, then: mv ~/Downloads/epoch-012.pt models/")
    return digest


def discover(root: Path):
    """Work out how this copy of LS-SSDD is laid out, and say so out loud.

    The published set is `JPEGImages/` beside `Annotations/`, which is what `dataset.LS_SSDD`
    describes. Mirrors are not: the one this was first run against splits the images into
    `JPEGImages_sub_train` and `JPEGImages_sub_test` and nests the annotations two deep. A path
    that does not match returns no tiles, and `catalogue` refuses rather than reporting an empty
    dataset — but it cannot tell anyone which directory to use instead.

    So this looks, and prints what it found. It guesses nothing in silence: every choice below is
    on the console beside the alternatives it was chosen over, and a mirror this cannot read is
    one the caller passes a `Layout` for by hand.

    The test half is preferred where the images are split, because the held-out scenes are the
    half this project reports its numbers over. Either half would answer the question — what the
    sea looks like in this dataset — but only one of them is the half the model was scored on.
    """
    from darkvessel.detect.dataset import Layout

    candidates = sorted(d for d in root.rglob("*") if d.is_dir() and "JPEGImages" in d.name)
    if not candidates:
        raise FileNotFoundError(f"no directory with 'JPEGImages' in its name under {root}")

    images = next((d for d in candidates if "test" in d.name.lower()), candidates[0])
    labels = next(root.rglob("*.xml"), None)
    if labels is None:
        raise FileNotFoundError(f"no .xml annotation anywhere under {root}")

    suffix = next(
        (f.suffix for f in sorted(images.iterdir()) if f.suffix.lower() != ".xml"), ".jpg"
    )

    print(f"images      {images.relative_to(root)}   (of {[d.name for d in candidates]})")
    print(f"annotations {labels.parent.relative_to(root)}")
    print(f"suffix      {suffix}")

    return Layout(
        images=str(images.relative_to(root)),
        annotations=str(labels.parent.relative_to(root)),
        image_suffix=suffix,
    )


def sea_histogram(root: Path = DATASET, layout=None) -> np.ndarray:
    """The 8-bit values of every sea pixel in the sampled held-out tiles, as 256 exact bins.

    Held out rather than training, because the training half is what the model was fitted *on*
    and the question here is what the sea looks like in this dataset generally. Either would do;
    the held-out half is the one this project already reports numbers over.

    Uses the repository's own catalogue and reader, so the split, the counting convention and the
    8-bit conversion are the tested ones rather than a second implementation that agrees with
    them until it does not.
    """
    from darkvessel.detect.dataset import catalogue, split_by_scene

    refs = catalogue(root, layout or discover(root))
    scenes = sorted({ref.scene for ref in refs})
    print(f"{len(refs):,} tiles, cut from scenes {scenes}")

    _, held_out = split_by_scene(refs)
    if not held_out:
        # A mirror whose test directory holds scenes 1-10, or names them differently. Measuring
        # the training half instead is a defensible answer to the question being asked; measuring
        # nothing and reporting a window fitted to an empty histogram is not.
        print("  none of these scenes are held out; measuring every tile found instead")
        held_out = refs

    sampled = held_out[::STRIDE]
    histogram = np.zeros(256, dtype=np.int64)

    for ref in sampled:
        tile = ref.read()
        sea = np.ones(tile.image.shape, dtype=bool)
        for box in tile.boxes:
            sea[
                max(int(box.min_row) - MARGIN_PX, 0) : int(np.ceil(box.max_row)) + MARGIN_PX,
                max(int(box.min_col) - MARGIN_PX, 0) : int(np.ceil(box.max_col)) + MARGIN_PX,
            ] = False

        histogram += np.bincount(np.rint(tile.image[sea] * 255.0).astype(np.uint8), minlength=256)

    print(f"{histogram.sum():,} sea pixels over {len(sampled)} of {len(held_out)} held-out tiles")
    return histogram


def quantile(histogram: np.ndarray, at: float) -> int:
    """An exact quantile. 8-bit values in 256 bins lose nothing, so this is not an estimate."""
    return int(np.searchsorted(np.cumsum(histogram), at * histogram.sum()))


def moments(histogram: np.ndarray) -> tuple[float, float]:
    """The sea's median and robust spread, in the 0..1 the reader hands the model.

    The spread is a median absolute deviation scaled to the standard deviation of the Gaussian
    that would produce it — the same estimator `amplitude.sea_level` applies to a scene, so the
    two sides of the match are measured the same way. Matching a robust spread against a plain
    one would fit the window to a difference in estimator rather than to a difference in sea.
    """
    median = quantile(histogram, 0.5)

    folded = np.zeros(256, dtype=np.int64)
    for value, count in enumerate(histogram):
        folded[abs(value - median)] += count

    return median / 255.0, quantile(folded, 0.5) * 1.4826 / 255.0


def sketch(histogram: np.ndarray, bars: int = 32) -> None:
    """The shape, which is what decides decibels against linear amplitude.

    A hump sitting away from zero and roughly symmetric means the source stretch was log-like:
    the affine belongs on decibels, and the window below is the answer. A spike jammed against
    the first bars with a long thin tail to the right means it was linear, and the mapping has to
    invert through sigma-nought first. Whichever it is, the other one gets written down as
    rejected — a decision nobody can see the evidence for is a guess with a paragraph attached.
    """
    coarse = histogram.reshape(bars, 256 // bars).sum(axis=1)
    width = 256 // bars

    for index, count in enumerate(coarse):
        filled = int(60 * count / coarse.max())
        print(f"  {index * width:3d}-{index * width + width - 1:3d} {'#' * filled:<60} {count:,}")


def window(mean: float, spread: float) -> tuple[float, float]:
    """The floor and ceiling that put this scene's sea where the model's sea was.

    The three lines `amplitude.fit_window` implements, written out because this script runs in a
    session where the package may not be installed yet. Two moments, two parameters.

    The floor lands near -29 dB whatever the reference turns out to be — it is the sea less a few
    sigma, and the scene settles it alone. The ceiling is the number this whole pass exists to
    fix, and it ranges over forty decibels across plausible references. Choosing the window by eye
    would have got one end right and the other anywhere.
    """
    span = SCENE_SPREAD_DB / spread
    floor = SCENE_SEA_DB - mean * span
    return floor, floor + span


def measure(root: Path = DATASET, layout=None) -> tuple[float, float]:
    """The half of this pass that can only be done here, and the only half that is a measurement.

    Returns the window. Separate from `main` because the two halves are independent: the weights
    are a file that can be fetched any number of ways, and once they are on a disk this part is
    still undone. A session that already has the checkpoint runs this alone.

    `layout` is for a mirror `discover` cannot read. Passing one skips the looking entirely.
    """
    print("== the sea the model was fitted on ==")
    histogram = sea_histogram(root, layout)
    mean, spread = moments(histogram)

    print(f"  median {mean:.4f}, robust spread {spread:.4f}  (of 1.0)")
    print(
        f"  p1 {quantile(histogram, 0.01)}  p25 {quantile(histogram, 0.25)}  "
        f"p75 {quantile(histogram, 0.75)}  p99 {quantile(histogram, 0.99)}  (of 255)"
    )

    print("\n== the shape: a hump away from zero means decibels, a spike at zero means linear ==")
    sketch(histogram)

    floor, ceiling = window(mean, spread)
    print("\n== paste into configs/kattegat-lane.yaml, run.trained.stretch ==")
    print(f"      floor_db: {floor:.2f}")
    print(f"      ceiling_db: {ceiling:.2f}")
    print(f"      sea_db: {SCENE_SEA_DB}")
    print(f"\n(reference sea {mean:.4f} +/- {spread:.4f}, for docs/decisions.md)")

    return floor, ceiling


def main(weights: bool = True) -> None:
    """Both halves, with the weights skipped where they have already been fetched.

    A missing checkpoint is reported and stepped over rather than raised. The measurement is the
    part of this pass that needs the dataset attached and cannot be repeated later on a laptop;
    letting a file that is already downloaded stop it would be the wrong way round.
    """
    if weights:
        print("== weights ==")
        try:
            package(find_checkpoint())
        except FileNotFoundError as absent:
            print(f"  skipped: {absent}")
        print()

    measure()


if __name__ == "__main__":
    main()
