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

# Where the real scene's own distribution sits, so a candidate window can be judged here without
# the 22 MB product being attached. Measured on data/real/kattegat-lane.tif with the holes
# excluded, and only ever read — nothing below fits anything to them.
SCENE_QUANTILES = {"p1": -29.54, "p50": -21.84, "p99": -17.25, "p99.99": 2.13, "max": 27.08}

# What a ship is worth in decibels on that scene. Not chosen: these are the peak values of the
# only two detections the threshold baseline matched against a declared vessel — a 24 m hull and
# a 228 m one — read off outputs/kattegat-lane.gpkg. Two vessels is a thin anchor and it is
# written here as thin rather than dressed up.
SCENE_SHIP_DB = (8.20, 18.76)

# Every fifth held-out tile — 600 of them, some 380 million sea pixels. Two moments need far
# less; the whole split is not read because the reading is the slow part and nothing is gained.
STRIDE = 5

# How far outside an annotated box a pixel still counts as ship rather than sea. A hull's bright
# return spills past the box the annotator drew, and counting that spill as sea would widen the
# reference spread with exactly the signal the window exists to preserve.
MARGIN_PX = 4

# Below this, the window is not describing a radar scene. Open sea to a hull on Sentinel-1 spans
# something like forty decibels, and a reference that fits everything into less than half of that
# has been measured over something other than water. The first run of this returned 11 dB, from a
# held-out split whose coastal tiles were counted as sea.
MINIMUM_SPAN_DB = 20.0

# What counts as a tile rather than a label or a stray text file. LS-SSDD publishes JPEG; mirrors
# have been seen carrying PNG, and the reader takes whatever rasterio opens as 8-bit.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DATASET = Path("/kaggle/input/datasets/petrarodriguez/ls-ssdd-v1-0")
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

    Directories are found by what they *contain*, never by what they are called. The first
    version of this looked for a directory named `JPEGImages` and picked one that held a single
    subdirectory of the same name — the mirror nests every folder twice — so it reported a suffix
    of nothing and handed `catalogue` a directory where it expected a tile. A name is a hint; a
    file with an image extension in it is evidence.
    """
    from darkvessel.detect.dataset import Layout

    # One pass. Everything below is decided from which directories hold which kind of file, so
    # walking the tree twice to ask two questions about the same entries buys nothing.
    pictures: dict[Path, int] = {}
    suffixes: dict[Path, str] = {}
    labels: dict[Path, int] = {}
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        if entry.suffix.lower() in IMAGE_SUFFIXES:
            pictures[entry.parent] = pictures.get(entry.parent, 0) + 1
            suffixes.setdefault(entry.parent, entry.suffix)
        elif entry.suffix.lower() == ".xml":
            labels[entry.parent] = labels.get(entry.parent, 0) + 1

    if not pictures:
        raise FileNotFoundError(
            f"nothing under {root} has an image extension {sorted(IMAGE_SUFFIXES)}; this is not "
            "a copy of LS-SSDD, or it is still unpacking"
        )
    if not labels:
        raise FileNotFoundError(f"no .xml annotation anywhere under {root}")

    held_out = [d for d in pictures if "test" in d.name.lower()]
    images = max(held_out or list(pictures), key=lambda d: pictures[d])
    annotations = max(labels, key=lambda d: labels[d])

    for directory, count in sorted(pictures.items()):
        mark = "->" if directory == images else "  "
        print(f"  {mark} {directory.relative_to(root)}  {count:,} images")
    print(f"annotations {annotations.relative_to(root)}  ({labels[annotations]:,} xml)")
    print(f"suffix      {suffixes[images]}")

    return Layout(
        images=str(images.relative_to(root)),
        annotations=str(annotations.relative_to(root)),
        image_suffix=suffixes[images],
    )


def offshore(refs, root: Path):
    """Keep the tiles that are open water, and drop the ones that are coast.

    LS-SSDD's held-out half is not all sea. It is cut from whole Sentinel-1 acquisitions, so it
    contains harbours, shoreline and the structures on them, and the dataset says so itself by
    shipping `test_inshore.txt` beside `test_offshore.txt`.

    Masking the annotated boxes removes ships. It does not remove land, and land in SAR is bright
    — which is why the first run of this measured a "sea" whose spread was almost as large as its
    median, over a histogram that decayed monotonically to white. That reference fitted a window
    eleven decibels wide, against the forty that separate water from a hull, and it would have
    crushed a seventh of the real scene to black. See docs/decisions.md.

    Only the offshore half is measured, and that is the right half rather than merely the clean
    one: the window exists to place *this chain's* scene, which is open water in the Kattegat,
    where the model's sea was. Fitting it against a mixture of water and coast would match a
    distribution the target scene does not have.
    """
    listing = next((p for p in root.rglob("*.txt") if "offshore" in p.name.lower()), None)
    if listing is None:
        print("  no offshore listing under this root; measuring every held-out tile, coast and all")
        return refs

    names = {Path(token).stem for token in listing.read_text().split() if token.strip()}
    kept = [ref for ref in refs if ref.name in names]
    print(f"  {listing.name}: {len(names):,} names, {len(kept):,} of {len(refs):,} tiles matched")

    if not kept:
        print("  none matched, so the listing names its tiles some other way; measuring them all")
        return refs
    return kept


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
    held_out = offshore(held_out, root)
    if not held_out:
        # A mirror whose test directory holds scenes 1-10, or names them differently. Measuring
        # the training half instead is a defensible answer to the question being asked; measuring
        # nothing and reporting a window fitted to an empty histogram is not.
        print("  none of these scenes are held out; measuring every tile found instead")
        held_out = refs

    sampled = held_out[::STRIDE]
    histogram = np.zeros(256, dtype=np.int64)
    per_tile = []

    for ref in sampled:
        tile = ref.read()
        sea = np.ones(tile.image.shape, dtype=bool)
        for box in tile.boxes:
            sea[
                max(int(box.min_row) - MARGIN_PX, 0) : int(np.ceil(box.max_row)) + MARGIN_PX,
                max(int(box.min_col) - MARGIN_PX, 0) : int(np.ceil(box.max_col)) + MARGIN_PX,
            ] = False

        water = tile.image[sea]
        if water.size:
            per_tile.append(_one_tile(water, tile.image[~sea] if tile.boxes else None))
        histogram += np.bincount(np.rint(water * 255.0).astype(np.uint8), minlength=256)

    print(f"{histogram.sum():,} sea pixels over {len(sampled)} of {len(held_out)} held-out tiles")
    return histogram, np.array(per_tile)


def _one_tile(water: np.ndarray, hulls: np.ndarray | None) -> tuple[float, float, float, float]:
    """Where this tile's sea sits, how much it varies, how lopsided it is, and how bright its
    ships are.

    The fourth number is the one the second was supposed to do the job of. Matching the *spread*
    of the sea sets the width of the window from how grainy the water is, and the two products
    are not equally grainy: LS-SSDD's sea has a relative spread near 0.8, this chain's Sentinel-1
    GRD near 0.27 in the same units, which is a difference in how many looks were averaged rather
    than a difference in how the bytes were made. Fitting a window to that fits it to
    multi-looking.

    A hull is the other thing a detector sees, it is annotated on this side, and it was measured
    on the real scene by the threshold baseline. Two anchors, both about brightness, neither
    about grain.

    p95 rather than the maximum: a maximum is one pixel and JPEG has been over it.

    Measured within the tile and never across tiles, because the number this is matched against —
    a scene's own spread in decibels — is a within-scene quantity. Five LS-SSDD acquisitions
    pooled into one heap carry their differences in sea state, incidence angle and calibration
    inside the spread, and matching that against one scene's speckle fits the window to a
    difference between acquisitions rather than to the texture of water. It is the same mistake
    as matching a robust spread against a plain one, wearing a different hat: the first run of
    this measured 0.157 pooled and refused a window of 14.7 dB because of it.

    The third number is Bowley's skew, from the quartiles. It answers the question the histogram
    is drawn for: radar amplitude over homogeneous water is Rayleigh-like and leans right, and a
    logarithmic stretch straightens it out. Near zero says the source was already in decibels;
    clearly positive says it was linear amplitude, and the mapping has to invert through
    sigma-nought before the affine.
    """
    first, middle, third = np.percentile(water, [25, 50, 75])
    spread = float(np.median(np.abs(water - middle)) * 1.4826)
    lean = float((third + first - 2 * middle) / (third - first)) if third > first else 0.0
    ship = float(np.percentile(hulls, 95)) if hulls is not None and hulls.size else np.nan
    return float(middle), spread, lean, ship


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
    histogram, per_tile = sea_histogram(root, layout)
    middles, spreads, leans, ships = (per_tile[:, index] for index in range(4))

    # Median of the per-tile figures, not the figure of the pooled pixels. See `_one_tile`.
    sea, spread = float(np.median(middles)), float(np.median(spreads))
    pooled_sea, pooled_spread = moments(histogram)
    hull, lean = float(np.nanmedian(ships)), float(np.median(leans))

    print(f"  sea, within a tile: median {sea:.4f}, robust spread {spread:.4f}  (of 1.0)")
    print(f"  sea, pooled:        median {pooled_sea:.4f}, robust spread {pooled_spread:.4f}")
    print(
        f"  tile medians run {middles.min():.4f} to {middles.max():.4f} — the difference between "
        "acquisitions, which pooling counts as sea texture and a tile-by-tile figure does not"
    )
    print(
        f"  ships, p95 inside their boxes: {hull:.4f}, over {int(np.isfinite(ships).sum())} tiles"
    )
    shape = "right-skewed, so the source was linear" if lean > 0.10 else "near symmetric"
    print(f"  lean (Bowley, median over tiles) {lean:+.3f}  ->  {shape}")

    print("\n== the shape, pooled over every sampled tile ==")
    sketch(histogram)

    print("\n== candidates, and where this scene's own quantiles land under each ==")
    candidates = {
        "sea level + sea spread": window(sea, spread),
        "sea level + ship brightness": by_anchors(sea, hull),
    }
    for name, (floor, ceiling) in candidates.items():
        placed = "  ".join(
            f"{label} {(db - floor) / (ceiling - floor):+.2f}"
            for label, db in SCENE_QUANTILES.items()
        )
        narrow = "   <-- implausible" if ceiling - floor < MINIMUM_SPAN_DB else ""
        print(f"  {name}")
        print(f"    {floor:7.2f} .. {ceiling:7.2f} dB   span {ceiling - floor:5.1f} dB{narrow}")
        print(f"    {placed}")

    print(
        f"\n(sea {sea:.4f} +/- {spread:.4f} within a tile, ship p95 {hull:.4f}, lean {lean:+.3f}, "
        f"over {len(per_tile)} tiles — send all of it back, the window is chosen from it)"
    )

    return candidates


def by_anchors(sea: float, hull: float) -> tuple[float, float]:
    """The window that puts this scene's water at LS-SSDD's water, and its hulls at LS-SSDD's.

    Two points, both of them brightnesses and neither of them a variance — which is the whole
    reason for it. What a detector separates is a hull from the water around it, so those are the
    two levels that have to line up, and the width of the window falls out of them instead of
    being set by whichever product happened to be averaged over more looks.

    The ship end of the real scene is the mean of the two vessels the threshold baseline matched
    against a declaration. Two hulls on one acquisition is the thinnest part of this argument,
    and it is written down as thin rather than dressed up.
    """
    ship_db = sum(SCENE_SHIP_DB) / len(SCENE_SHIP_DB)
    span = (ship_db - SCENE_SEA_DB) / (hull - sea)
    floor = SCENE_SEA_DB - sea * span
    return floor, floor + span


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
