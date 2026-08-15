"""What the chain exports, in the unit the detector was fitted on.

The chain hands out calibrated decibels, where this project's first real scene has its sea at
-21.84 dB. The model was fitted on 8-bit amplitude divided by 255. These are not the same
quantity, and the stretch LS-SSDD's authors used to turn one into the other is not recorded in
the dataset and cannot be recovered from it — so the mapping between them is chosen rather than
derived, and the choice is made by matching the sea. See `fit_window` and docs/decisions.md.

What ships is a fixed window in decibels, not a fit performed per scene. That distinction carries
the design. A window refitted on every acquisition is a percentile stretch under another name:
the same hull would take a different value under a different sea state, comparing two scenes
would compare two units, and a score threshold would stop meaning the same thing from one run to
the next. The fitting happens once, against a reference measured on the training set, and its
answer is two numbers in a config file where a reader can see both ends of the window at once.

Nothing here imports torch. The window, the sea estimate and the hole guard are exactly the
decisions that go wrong without a symptom — a scene converted through the wrong window does not
crash, it returns plausible detections in plausible places with scores — so they live on the side
of the seam a laptop tests in a second, beside `dataset.py` and `checkpoints.py`.
"""

from dataclasses import dataclass

import numpy as np

# Turns a median absolute deviation into the standard deviation of the Gaussian that would have
# produced it. Used instead of a plain standard deviation because a scene contains ships, and a
# ship stands forty decibels above the water it sits in: on kattegat-lane.tif the plain figure is
# 2.57 dB against 2.30 dB robustly, and the whole of that difference is the targets — which would
# then widen the very window that is supposed to make them stand out.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class DecibelStretch:
    """The window of decibels the model's 0..1 covers, and where a hole sits inside it.

    `sea_db` is not redundant with the two ends. It is where this product's sea stands, and it is
    what a nodata hole receives *before* the stretch is applied — so the fill follows the window
    automatically, and two numbers that have to agree cannot drift apart when one of them is
    edited.
    """

    floor_db: float
    ceiling_db: float
    sea_db: float

    def __post_init__(self) -> None:
        if self.ceiling_db <= self.floor_db:
            raise ValueError(
                f"the ceiling is {self.ceiling_db} dB and the floor {self.floor_db} dB; a window "
                "has to widen upwards, or it maps every pixel in the scene to the same value"
            )
        if not self.floor_db <= self.sea_db <= self.ceiling_db:
            raise ValueError(
                f"the sea is at {self.sea_db} dB, outside the window {self.floor_db} to "
                f"{self.ceiling_db} dB; a hole would then be filled at one end of the range "
                "rather than at the sea, which is the one thing this fill exists to avoid"
            )

    @property
    def sea(self) -> float:
        """Where the sea lands once the window is applied, and therefore what a hole comes back
        as. The number to compare against the reference the window was fitted to."""
        return (self.sea_db - self.floor_db) / (self.ceiling_db - self.floor_db)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """One scene or tile of decibels, as the amplitude in 0..1 the model was fitted on.

        Holes are filled before the stretch rather than after, and at the sea rather than at the
        floor. A NaN reaching the network is not a pixel it ignores — it propagates through every
        convolution that touches it and empties the whole tile, with no crash and no warning.
        `scene.py` writes nodata as NaN precisely because every comparison against NaN is false,
        which immunises a threshold detector; a network has no such immunity, and this is where
        that difference is paid for.

        Filling at the floor instead would be simpler to state and would move the problem rather
        than solve it: six per cent of the first real scene is nodata, and a perfectly black patch
        with a hard edge is a strong feature for a detector. Filling at the sea leaves almost no
        contrast at the boundary. What stops a hole being *reported* is `without_holes`, which is
        a separate mechanism on purpose.

        Returns a new array. The guard downstream reads the holes off the original, so the
        original keeps them.
        """
        filled = np.where(np.isnan(image), np.float32(self.sea_db), image)
        scaled = (filled - self.floor_db) / (self.ceiling_db - self.floor_db)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class SeaReference:
    """Where the sea stood in the images the model was fitted on, in the 0..1 it was handed.

    A property of the training set rather than of any scene the chain later reads, measured over
    the held-out tiles outside the annotated boxes and recorded in docs/decisions.md beside the
    run that measured it. It is the one thing about LS-SSDD's processing that *is* recoverable:
    the stretch its authors applied is gone, but the statistics the model was fitted under are
    still sitting in the pixels.
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

    Two moments, two parameters. Matching only the first would place the water correctly and
    leave the contrast between a hull and the sea it sits in scaled by an arbitrary factor, which
    is the half of the problem a detector actually keys on.

    Called to *derive* the constants a config then carries, not on a run. Fitting per scene would
    make this an adaptive stretch: the same hull would take a different value under a different
    sea state, and a score threshold would stop meaning the same thing between two acquisitions.
    See the module docstring.

    One end of the answer is settled by the scene alone and the other is not. The floor comes out
    near the sea less a few sigma whatever reference is supplied — on kattegat-lane.tif it lands
    within a decibel of -29 across every plausible reference — while the ceiling ranges over
    forty decibels. That asymmetry is the argument for measuring the reference rather than
    choosing a window by eye: by eye, one end would have been right and the other anywhere.
    """
    span = spread_db / reference.spread
    floor = sea_db - reference.mean * span
    return DecibelStretch(floor_db=floor, ceiling_db=floor + span, sea_db=sea_db)


def sea_level(image: np.ndarray) -> tuple[float, float]:
    """This scene's sea, as a median and a spread in decibels, ignoring the holes.

    Robust rather than the plain mean and standard deviation, for the reason `MAD_TO_SIGMA` is
    here: a scene contains ships, and a ship stands forty decibels above the water.

    This is the other half of the pair `fit_window` matches, and it is deliberately the same
    estimator the reference is measured with. Matching a robust spread against a plain one would
    fit the window to a difference in estimator rather than to a difference in sea.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError(
            "every pixel of this scene is nodata, so it has no sea to measure; the export "
            "covered no water, or the product's fill value was read as data"
        )

    median = float(np.median(finite))
    return median, float(np.median(np.abs(finite - median)) * MAD_TO_SIGMA)
