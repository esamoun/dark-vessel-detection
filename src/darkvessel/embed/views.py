"""Two views of one crop, and what a view is allowed to change.

A contrastive representation is learned by asking what two views of the same object have in
common. Which transformations are allowed to stand between them is therefore not a detail of the
training loop — it is the statement of what the representation is asked to ignore, and it is the
whole of the supervision here, because there are no labels.

Radar amplitude narrows the choice sharply. Colour and contrast jitter have no physical meaning
on a backscatter coefficient: a hull is bright because it scatters, not because of an exposure,
and a network told to ignore a shift in decibels is told to ignore the one measurement the image
carries. So what a view may do is the eight symmetries of the square, a translation of a few
pixels, and speckle.

Speckle is the radar-native one, and it is the reason this module exists rather than
`dataset.py`'s eight symmetries being reused as they stand. `dataset.py` says of it: "it needs a
speckle model to be argued for, and that belongs with the rest of the work on what this data
actually is." This is that work. A multi-looked intensity image carries a multiplicative
fluctuation that is Gamma distributed with shape equal to the number of looks, so a second look
at the same sea is the same scene times a draw from that distribution. In decibels the
multiplication is an addition, which is what `Speckle` applies — and the number of looks is
measured off the scene rather than quoted from the product specification, by `looks_of`.

The translation needs the margin `crops.py` cuts. Rolling a crop around its own edge would put a
piece of one side of a vessel against the other; reaching past a stored crop would invent pixels.
Taking a different window of a larger square does neither, which is why the archive stores more
than the encoder is shown.

No torch here, deliberately, like everything else that decides what a model is shown.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from darkvessel.detect.amplitude import sea_level
from darkvessel.detect.dataset import SYMMETRIES, LabelledTile

# What a Sentinel-1 IW GRDH product is nominally multi-looked to. Not used as the answer — the
# shipped configuration carries what `looks_of` measured on the scene the archive is built from —
# but kept here as the figure a measurement is sane against. See docs/decisions.md.
NOMINAL_LOOKS = 4.4

# Where the sea ends, for the purpose of measuring speckle, in robust standard deviations above
# its own median. A ship stands forty decibels above the water it sits in — some seventeen of
# these — so this excludes every target while cutting away three hundredths of one per cent of the
# speckle distribution itself. A percentile would be the obvious alternative and is wrong here:
# cutting the brightest tenth of a sea removes most of what speckle *is*, and on a synthetic sea
# built at 4.4 looks it reports 6.6.
_SEA_SIGMAS = 4.0


@dataclass(frozen=True)
class Speckle:
    """A second look at the same scene, as the sensor's own fluctuation would have produced it.

    `looks` is the equivalent number of looks: the shape of the Gamma distribution whose mean is
    one and whose relative variance is `1 / looks`. Small values shake a pixel hard, large ones
    barely at all, and the limit at infinity is no augmentation.

    Applied in decibels, which is the unit the chain deals in, so the multiplication a speckle
    model describes arrives here as an addition.
    """

    looks: float

    def __post_init__(self) -> None:
        if self.looks <= 0:
            raise ValueError(
                f"{self.looks} looks describes no measurement; the number of looks is the shape "
                "of a Gamma distribution and has to be positive"
            )

    def __call__(self, crop_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One realisation of `crop_db` under a second look, in decibels.

        Holes stay holes: NaN plus anything is NaN, which is what should happen — a pixel the
        product never measured does not acquire a measurement by being looked at again.
        """
        gain = rng.gamma(shape=self.looks, scale=1.0 / self.looks, size=crop_db.shape)
        return (crop_db + 10.0 * np.log10(gain)).astype(np.float32)


def looks_of(image: np.ndarray) -> float:
    """The equivalent number of looks in this scene's sea, measured rather than quoted.

    The definition is the one radar uses: over a homogeneous surface, the ratio of the squared
    mean of intensity to its variance. It is computed in intensity and not in decibels, because
    the ratio is a statement about a multiplicative fluctuation and taking a logarithm first
    measures a different quantity.

    What counts as homogeneous is decided by `_SEA_SIGMAS`, against the robust median and spread
    `amplitude.sea_level` already measures for the stretch — the same estimator, so the two
    numbers this project quotes about a sea are quotes about the same sea. Holes are left out too:
    a nodata pixel is not a calm sea, it is no measurement.

    What comes back is the variability of this sea and not only the processor's looks, and the
    archive says so plainly: across fifty acquisitions of the same rectangle the figure runs from
    0.01 to 5.14, with a median of 4.12 against the nominal 4.4. The low end is not more speckle,
    it is less sea — a calm scene backscatters at -37 dB, close enough to the noise floor that
    its relative variation in decibels is five times that of a windy one. Measured per scene,
    reported by `darkvessel crops`, and the figure a config carries is the median rather than any
    one acquisition's. See docs/decisions.md.
    """
    median, spread = sea_level(image)

    finite = image[np.isfinite(image)]
    sea_db = finite[finite <= median + _SEA_SIGMAS * spread].astype(np.float64)
    intensity = np.power(10.0, sea_db / 10.0)
    variance = float(np.var(intensity))
    if variance == 0.0:
        raise ValueError(
            "this scene's sea has no variation in it at all, so it carries no speckle to "
            "measure; a synthetic scene is the usual reason"
        )

    return float(np.mean(intensity)) ** 2 / variance


def rng_for(*parts: object) -> np.random.Generator:
    """A generator named by what it is drawing for, rather than drawn from a shared stream.

    The convention `dataset.symmetry_for` already applies, for the reasons it states: a resumed
    session has to reproduce the augmentation the interrupted one was applying, or the two halves
    of a run are two experiments; and a loader with several workers draws from several streams, so
    a global generator lets the order crops arrive in decide what happens to them. A hash of the
    names depends on neither.
    """
    key = ":".join(str(part) for part in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def view(
    crop: np.ndarray,
    *,
    crop_px: int,
    speckle: Speckle | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """One view of one stored crop: a window of it, laid down one of eight ways, looked at again.

    The window comes first and the symmetry second, so that the translation is a translation of
    the object rather than of whichever corner a rotation has just moved into view. The order is
    arbitrary in what it can express and not arbitrary in what it means, so it is fixed here.
    """
    stored = crop.shape[-1]
    if crop_px > stored:
        raise ValueError(
            f"a view of {crop_px} px cannot be taken from a crop of {stored} px without "
            "resampling it, and resampling radar amplitude is a decision"
        )

    slack = stored - crop_px
    top, left = (int(rng.integers(slack + 1)) for _ in range(2))
    window = crop[top : top + crop_px, left : left + crop_px]

    laid = laid_down(window, int(rng.integers(len(SYMMETRIES))))
    return laid if speckle is None else speckle(laid, rng)


def laid_down(crop: np.ndarray, symmetry: int) -> np.ndarray:
    """One of the eight ways a square can be laid down without changing any pixel value.

    Through `dataset.SYMMETRIES` rather than a second enumeration of the same eight: the detector
    and the representation are augmented under one definition of what a symmetry is, and a crop
    with no boxes in it is a `LabelledTile` with no boxes in it.
    """
    tile = SYMMETRIES[symmetry](LabelledTile(name="", image=crop, boxes=()))
    return np.ascontiguousarray(tile.image)
