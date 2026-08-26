"""Nearest-neighbour retrieval over the archive, and the check that it means anything.

"Show me the detections that look like this one" is the question this level exists to answer, and
it is the question that can most easily be answered wrongly while looking right. A representation
that has collapsed — every crop mapped to the same point — returns neighbours for every query,
ranked, with similarities near one. So retrieval is not shipped on the strength of a plausible
contact sheet: it is shipped with two numbers beside it, and both of them are computed here.

The first needs no labels at all, which is the point. Take a crop, take a second view of the same
crop through the same augmentations the training used, and ask where the twin ranks among the
whole archive. A representation that has learned an object puts it first. A collapsed one ranks
it at chance, and so does one that has learned the sea. This is the check that fails loudest and
costs nothing to run, so it is recorded every epoch rather than at the end.

The second is a property of the pixels rather than of the embedding: how far the bright thing in
the middle of a crop extends. It is not an independent label — it is measured from the same
pixels the encoder saw, and this file says so rather than dressing it as ground truth. What it
answers is narrower and still worth having: do the neighbours a query returns agree about the
size of the object more than a detection drawn at random from the archive does?

The contact sheet is the third thing, and it is the only one aimed at a person. Two numbers can
be satisfied by a representation that has learned something real and useless; a reader looking at
six queries and their neighbours can see in a second what neither number says.

Nothing here imports torch, and that is load-bearing rather than incidental: a ranking, a chance
level and the equivalence a check counts under are exactly the decisions that go wrong without a
symptom, so they sit beside `crops.py` and `views.py` on the side of the seam a laptop tests in a
second. The training loop imports this module rather than the other way round, which is what lets
the number recorded every epoch be the same number the check reports at the end.
"""

import base64
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from darkvessel.detect.amplitude import DecibelStretch, sea_level
from darkvessel.embed.crops import centre

# How far above its own sea a pixel has to stand to count as part of the object in the middle of
# a crop. Four robust standard deviations, the figure `views.looks_of` uses to decide the same
# question the other way round — there to exclude the targets, here to find them.
_TARGET_SIGMAS = 4.0


def unit(vectors: np.ndarray) -> np.ndarray:
    """Vectors scaled to unit length, so that a dot product is a cosine.

    A vector of length zero stays a vector of length zero rather than becoming a NaN. It is the
    signature of a representation that has collapsed onto the origin, and the checks below should
    report that as a bad answer rather than as no answer.
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


@dataclass(frozen=True)
class Neighbour:
    """One retrieved crop, how close it stands to what was asked for, and what to call it."""

    index: int
    similarity: float
    name: str


@dataclass(frozen=True)
class Retrieved:
    """One query and what came back with it, in one object rather than in two parallel lists.

    The names travel with the indices because everything downstream needs both — the table prints
    the names, the contact sheet draws the crops the indices point at, and a sheet whose captions
    have drifted one row from its images is a figure that argues for the wrong thing.
    """

    query: int
    name: str
    found: tuple[Neighbour, ...]

    def indices(self) -> list[int]:
        """The query and its neighbours, in the order a row of the sheet draws them."""
        return [self.query, *(neighbour.index for neighbour in self.found)]


def neighbours(vectors: np.ndarray, query: int, count: int) -> list[int]:
    """The `count` archive entries most like `query`, nearest first, excluding the query itself.

    Cosine similarity, which is the geometry the contrastive loss was written in: NT-Xent scores
    normalised vectors against one another, so ranking by anything else would rank by a quantity
    the training never optimised.
    """
    return [index for index, _ in _ranked(vectors, query, count)]


def retrieve(vectors: np.ndarray, names: Sequence[str], query: int, count: int) -> Retrieved:
    """One query, its neighbours, and the names of both."""
    if len(names) != len(vectors):
        raise ValueError(f"{len(names)} names for {len(vectors)} vectors")

    return Retrieved(
        query=query,
        name=names[query],
        found=tuple(
            Neighbour(index=index, similarity=similarity, name=names[index])
            for index, similarity in _ranked(vectors, query, count)
        ),
    )


def _ranked(vectors: np.ndarray, query: int, count: int) -> list[tuple[int, float]]:
    """Indices and cosine similarities, nearest first, with the query itself taken out."""
    if not 0 <= query < len(vectors):
        raise ValueError(f"no crop {query} in an archive of {len(vectors)}")

    similarity = unit(vectors) @ unit(vectors)[query]
    similarity[query] = -np.inf
    return [(int(index), float(similarity[index])) for index in np.argsort(-similarity)[:count]]


def _nearest(vectors: np.ndarray, keeping: np.ndarray | None = None) -> np.ndarray:
    """Each crop's nearest neighbour, as one index per row.

    `keeping` is a square of booleans marking which neighbours a row is allowed to retrieve, and
    the checks below differ only in what they pass here: everything but the query itself, or
    everything that is not the query's own object. It is one function rather than the same
    argmax written in three places, because the three have to agree about what "nearest" means.
    """
    similarity = unit(vectors) @ unit(vectors).T
    np.fill_diagonal(similarity, -np.inf)
    if keeping is not None:
        similarity = np.where(keeping, similarity, -np.inf)

    return np.argmax(similarity, axis=1)


def twin_recall(vectors: np.ndarray, twins: np.ndarray, same_as: np.ndarray | None = None) -> float:
    """How often a second view of a crop retrieves that crop, or another cut of the same object.

    One number in 0..1, against the chance level `chance_of` computes for the same equivalence.
    The twins are embedded from views the encoder was not shown while it trained, which is what
    makes it a check rather than a restatement of the loss.

    `same_as` says which crops count as the same object — `Archive.co_located` builds it, and
    without it only a crop itself does. It is not a leniency added to flatter the number. A
    detector run at an archive's operating point cuts a large hull several times, and a ranking
    that called the second cut a wrong answer would be measuring how duplicated the archive is.
    """
    if vectors.shape != twins.shape:
        raise ValueError(f"{len(twins)} twins against {len(vectors)} crops; they must pair up")
    if len(vectors) < 2:
        raise ValueError("a single crop is its own nearest neighbour whatever the encoder does")

    equivalent = np.eye(len(vectors), dtype=bool) if same_as is None else same_as
    # Not `_nearest`: a twin ranks against the crop's own vector too, and that is the answer it
    # is meant to find. The diagonal it would blank out is the whole point of the check.
    first = np.argmax(unit(twins) @ unit(vectors).T, axis=1)
    return float(np.mean(equivalent[np.arange(len(vectors)), first]))


def chance_of(same_as: np.ndarray, *, excluding_query: bool = False) -> float:
    """What a representation with nothing in it would score against this equivalence.

    Two readings, because the two checks rank over different things and a baseline that is right
    for one is wrong for the other by a factor of two. A twin is ranked against every crop in the
    archive, its own included — a twin is a second view and not the crop's own vector — so a
    random ranking is right with probability `|same as i| / n`. Retrieval takes the query out
    before ranking, so a random one is right with probability `(|same as i| - 1) / (n - 1)`:
    the query is no longer a candidate and neither is it a way to be right.
    """
    count = len(same_as)
    if excluding_query:
        return float(np.mean((same_as.sum(axis=1) - 1) / (count - 1)))
    return float(np.mean(same_as.sum(axis=1) / count))


def extent(crops: np.ndarray, crop_px: int) -> np.ndarray:
    """How many pixels of each crop stand above its own sea: the size of the thing in the middle.

    Measured on the window the encoder is shown rather than on the stored square, so a second
    object sitting in the margin does not count towards the first one's size. Robustly, against
    the median and spread `amplitude.sea_level` measures, because a crop of a 274 m hull is a
    crop whose plain standard deviation is mostly hull.
    """
    windows = centre(crops, crop_px)
    counted = []
    for window in windows:
        median, spread = sea_level(window)
        counted.append(int(np.sum(window > median + _TARGET_SIGMAS * spread)))
    return np.array(counted, dtype=np.int64)


@dataclass(frozen=True)
class Agreement:
    """Whether the nearest *different* object agrees about size more than chance does.

    Different, and that qualification is the whole value of the figure. Two thirds of the nearest
    neighbours in this archive are another cut of the query's own hull, so a version of this
    measured over them would restate the duplication `docs/failures.md` records rather than say
    anything about resemblance between objects. Ranked over everything the query is not, it is the
    one number that speaks to the claim the ticket actually makes: that retrieval returns visually
    similar objects.

    Both figures are median absolute differences in the pixel count `extent` measures, so smaller
    is closer. `chance` is what the same measurement gives against a crop drawn at random from the
    archive, which is the only baseline that makes `retrieved` a number rather than a size.
    """

    retrieved: float
    chance: float

    def line(self) -> str:
        return (
            f"the nearest different object differs by {self.retrieved:.1f} px of target against "
            f"{self.chance:.1f} px for a crop drawn at random"
        )


def agreement(
    vectors: np.ndarray, sizes: np.ndarray, same_as: np.ndarray | None = None, seed: int = 0
) -> Agreement:
    """How closely the nearest different object matches a query's apparent size, against a draw.

    Not a label: `sizes` is measured from the pixels the encoder was shown, and this file says so
    rather than dressing it as ground truth. It is reported for what it rules out rather than for
    what it proves — a representation whose neighbours are no closer in size than a random crop
    has not learned the object.

    A crop with no different object to retrieve is left out of both figures rather than counted
    against either. There is no such crop in this archive, and an archive of one heavily cut
    vessel would otherwise report the size agreement of an argmax over nothing.
    """
    allowed = np.ones((len(vectors), len(vectors)), dtype=bool) if same_as is None else ~same_as
    np.fill_diagonal(allowed, False)
    measurable = allowed.any(axis=1)
    if not measurable.any():
        raise ValueError(
            "every crop here is the same object as every other, so there is no different object "
            "to retrieve and this figure would be a median of nothing"
        )

    nearest = _nearest(vectors, allowed)[measurable]
    drawn = np.random.default_rng(seed).permutation(len(vectors))[measurable]
    theirs = sizes[measurable]

    return Agreement(
        retrieved=float(np.median(np.abs(theirs - sizes[nearest]))),
        chance=float(np.median(np.abs(theirs - sizes[drawn]))),
    )


@dataclass(frozen=True)
class SameObject:
    """What the nearest neighbour of a detection turns out to be.

    Three shares of the same ranking, and the third is the one to read first. `retrieved` is how
    often the nearest neighbour is another cut of the object the query is — the strongest
    agreement a representation of an object can show, because two cuts of one hull are the most
    similar pair the archive contains. `elsewhere` is how often it is a different object in the
    *same* acquisition, and that is the diagnostic this level most needs: the window between
    decibels and amplitude is fixed across the archive and the sea under it is not, running from
    -37 dB on a calm morning to -11 dB in a blow. A representation that keyed on that would return
    beautiful neighbours, all of them from one acquisition, and would have learned the weather.
    """

    retrieved: float
    chance: float
    elsewhere: float

    def line(self) -> str:
        return (
            f"{self.retrieved:.0%} of nearest neighbours are another cut of the query's own "
            f"object, against {self.chance:.1%} at chance; {self.elsewhere:.0%} are a different "
            "object in the same acquisition"
        )


def same_object(vectors: np.ndarray, same_as: np.ndarray, scenes: Sequence[str]) -> SameObject:
    """Whether retrieval is returning the same object, a different one, or the same weather."""
    if not len(same_as) == len(scenes) == len(vectors):
        raise ValueError("the vectors, the equivalence and the scene names must line up")

    names = np.asarray(scenes)
    nearest = _nearest(vectors)
    is_same_object = same_as[np.arange(len(vectors)), nearest]

    return SameObject(
        retrieved=float(np.mean(is_same_object)),
        # Excluding the query, because retrieval takes it out before ranking. The reading
        # `twin_recall` uses is right there and twice this here.
        chance=chance_of(same_as, excluding_query=True),
        elsewhere=float(np.mean((names[nearest] == names) & ~is_same_object)),
    )


def queries_over(sizes: np.ndarray, count: int) -> list[int]:
    """Evenly spaced over the archive's range of target sizes, smallest to largest.

    Deterministic, so two runs of the command draw the same sheet, and spread rather than picked,
    so the sheet shows what the archive holds rather than what flatters the encoder.
    """
    if count < 1:
        raise ValueError(f"a sheet of {count} queries has nothing on it")

    ranked = sorted(range(len(sizes)), key=lambda index: (int(sizes[index]), index))
    if count >= len(ranked):
        return ranked
    # Before the arithmetic below, not after it: `(count - 1)` is a division, so a single query
    # raised rather than taking the branch written for it.
    if count == 1:
        return [ranked[len(ranked) // 2]]

    steps = [round(step * (len(ranked) - 1) / (count - 1)) for step in range(count)]
    return [ranked[step] for step in steps]


def table(retrieved: Sequence[Retrieved]) -> str:
    """One line per query: what was asked for, and what came back with it.

    The same shape `curve.table` has, for the same reason — a run should say what it found in the
    terminal it was run from, not only in a file someone has to go and open.
    """
    return "\n".join(
        f"{row.name:>26}  ->  "
        + "  ".join(f"{found.name} {found.similarity:.3f}" for found in row.found)
        for row in retrieved
    )


def contact_sheet(
    crops: np.ndarray,
    retrieved: Sequence[Retrieved],
    *,
    stretch: DecibelStretch,
    crop_px: int,
) -> str:
    """The queries and their neighbours as an image a reader can judge in a second.

    Drawn through the same window the encoder is shown, so that two crops side by side are two
    crops in one unit — a per-crop stretch would make every object look equally bright and would
    be exactly the flattering figure this is meant not to be.

    An SVG holding one PNG per cell rather than one rect per pixel: a 64 px crop is four thousand
    rectangles, and a sheet of forty-two of them is a file nobody opens twice.
    """
    cell, gap, label = 96, 10, 26
    columns = max(len(row.found) + 1 for row in retrieved)
    width = gap + columns * (cell + gap)
    height = gap + len(retrieved) * (cell + label + gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="9">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for row, found in enumerate(retrieved):
        top = gap + row * (cell + label + gap)
        similarities = [None, *(neighbour.similarity for neighbour in found.found)]
        names = [found.name, *(neighbour.name for neighbour in found.found)]
        for column, (index, name, similarity) in enumerate(
            zip(found.indices(), names, similarities, strict=True)
        ):
            left = gap + column * (cell + gap)
            image = _as_png(centre(crops[index][None], crop_px)[0], stretch)
            parts.append(
                f'<image x="{left}" y="{top}" width="{cell}" height="{cell}" '
                f'image-rendering="pixelated" href="data:image/png;base64,{image}"/>'
            )
            # Two lines rather than one. A name and a similarity written side by side are wider
            # than the cell they belong to, and captions that overrun their cell label the crop
            # next door — which on a figure whose whole argument is "look at these side by side"
            # is not a cosmetic fault.
            parts.append(f'<text x="{left}" y="{top + cell + 11}" fill="#222222">{name}</text>')
            if similarity is not None:
                parts.append(
                    f'<text x="{left}" y="{top + cell + 22}" fill="#666666">{similarity:.3f}</text>'
                )
            else:
                parts.append(f'<text x="{left}" y="{top + cell + 22}" fill="#666666">query</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _as_png(crop: np.ndarray, stretch: DecibelStretch) -> str:
    """One crop of decibels as a base64 greyscale PNG, written by hand.

    By hand because the alternative is a dependency on an image library for eight bytes of header
    and one call to zlib, in a package whose whole argument is that the chain installs without
    anything it does not need.
    """
    grey = np.clip(stretch(crop) * 255.0, 0, 255).astype(np.uint8)
    rows = b"".join(b"\x00" + row.tobytes() for row in grey)
    height, width = grey.shape

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    chunks = b"".join(
        _chunk(kind, payload)
        for kind, payload in (
            (b"IHDR", header),
            (b"IDAT", zlib.compress(rows, 9)),
            (b"IEND", b""),
        )
    )
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunks).decode("ascii")


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )
