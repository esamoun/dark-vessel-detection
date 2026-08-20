"""Dataset and augmentations for SAR detection.

Only geometry-preserving augmentations are physically valid on radar amplitude: flips and
rotations yes, colour and contrast jitter no. What is applied here is the eight symmetries of the
square and nothing else. Speckle perturbation is the radar-native way to go further — it changes
pixel values, but according to the distribution the sensor itself imposes rather than
arbitrarily — and it is not implemented: it needs a speckle model to be argued for, and that
belongs with the rest of the work on what this data actually is.

The labels are LS-SSDD-v1.0 — 15 large Sentinel-1 IW acquisitions cut into 9000 sub-images of
800 x 800, VV, labelled by SAR experts against AIS and Google Earth. It is the one public set
whose physics matches what this chain will actually be handed: same satellite, same 10 m pixel,
same ships of three pixels against a large empty sea. Why it rather than the higher-resolution
sets, and what the training subset leaves out, are in docs/decisions.md.

Nothing here imports torch. The decisions that can be got wrong quietly — which scenes are held
out, which tiles are dropped, what an augmentation is allowed to do to a pixel — are all on this
side of the seam, so they are covered by a test suite that installs in a few seconds and runs
without a GPU. `train.py` holds the half that needs the framework.
"""

import hashlib
import random
import warnings
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

# LS-SSDD's own split: sub-images cut from the first ten large scenes train, the last five are
# held out. Kept as the dataset publishes it so that the numbers this repository reports can be
# put next to the published baselines rather than only next to each other. Only the held-out half
# is named, because it is the half the split is drawn on — everything else trains, including a
# sixteenth scene nobody has added yet.
HELD_OUT_SCENES = tuple(range(11, 16))


@dataclass(frozen=True)
class Layout:
    """Where the images and the annotations sit, as LS-SSDD ships them.

    `images` is a single directory name where the dataset ships that way, which is every config
    this repository has shipped so far and so remains the type a plain string satisfies without
    change. A mirror can instead split its images across several directories — LS-SSDD's own
    train/test split, mounted as two — and `images` may then name a sequence of them; `catalogue`
    reads the union. That shape matters beyond convenience: `split_by_scene` draws the held-out
    set from whichever scene numbers are present in the catalogue, and does not raise if none
    are. Point it at a directory that happens to hold only the training scenes and the held-out
    split comes back empty, silently, and a run would score it and report the number. Naming
    every directory the images sit under is what keeps that scene reachable at all.

    `first_index` is whether the annotations count their pixels from zero or from one. Left as
    None it is measured from the boxes themselves, which is what a full dataset always allows;
    it is here for the subset too small to settle the question on its own.
    """

    images: str | Sequence[str] = "JPEGImages"
    annotations: str = "Annotations"
    image_suffix: str = ".jpg"
    first_index: int | None = None


LS_SSDD = Layout()


@dataclass(frozen=True)
class Box:
    """One labelled ship, in half-open pixel-edge coordinates.

    The row interval is [`min_row`, `max_row`) and the column interval [`min_col`, `max_col`),
    measured from the top-left *corner* of the image. A box one pixel across therefore has a
    width of exactly 1, which is the whole reason for the convention: the annotations arrive as
    inclusive pixel indices, where the same box is 0 wide, and a detector trained on a ship
    reported as three pixels when it is four has learnt the wrong size of the only thing it is
    looking for.

    This is the same half-pixel that `geo.py` handles at the other end of the chain, and it is
    handled the same way — once, here, with the conversion named.
    """

    min_row: float
    min_col: float
    max_row: float
    max_col: float

    @property
    def height(self) -> float:
        return self.max_row - self.min_row

    @property
    def width(self) -> float:
        return self.max_col - self.min_col

    def centre(self) -> tuple[float, float]:
        """The middle of the box, in the fractional pixel indices `PixelDetection` reports.

        Pixel-edge coordinates count corners and pixel indices count centres, so the half
        subtracted here is the difference between the two frames. A ship labelled over rows 3
        to 5 has its centre at row 4, which is a pixel that exists.
        """
        return (
            (self.min_row + self.max_row) / 2 - 0.5,
            (self.min_col + self.max_col) / 2 - 0.5,
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """The same box as torchvision states one: x before y, and the corners in that order.

        The swap lives here, with the convention it swaps, for the reason the half-pixel above
        does. Written out at each call site instead, it is four subscripts that read the same
        whether or not they are right, in three places, and a transposed pair trains the model
        on ships rotated into the sea beside them.
        """
        return (self.min_col, self.min_row, self.max_col, self.max_row)

    @classmethod
    def from_xyxy(cls, xyxy: Sequence[float]) -> "Box":
        """The inverse, for boxes coming back out of a model."""
        return cls(min_row=xyxy[1], min_col=xyxy[0], max_row=xyxy[3], max_col=xyxy[2])


@dataclass(frozen=True)
class LabelledTile:
    """One sub-image and the ships in it, as the model is handed them."""

    name: str
    image: np.ndarray
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class TileRef:
    """A tile that has been catalogued but not read.

    The labels are a few kilobytes and the pixels are 9000 files; the split and the subset are
    decided from the labels alone, so the images are opened one at a time by whoever trains and
    never all at once. Holding the full set in memory would be 15 GB of float32 for a decision
    that needs none of it.
    """

    name: str
    scene: int
    image_path: Path
    boxes: tuple[Box, ...]

    def read(self) -> LabelledTile:
        """Open the sub-image and return it as amplitude in 0..1.

        A sub-image carries no georeferencing and is not supposed to: the detector works in
        pixels, and the chain places its answers on the ground afterwards in `geo.py`. rasterio
        warns about it once per file, which over 9000 tiles an epoch buries every line the run
        actually wanted to say — so the one warning that is expected here is silenced, and only
        that one.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            raster = rasterio.open(self.image_path)

        with raster:
            if raster.dtypes[0] != "uint8":
                raise ValueError(
                    f"{self.image_path} is {raster.dtypes[0]}, and this reader takes the 8-bit "
                    "amplitude LS-SSDD ships; a scene in dB is a different quantity and the "
                    "stretch between them is a decision, not a cast (see docs/decisions.md)"
                )
            band = raster.read(1)

        return LabelledTile(
            name=self.name,
            image=(band.astype(np.float32) / 255.0),
            boxes=self.boxes,
        )


def catalogue(root: Path, layout: Layout = LS_SSDD) -> list[TileRef]:
    """Read every annotation under `root` and return the tiles, in a fixed order.

    Sorted by name, because a run that is resumed has to see the tiles the interrupted one saw:
    the subset is drawn from this list, and a list in directory order is a different list on a
    different filesystem. Where `layout.images` names several directories, the tiles are the
    union of all of them, still sorted by name rather than by directory-then-name — the second
    would make the list depend on which directory was listed first, which is not a property
    either the resume guarantee or the subset draw can tolerate.

    Enumerated from the images rather than from the annotations, so that a half-attached dataset
    is an error rather than a smaller dataset. The images are what gets trained on; an image
    whose label never arrived would otherwise be a tile the run quietly never saw, or worse, a
    tile it saw as empty sea. Each named directory is checked for this on its own: a mirror that
    splits its images across directories can have one of them fail to attach while the others do,
    and reporting the set rather than the one that came back empty would leave that half of the
    fix undone.

    A stem is refused if it names an image under two of the directories. `_annotation_at` looks
    an annotation up by stem alone, so a duplicate would attach one label to two images and train
    the tile twice under it — silently, because nothing else here would notice two images that
    happen to share a name.

    Every annotation is read before any box is built, because a box cannot be converted until the
    set has decided where its indices start counting — see `_first_index`.
    """
    seen: dict[str, Path] = {}
    images: list[Path] = []
    for directory in _directories(layout.images):
        found = list((root / directory).glob(f"*{layout.image_suffix}"))
        if not found:
            raise FileNotFoundError(
                f"no {layout.image_suffix} images under {root / directory}; the dataset is "
                "attached at a different path, or under a different layout"
            )
        for image in found:
            if image.stem in seen:
                raise ValueError(
                    f"{image.stem!r} names an image under both {seen[image.stem].parent} and "
                    f"{image.parent}; the same tile would be catalogued twice, under one label"
                )
            seen[image.stem] = image
        images.extend(found)

    images = sorted(images, key=lambda image: image.name)

    annotations = root / layout.annotations
    annotated = [_annotation_at(annotations / f"{image.stem}.xml", image) for image in images]
    first_index = _first_index(annotated, layout.first_index)

    return [_ref_from(tile, first_index) for tile in annotated]


def _directories(images: str | Sequence[str]) -> tuple[str, ...]:
    """Normalise `Layout.images` to the directory names it holds, one or several."""
    return (images,) if isinstance(images, str) else tuple(images)


def split_by_scene(
    refs: Sequence[TileRef],
    held_out: Sequence[int] = HELD_OUT_SCENES,
) -> tuple[list[TileRef], list[TileRef]]:
    """Divide the tiles by the large scene they were cut from, never by sub-image.

    Two 800 px cuts of one Sentinel-1 acquisition are not two independent samples. They carry
    the same sea state, the same incidence angle, the same speckle statistics and the same
    calibration, and a ship on the seam between them appears in both. Split over sub-images and
    the held-out set measures how well the model recognises scenes it has already been trained
    on, which is a number that goes up and means nothing.
    """
    return (
        [ref for ref in refs if ref.scene not in held_out],
        [ref for ref in refs if ref.scene in held_out],
    )


@dataclass(frozen=True)
class Subset:
    """The training subset: every tile with a ship in it, and a bounded share of the empty sea.

    LS-SSDD is mostly background by design — 9000 sub-images against some six thousand ships —
    and on a free tier the binding constraint is the hours, not the disk. Dropping empty tiles
    buys epochs. It is bounded rather than total because a detector that has never been shown
    open water will find ships in it: `empty_per_ship_tile` is the ratio that is kept, and it is
    written in the config so that a run states how much sea it trained against.

    Only the training side is subsetted. The held-out split is scored entire, empty tiles and
    all, because dropping them there would remove exactly the tiles where a false positive can
    happen and report a precision the detector has not earned.
    """

    empty_per_ship_tile: float
    seed: int

    def of(self, refs: Sequence[TileRef]) -> list[TileRef]:
        with_ship = [ref for ref in refs if ref.boxes]
        empty = [ref for ref in refs if not ref.boxes]
        keep = min(len(empty), round(self.empty_per_ship_tile * len(with_ship)))

        # Seeded, and drawn from a list that is already in a fixed order, so that an interrupted
        # session and the session that resumes it train on the same tiles. An unseeded draw makes
        # the second half of a run a different experiment from the first.
        kept = with_ship + random.Random(self.seed).sample(empty, keep)
        return sorted(kept, key=lambda ref: ref.name)

    def line(self, kept: Sequence[TileRef], out_of: int) -> str:
        """What the subset kept and what it left, for the run to say out loud.

        Takes the selection rather than making a second one. Drawing it again here would walk
        six thousand tiles to describe a list the caller is already holding, and would describe
        a different list the day the selection stops being a pure function of the seed.
        """
        ships = sum(len(ref.boxes) for ref in kept)
        return (
            f"{len(kept)} of {out_of} tiles: {len([r for r in kept if r.boxes])} carrying "
            f"{ships} ships, {len([r for r in kept if not r.boxes])} empty at "
            f"{self.empty_per_ship_tile:g} per ship tile"
        )


@dataclass(frozen=True)
class Symmetry:
    """One of the eight ways a square tile can be laid down without changing any pixel value.

    Mirror first, then quarter turns: composing the two primitives is what keeps the boxes in
    step with the image, because each primitive moves both by the same arithmetic and neither
    has to be derived a second time for the composition.
    """

    name: str
    quarter_turns: int
    mirrored: bool

    def __call__(self, tile: LabelledTile) -> LabelledTile:
        image, boxes = tile.image, tile.boxes
        if self.mirrored:
            image, boxes = _mirrored(image, boxes)
        for _ in range(self.quarter_turns):
            image, boxes = _quarter_turned(image, boxes)

        return LabelledTile(name=tile.name, image=image, boxes=boxes)


SYMMETRIES = tuple(
    Symmetry(
        name=("mirror+" if mirrored else "") + f"rot{turns * 90}",
        quarter_turns=turns,
        mirrored=mirrored,
    )
    for mirrored in (False, True)
    for turns in range(4)
)


def symmetry_for(name: str, epoch: int, seed: int) -> Symmetry:
    """Which way this tile is laid down, in this epoch, in this run.

    Derived from the names rather than drawn from a stream, for two reasons that are the same
    reason. A resumed run has to reproduce the augmentation the interrupted one was applying, or
    the two halves are different experiments; and a data loader with several workers draws from
    several streams, so a global generator makes the order of the tiles decide what happens to
    them. A hash depends on neither.
    """
    key = f"{seed}:{epoch}:{name}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return SYMMETRIES[int.from_bytes(digest, "big") % len(SYMMETRIES)]


def _mirrored(image: np.ndarray, boxes: tuple[Box, ...]) -> tuple[np.ndarray, tuple[Box, ...]]:
    """Reflect left to right. In half-open coordinates the two edges simply swap places."""
    width = image.shape[1]
    return (
        np.ascontiguousarray(image[:, ::-1]),
        tuple(
            Box(
                min_row=box.min_row,
                min_col=width - box.max_col,
                max_row=box.max_row,
                max_col=width - box.min_col,
            )
            for box in boxes
        ),
    )


def _quarter_turned(
    image: np.ndarray, boxes: tuple[Box, ...]
) -> tuple[np.ndarray, tuple[Box, ...]]:
    """Turn a quarter turn anticlockwise. The old columns become the new rows, reversed."""
    width = image.shape[1]
    return (
        np.ascontiguousarray(np.rot90(image, 1)),
        tuple(
            Box(
                min_row=width - box.max_col,
                min_col=box.min_row,
                max_row=width - box.min_col,
                max_col=box.max_row,
            )
            for box in boxes
        ),
    )


@dataclass(frozen=True)
class _Annotated:
    """One annotation file, read but not yet interpreted.

    Held in this half-read state because a box cannot be converted until the whole set has been
    seen: what its numbers mean depends on where the annotations start counting, and that is a
    property of the dataset rather than of any one file in it.
    """

    name: str
    image_path: Path
    height: int
    width: int
    corners: tuple[dict[str, int], ...]


def _annotation_at(path: Path, image_path: Path) -> _Annotated:
    if not path.exists():
        raise FileNotFoundError(
            f"{image_path.stem} has an image but no annotation at {path}; an incomplete "
            "download reads as a sea with no ships in it, and trains exactly that"
        )

    root = ElementTree.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{path} declares no image size, so its boxes cannot be checked")

    return _Annotated(
        name=path.stem,
        image_path=image_path,
        height=int(size.findtext("height", "0")),
        width=int(size.findtext("width", "0")),
        corners=tuple(_corners_of(bndbox, path) for bndbox in root.iter("bndbox")),
    )


def _corners_of(bndbox: ElementTree.Element, path: Path) -> dict[str, int]:
    corners = {}
    for edge in ("xmin", "ymin", "xmax", "ymax"):
        written = bndbox.findtext(edge)
        if written is None:
            raise ValueError(f"{path}: a box has no {edge}, so it describes no region at all")
        corners[edge] = int(float(written))

    return corners


def _first_index(annotated: Sequence[_Annotated], declared: int | None) -> int:
    """Whether the annotations count their pixels from zero or from one.

    Measured from the boxes rather than assumed, because the file does not say and the two
    readings differ by a pixel — which on a ship four pixels across is a quarter of the target,
    applied to every ship in the set, in the same direction, invisibly. PASCAL VOC as originally
    published counts from one; sets written with later tools frequently do not, and LS-SSDD says
    nothing either way.

    The evidence is decisive when it exists. An index of 0 cannot occur in a file counting from
    one, and an index equal to the width cannot occur in a file counting from zero — so a single
    box touching either edge settles it, and over thousands of tiles cut from a scene there are
    always many. Where a subset is too small to contain one, this refuses rather than guesses,
    and names the setting that answers it.
    """
    lower = any(
        corner[edge] == 0 for tile in annotated for corner in tile.corners for edge in corner
    )
    upper = any(
        corner[edge] == extent
        for tile in annotated
        for corner in tile.corners
        for edge, extent in (
            ("xmin", tile.width),
            ("xmax", tile.width),
            ("ymin", tile.height),
            ("ymax", tile.height),
        )
    )

    if lower and upper:
        raise ValueError(
            "the annotations hold both an index of 0 and one equal to the image size, which no "
            "single convention allows; they are not all in the same frame"
        )
    if declared is not None:
        return declared
    if lower:
        return 0
    if upper:
        return 1

    raise ValueError(
        "cannot tell whether these annotations count their pixels from zero or from one: no box "
        "touches either edge of its tile. Over the whole dataset one always does, so this is a "
        "subset — set `data.first_index` in the config to say which, and see docs/decisions.md"
    )


def _ref_from(annotated: _Annotated, first_index: int) -> TileRef:
    """Turn one read annotation into a tile, converting inclusive indices to half-open edges."""
    for corner in annotated.corners:
        for edge, extent in (
            ("xmin", annotated.width),
            ("xmax", annotated.width),
            ("ymin", annotated.height),
            ("ymax", annotated.height),
        ):
            if not first_index <= corner[edge] < extent + first_index:
                raise ValueError(
                    f"{annotated.name}: {edge} is {corner[edge]}, outside a "
                    f"{annotated.width} x {annotated.height} image whose annotations count from "
                    f"{first_index}"
                )

    return TileRef(
        name=annotated.name,
        scene=_scene_of(annotated.name),
        image_path=annotated.image_path,
        boxes=tuple(
            Box(
                min_row=float(corner["ymin"] - first_index),
                min_col=float(corner["xmin"] - first_index),
                max_row=float(corner["ymax"] - first_index + 1),
                max_col=float(corner["xmax"] - first_index + 1),
            )
            for corner in annotated.corners
        ),
    )


def _scene_of(name: str) -> int:
    """Which large acquisition a sub-image was cut from. LS-SSDD names them `<scene>_<index>`."""
    scene, _, index = name.partition("_")
    if not index or not scene.isdigit():
        raise ValueError(
            f"cannot tell which scene {name!r} was cut from; LS-SSDD names its sub-images "
            "'<scene>_<index>', and the split is drawn on that number"
        )
    return int(scene)
