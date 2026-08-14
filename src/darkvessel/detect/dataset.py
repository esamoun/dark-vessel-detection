"""Dataset and augmentations for SAR detection.

Only geometry-preserving augmentations are physically valid on radar amplitude: flips and
rotations yes, colour and contrast jitter no. Speckle perturbation is the radar-native option.

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
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

# LS-SSDD's own split: sub-images cut from the first ten large scenes train, the last five are
# held out. Kept as the dataset publishes it so that the numbers this repository reports can be
# put next to the published baselines rather than only next to each other.
TRAINING_SCENES = tuple(range(1, 11))
HELD_OUT_SCENES = tuple(range(11, 16))


@dataclass(frozen=True)
class Layout:
    """Where the images and the annotations sit, as LS-SSDD ships them."""

    images: str = "JPEGImages"
    annotations: str = "Annotations"
    image_suffix: str = ".jpg"


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
        """Open the sub-image and return it as amplitude in 0..1."""
        with rasterio.open(self.image_path) as raster:
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
    different filesystem.

    Enumerated from the images rather than from the annotations, so that a half-attached dataset
    is an error rather than a smaller dataset. The images are what gets trained on; an image
    whose label never arrived would otherwise be a tile the run quietly never saw, or worse, a
    tile it saw as empty sea.
    """
    images = sorted((root / layout.images).glob(f"*{layout.image_suffix}"))
    if not images:
        raise FileNotFoundError(
            f"no {layout.image_suffix} images under {root / layout.images}; the dataset is "
            "attached at a different path, or under a different layout"
        )

    annotations = root / layout.annotations
    return [_ref_from(annotations / f"{image.stem}.xml", image) for image in images]


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

    def line(self, refs: Sequence[TileRef]) -> str:
        """What the subset kept and what it left, for the run to say out loud."""
        kept = self.of(refs)
        ships = sum(len(ref.boxes) for ref in kept)
        return (
            f"{len(kept)} of {len(refs)} tiles: {len([r for r in kept if r.boxes])} carrying "
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


def _ref_from(annotation: Path, image_path: Path) -> TileRef:
    if not annotation.exists():
        raise FileNotFoundError(
            f"{image_path.stem} has an image but no annotation at {annotation}; an incomplete "
            "download reads as a sea with no ships in it, and trains exactly that"
        )

    root = ElementTree.parse(annotation).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{annotation} declares no image size, so its boxes cannot be checked")
    shape = (int(size.findtext("height", "0")), int(size.findtext("width", "0")))

    return TileRef(
        name=annotation.stem,
        scene=_scene_of(annotation.stem),
        image_path=image_path,
        boxes=tuple(_box_from(bndbox, shape, annotation) for bndbox in root.iter("bndbox")),
    )


def _box_from(bndbox: ElementTree.Element, shape: tuple[int, int], annotation: Path) -> Box:
    """Read one VOC box, from inclusive pixel indices into half-open pixel edges.

    The file does not say whether its indices count from zero or from one, and on a ship four
    pixels across the difference is a quarter of the target. Zero is assumed, and the assumption
    is checked rather than trusted: under it no index can reach the size of the image, so one
    that does means the file counts from one and every box in the set is off by a pixel.
    """
    corners = {edge: int(bndbox.findtext(edge, "-1")) for edge in ("xmin", "ymin", "xmax", "ymax")}
    height, width = shape

    for edge, limit in (("xmin", width), ("xmax", width), ("ymin", height), ("ymax", height)):
        if not 0 <= corners[edge] < limit:
            raise ValueError(
                f"{annotation}: {edge} is {corners[edge]}, outside a {width} x {height} image; "
                "the annotations count their pixels from one, not from zero"
            )

    return Box(
        min_row=float(corners["ymin"]),
        min_col=float(corners["xmin"]),
        max_row=float(corners["ymax"] + 1),
        max_col=float(corners["xmax"] + 1),
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
