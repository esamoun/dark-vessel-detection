"""The labelled data, and what may be done to it.

Two classes of fault live here and neither one crashes. A split that lets sub-images from the
same large scene fall on both sides reports a recall it did not earn — adjacent 800 px cuts of
one Sentinel-1 acquisition share their speckle statistics, their sea state and sometimes the two
halves of one ship. And an augmentation that changes pixel values rather than only moving them
is physically meaningless on radar amplitude: brightness *is* the measurement, so a contrast
jitter does not produce a plausible second look at the same ship, it produces a ship made of a
different material.

Both are asserted here as properties of the code rather than checked by eye on the first run.
The dataset is built by hand in the fixture below, small enough to count: what LS-SSDD-v1.0
ships is 9000 sub-images of 800 x 800, which is not something a test suite downloads.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from darkvessel.detect.dataset import (
    HELD_OUT_SCENES,
    SYMMETRIES,
    Box,
    LabelledTile,
    Layout,
    Subset,
    catalogue,
    split_by_scene,
    symmetry_for,
)

# A fixture written as GeoTIFF rather than the JPEG LS-SSDD ships, because GDAL's JPEG driver
# copies files and does not create them, so rasterio cannot write one. Nothing in the reader
# cares: it opens whatever the layout names and takes band 1.
FIXTURE = Layout(image_suffix=".tif")

SIZE = 16

# A sub-image carries no georeferencing, here or in LS-SSDD, and that is the point: the detector
# works in pixels and the chain places its answers on the ground afterwards, in `geo.py`.
pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")


def write_dataset(
    root: Path, ships: dict[str, list[tuple[int, int, int, int]]], size: int = SIZE
) -> Path:
    """Write a dataset in LS-SSDD's layout, with `ships` as inclusive pixel indices per tile."""
    (root / FIXTURE.images).mkdir(parents=True, exist_ok=True)
    (root / FIXTURE.annotations).mkdir(parents=True, exist_ok=True)

    for name, boxes in ships.items():
        image = np.zeros((size, size), dtype=np.uint8)
        for min_row, min_col, max_row, max_col in boxes:
            image[min_row : max_row + 1, min_col : max_col + 1] = 255

        with rasterio.open(
            root / FIXTURE.images / f"{name}{FIXTURE.image_suffix}",
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=1,
            dtype="uint8",
        ) as raster:
            raster.write(image, 1)

        objects = "".join(
            "<object><name>ship</name><bndbox>"
            f"<xmin>{min_col}</xmin><ymin>{min_row}</ymin>"
            f"<xmax>{max_col}</xmax><ymax>{max_row}</ymax>"
            "</bndbox></object>"
            for min_row, min_col, max_row, max_col in boxes
        )
        (root / FIXTURE.annotations / f"{name}.xml").write_text(
            f"<annotation><filename>{name}{FIXTURE.image_suffix}</filename>"
            f"<size><width>{size}</width><height>{size}</height><depth>1</depth></size>"
            f"{objects}</annotation>"
        )

    return root


def a_dataset_spanning_every_scene(root: Path) -> Path:
    """One tile carrying a ship and one pure background, from each of LS-SSDD's 15 scenes."""
    return write_dataset(
        root,
        {
            f"{scene:02d}_{index}": [(0, 0, 2, 3)] if index == 1 else []
            for scene in range(1, 16)
            for index in (1, 2)
        },
    )


def test_no_scene_contributes_to_both_sides_of_the_split(tmp_path: Path) -> None:
    """The property the split exists for. Sub-images are cut from 15 large acquisitions, so a
    split drawn over sub-images puts neighbouring cuts of one scene on both sides and measures
    memorisation as generalisation."""
    refs = catalogue(a_dataset_spanning_every_scene(tmp_path), FIXTURE)

    training, held_out = split_by_scene(refs)

    scenes = lambda refs: {ref.scene for ref in refs}  # noqa: E731
    assert not scenes(training) & scenes(held_out)
    assert scenes(training) | scenes(held_out) == set(range(1, 16))


def test_the_held_out_side_is_the_one_lsssdd_holds_out(tmp_path: Path) -> None:
    """LS-SSDD's own split is scenes 01-10 against 11-15, and published baselines are measured
    against it. Drawing a different one would make every number here incomparable."""
    _, held_out = split_by_scene(catalogue(a_dataset_spanning_every_scene(tmp_path), FIXTURE))

    assert {ref.scene for ref in held_out} == set(HELD_OUT_SCENES)


def three_ships_and_thirty_six_empty_tiles(root: Path) -> list:
    """One scene's worth: 3 tiles carrying a ship and 36 of nothing but sea."""
    return catalogue(
        write_dataset(
            root,
            {f"01_{index}": [(0, 0, 2, 3)] if index < 4 else [] for index in range(1, 40)},
        ),
        FIXTURE,
    )


def test_a_tile_carrying_a_ship_is_never_dropped_from_the_subset(tmp_path: Path) -> None:
    refs = three_ships_and_thirty_six_empty_tiles(tmp_path)

    kept = Subset(empty_per_ship_tile=1.0, seed=0).of(refs)

    assert {ref.name for ref in refs if ref.boxes} <= {ref.name for ref in kept}


def test_the_subset_keeps_the_stated_number_of_empty_tiles(tmp_path: Path) -> None:
    """What is excluded, said as a number. Three tiles carry a ship, so at one empty per ship
    tile three of the 36 pure backgrounds survive and 33 are left on the disk."""
    refs = three_ships_and_thirty_six_empty_tiles(tmp_path)

    kept = Subset(empty_per_ship_tile=1.0, seed=0).of(refs)

    assert len([ref for ref in kept if not ref.boxes]) == 3
    assert len(kept) == 6
    assert Subset(empty_per_ship_tile=1.0, seed=0).line(kept, out_of=len(refs)) == (
        "6 of 39 tiles: 3 carrying 3 ships, 3 empty at 1 per ship tile"
    )


def test_the_same_seed_selects_the_same_tiles_in_the_next_session(tmp_path: Path) -> None:
    """A resumed run must train on the data the interrupted one was training on. Draw the empty
    tiles from an unseeded source and epoch 4 of a Kaggle session sees a different dataset from
    epoch 3 of the session before it, which is not the same run continued."""
    refs = catalogue(
        write_dataset(
            tmp_path,
            {f"01_{index}": [] for index in range(1, 40)} | {"01_0": [(0, 0, 2, 2)]},
        ),
        FIXTURE,
    )
    subset = Subset(empty_per_ship_tile=4.0, seed=7)

    assert [ref.name for ref in subset.of(refs)] == [ref.name for ref in subset.of(refs)]


def with_annotation(root: Path, name: str, boxes: str) -> Path:
    """Replace one tile's annotation, to write corner values the fixture would never produce."""
    (root / FIXTURE.annotations / f"{name}.xml").write_text(
        f"<annotation><size><width>{SIZE}</width><height>{SIZE}</height></size>{boxes}</annotation>"
    )
    return root


def a_box(xmin: int, ymin: int, xmax: int, ymax: int) -> str:
    return (
        f"<object><name>ship</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>"
    )


def test_annotations_that_count_from_zero_are_read_as_counting_from_zero(tmp_path: Path) -> None:
    """The file does not say where its indices start, and the two readings differ by a pixel —
    on a ship four pixels across, a quarter of the target, applied to every ship in the same
    direction and visible nowhere. So it is measured: an index of 0 cannot occur in a set that
    counts from one, and a single box carrying one settles the question."""
    root = write_dataset(tmp_path, {"01_1": [], "01_2": []})
    with_annotation(tmp_path, "01_1", a_box(xmin=0, ymin=2, xmax=3, ymax=5))
    with_annotation(tmp_path, "01_2", a_box(xmin=4, ymin=4, xmax=6, ymax=6))

    refs = catalogue(root, FIXTURE)

    assert refs[0].boxes == (Box(min_row=2.0, min_col=0.0, max_row=6.0, max_col=4.0),)
    assert refs[1].boxes[0].width == 3.0


def test_annotations_that_count_from_one_are_read_as_counting_from_one(tmp_path: Path) -> None:
    """The other side. PASCAL VOC as published counts from one, and an index equal to the width
    is the evidence of it: a set counting from zero cannot reach it."""
    root = write_dataset(tmp_path, {"01_1": [], "01_2": []})
    with_annotation(tmp_path, "01_1", a_box(xmin=4, ymin=2, xmax=SIZE, ymax=5))
    with_annotation(tmp_path, "01_2", a_box(xmin=4, ymin=4, xmax=6, ymax=6))

    refs = catalogue(root, FIXTURE)

    assert refs[0].boxes == (Box(min_row=1.0, min_col=3.0, max_row=5.0, max_col=float(SIZE)),)
    assert refs[1].boxes[0].width == 3.0


def test_a_subset_that_cannot_settle_the_question_refuses_to_guess(tmp_path: Path) -> None:
    """Over 9000 tiles some box always touches an edge. Over a handful none may, and a default
    would be a silent one-pixel shift — so it refuses, and names the setting that answers it."""
    root = write_dataset(tmp_path, {"01_1": []})
    with_annotation(tmp_path, "01_1", a_box(xmin=4, ymin=4, xmax=6, ymax=6))

    with pytest.raises(ValueError, match="first_index"):
        catalogue(root, FIXTURE)


def test_a_subset_can_be_told_which_convention_it_is_in(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, {"01_1": []})
    with_annotation(tmp_path, "01_1", a_box(xmin=4, ymin=4, xmax=6, ymax=6))

    refs = catalogue(root, Layout(image_suffix=".tif", first_index=1))

    assert refs[0].boxes == (Box(min_row=3.0, min_col=3.0, max_row=6.0, max_col=6.0),)


def test_annotations_in_two_different_frames_are_refused(tmp_path: Path) -> None:
    """Both an index of 0 and one equal to the width, which no single convention allows."""
    root = write_dataset(tmp_path, {"01_1": [], "01_2": []})
    with_annotation(tmp_path, "01_1", a_box(xmin=0, ymin=2, xmax=3, ymax=5))
    with_annotation(tmp_path, "01_2", a_box(xmin=4, ymin=4, xmax=SIZE, ymax=6))

    with pytest.raises(ValueError, match="not all in the same frame"):
        catalogue(root, FIXTURE)


def test_a_box_missing_a_corner_says_so(tmp_path: Path) -> None:
    """Rather than defaulting the corner and failing later with a message about conventions."""
    root = write_dataset(tmp_path, {"01_1": []})
    with_annotation(
        tmp_path,
        "01_1",
        "<object><name>ship</name><bndbox><ymin>2</ymin><xmax>3</xmax><ymax>5</ymax></bndbox>"
        "</object>",
    )

    with pytest.raises(ValueError, match="no xmin"):
        catalogue(root, FIXTURE)


def test_a_tile_with_no_annotation_file_is_refused(tmp_path: Path) -> None:
    """An incomplete download reads as a sea with no ships in it: every tile trains as pure
    background, the loss falls, and the run reports a recall of zero at the end of the evening."""
    root = write_dataset(tmp_path, {"01_1": [(0, 0, 2, 3)], "01_2": [(0, 0, 2, 3)]})
    (root / FIXTURE.annotations / "01_2.xml").unlink()

    with pytest.raises(FileNotFoundError, match="01_2"):
        catalogue(root, FIXTURE)


def test_a_tile_reads_back_the_ship_that_was_written_into_it(tmp_path: Path) -> None:
    """The end-to-end reading step, including the inclusive-to-half-open conversion: a ship
    occupying rows 3 to 5 is three pixels tall, not two and not four, and its centre is at row
    4 — a pixel that exists — rather than at 4.5."""
    root = write_dataset(tmp_path, {"01_1": [(3, 4, 5, 7)]})

    tile = catalogue(root, Layout(image_suffix=".tif", first_index=0))[0].read()

    assert tile.image.shape == (SIZE, SIZE)
    assert tile.boxes == (Box(min_row=3.0, min_col=4.0, max_row=6.0, max_col=8.0),)
    assert (tile.boxes[0].height, tile.boxes[0].width) == (3.0, 4.0)
    assert tile.boxes[0].centre() == (4.0, 5.5)
    assert tile.image.max() == pytest.approx(1.0)


def a_tile_with_a_ship() -> LabelledTile:
    """A ship of a value nothing else in the image carries, so it can be found after a move."""
    image = np.zeros((8, 12), dtype=np.float32)
    image[2:4, 5:9] = 1.0
    return LabelledTile(
        name="01_1",
        image=image,
        boxes=(Box(min_row=2.0, min_col=5.0, max_row=4.0, max_col=9.0),),
    )


@pytest.mark.parametrize("symmetry", SYMMETRIES, ids=lambda s: s.name)
def test_an_augmentation_moves_pixels_and_never_changes_them(symmetry) -> None:
    """The acceptance condition, as a property rather than as a list of what was not used.

    Amplitude on radar is the measurement: a hull returns more energy than the sea around it, and
    a contrast jitter claims the same hull was made of something else. Any augmentation kept here
    is a permutation of the pixels, so the sorted values of the tile come back identical.
    """
    tile = a_tile_with_a_ship()

    moved = symmetry(tile)

    assert np.array_equal(np.sort(moved.image, axis=None), np.sort(tile.image, axis=None))


@pytest.mark.parametrize("symmetry", SYMMETRIES, ids=lambda s: s.name)
def test_a_box_follows_the_ship_it_labels(symmetry) -> None:
    """The other half: a transform applied to the image and not to its boxes trains the model to
    predict a ship where the sea is, and nothing in the loss curve says so."""
    tile = a_tile_with_a_ship()

    moved = symmetry(tile)

    inside = np.zeros_like(moved.image, dtype=bool)
    for box in moved.boxes:
        inside[int(box.min_row) : int(box.max_row), int(box.min_col) : int(box.max_col)] = True
    assert np.array_equal(moved.image > 0.0, inside)


def test_a_resumed_epoch_lays_every_tile_down_the_way_the_interrupted_one_did() -> None:
    """Which symmetry a tile gets is a function of the run, the epoch and the tile's name, not
    of a generator's position in a stream. A session killed halfway through epoch 4 and restarted
    has to continue epoch 4, not start a differently-augmented one."""
    before = [symmetry_for(f"01_{index}", epoch=4, seed=3) for index in range(20)]

    after = [symmetry_for(f"01_{index}", epoch=4, seed=3) for index in range(20)]

    assert before == after


def test_a_tile_is_laid_down_differently_across_the_schedule() -> None:
    """The other half of the same property: fixed within an epoch, varying across them, or the
    augmentation is a one-off relabelling of the dataset rather than an augmentation."""
    over_epochs = {symmetry_for("01_1", epoch=epoch, seed=3).name for epoch in range(40)}

    assert len(over_epochs) > 1


def test_a_box_survives_the_trip_through_torchvision_s_ordering() -> None:
    """x before y, and the corners in that order. The swap has one owner because written out at
    a call site it reads the same whether or not it is right, and a transposed pair trains the
    model on ships rotated into the sea beside them."""
    box = Box(min_row=2.0, min_col=5.0, max_row=4.0, max_col=9.0)

    assert box.to_xyxy() == (5.0, 2.0, 9.0, 4.0)
    assert Box.from_xyxy(box.to_xyxy()) == box


def test_the_symmetries_are_eight_and_all_different() -> None:
    """The dihedral group of the square. Fewer means one was dropped; a repeat means a tile is
    twice as likely to be seen in one orientation as another."""
    tile = a_tile_with_a_ship()

    seen = {symmetry(tile).image.tobytes() for symmetry in SYMMETRIES}

    assert len(SYMMETRIES) == 8
    assert len(seen) == 8
