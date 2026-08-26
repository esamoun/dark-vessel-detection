"""The detection archive: every crop the chain has cut, and where each one came from.

Retrieval is a claim across acquisitions rather than inside one. A single Sentinel-1 box over the
Kattegat holds a handful of vessels, and a representation fitted on a handful of objects has
learned a handful of objects; what makes the question "show me the ones that look like this" mean
anything is a file that accumulates them. So this is the one artefact of this level that grows: a
run over a new scene appends to it, and nothing is recomputed.

Two things are stored beside the pixels and neither is decoration. The provenance, because a
neighbour that cannot be pointed at on a map or in an acquisition is not evidence of anything —
"these two look alike" is only useful if both can be looked at. And the crop geometry, because an
encoder fitted on crops of one size and applied to crops of another is the same class of silent
fault as a checkpoint loaded with the wrong anchors: nothing raises, and the answers stay
plausible. `train.py` records what built a detector for that reason, and this records what built
an archive.

Nothing here imports torch. What a crop *is* has to be readable without the framework that
learned from it, for the same reason `Journal` writes numbers rather than a pickle.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from darkvessel.detect.checkpoints import atomically

# What is recorded about each crop, in the order a table of them reads best. `scene` and
# `acquired_at` say which acquisition; `row` and `col` where in it; `x` and `y` where on the
# ground, so a neighbour can be opened in QGIS without going back through a transform; `score`
# what the detector thought of it, which is not a label and is the nearest thing to one.
PROVENANCE = ("scene", "acquired_at", "row", "col", "x", "y", "score")


@dataclass(frozen=True)
class Archive:
    """Crops, where each came from, and the geometry they were cut at."""

    crops: np.ndarray
    provenance: pd.DataFrame
    crop_px: int
    margin_px: int

    def __post_init__(self) -> None:
        if self.crops.ndim != 3 or self.crops.shape[1] != self.crops.shape[2]:
            raise ValueError(f"crops must be a stack of squares, got shape {self.crops.shape}")
        if len(self.crops) != len(self.provenance):
            raise ValueError(
                f"{len(self.crops)} crops against {len(self.provenance)} rows of provenance; "
                "a crop nobody can place is not evidence of anything"
            )
        missing = [column for column in PROVENANCE if column not in self.provenance.columns]
        if missing:
            raise ValueError(f"the provenance is missing {', '.join(missing)}")

    def __len__(self) -> int:
        return len(self.crops)

    def scenes(self) -> list[str]:
        """The acquisitions this archive draws on, in the order they were acquired."""
        ordered = self.provenance.sort_values("acquired_at")
        return list(dict.fromkeys(ordered["scene"].tolist()))

    def with_more(self, other: "Archive") -> "Archive":
        """This archive and another, as one. Refuses two cut at different geometries."""
        if (self.crop_px, self.margin_px) != (other.crop_px, other.margin_px):
            raise ValueError(
                f"these crops are {self.crop_px}/{self.margin_px} px and those are "
                f"{other.crop_px}/{other.margin_px}; an encoder fitted across both would be "
                "fitted at two scales and would say so nowhere"
            )
        return Archive(
            crops=np.concatenate([self.crops, other.crops]),
            provenance=pd.concat([self.provenance, other.provenance], ignore_index=True),
            crop_px=self.crop_px,
            margin_px=self.margin_px,
        )

    def co_located(self, tolerance_m: float) -> np.ndarray:
        """Which crops are the same object as which, as a square of booleans.

        A detector run at the operating point an archive wants returns a large hull several times
        — two thirds of these crops have another detection within 200 m of them in their own
        acquisition, and the median distance between such a pair is 31 m. Those are not two
        objects that resemble each other. They are one object, cut twice.

        It matters because every check at this level is a ranking, and a ranking that counts the
        second cut of a vessel as a wrong answer measures the archive's duplication rather than
        the representation. The tolerance is the fusion's own — the distance at which this project
        already says two positions are one vessel — so there is one definition of "the same
        object" and not a second one invented here.

        The diagonal is true: a crop is itself.
        """
        places = self.provenance[["x", "y"]].to_numpy()
        apart = np.hypot(
            places[:, 0][:, None] - places[:, 0][None, :],
            places[:, 1][:, None] - places[:, 1][None, :],
        )
        scene = self.provenance["scene"].to_numpy()
        return (scene[:, None] == scene[None, :]) & (apart <= tolerance_m)

    def write(self, path: Path) -> None:
        """Write the archive whole, or leave whatever was there untouched.

        Through `atomically` for the reason the checkpoints use it: this file is appended to over
        many sessions, and a process killed part way through writing it would otherwise leave a
        truncated archive under the name the next session reads.
        """
        with atomically(path) as partial:
            with partial.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    crops=self.crops,
                    crop_px=np.int64(self.crop_px),
                    margin_px=np.int64(self.margin_px),
                    **{column: _as_stored(self.provenance[column]) for column in PROVENANCE},
                )

    @classmethod
    def read(cls, path: Path) -> "Archive":
        """The archive at `path`, refusing anything that would need unpickling to read.

        `allow_pickle=False` is the guard rather than a default: an archive is a record, and a
        record that can execute on the way in is not one. It is also what forces the string
        columns to be written as fixed-width text — see `_as_stored`.
        """
        with np.load(path, allow_pickle=False) as stored:
            return cls(
                crops=stored["crops"],
                provenance=pd.DataFrame({column: stored[column] for column in PROVENANCE}),
                crop_px=int(stored["crop_px"]),
                margin_px=int(stored["margin_px"]),
            )


def _as_stored(column: "pd.Series") -> np.ndarray:
    """One provenance column, in a dtype `np.savez` can write without pickling it.

    Pandas holds a column of strings as objects, and an object array reaches `.npz` as a pickle —
    which `read` then refuses, so an archive would be written that nothing could read back. Fixed
    width text instead, which is what the column actually is.
    """
    values = column.to_numpy()
    return values.astype("U") if values.dtype == object else values
