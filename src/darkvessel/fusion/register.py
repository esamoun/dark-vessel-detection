"""The positions this chain will not call a dark vessel, and why each one is on the list.

A detection at a registered position is a fixed structure: a wind turbine, a transformer
platform, a mast. It is not matched against AIS and it is not dark, because neither verdict is
about it — a turbine has no transponder to have switched off, and reporting one as an undeclared
vessel is the chain producing a confident finding about a thing that has stood in the same place
since 2013.

**The exclusion is reported, never silent.** That is the whole design constraint of this module,
and it is what shapes everything below. A registered detection keeps its row, keeps its geometry,
and carries `structure_distance_m` saying how far it stood from the register entry that explained
it — so a run's output can be audited by anyone who suspects the register of eating a ship. A
pipeline that quietly dropped rows would be smaller, faster, and unauditable, and the count of
what it dropped would exist nowhere.

The register is a *file*, not a rule inferred at run time. One acquisition cannot tell a fixed
structure from a ship that happens to be there; only the archive can, and it does so in
`embed/structures.py` across ten weeks. What crosses the seam between them is a small table of
coordinates that a person can open, check against a chart, and correct by hand.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from darkvessel.detect.checkpoints import atomically
from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED

# The fourth verdict a detection can carry, beside `matched`, `dark` and `unsearched`. A separate
# status rather than a flag on `dark`, because it is a different claim about a different kind of
# object: `dark` says a search happened and explained nothing, and this says no search was owed.
STRUCTURE = "structure"

# What a register entry records. `x` and `y` place it in the CRS named beside them; `acquisitions`
# is the evidence — how many distinct acquisitions of the archive carried a detection here — and
# `source` says what put it on the list, so an entry added by hand off a chart is distinguishable
# from one the archive found.
ENTRY = ("x", "y", "acquisitions", "source")


@dataclass(frozen=True)
class Register:
    """Fixed structures, where they stand, and the radius around each one that they explain."""

    positions: pd.DataFrame
    crs: str
    tolerance_m: float

    def __post_init__(self) -> None:
        missing = [column for column in ENTRY if column not in self.positions.columns]
        if missing:
            raise ValueError(f"the register is missing {', '.join(missing)}")
        if self.tolerance_m <= 0:
            raise ValueError(
                f"a radius of {self.tolerance_m} m explains nothing; a register entry stands for "
                "a structure a detection may be a few pixels away from, not for a single pixel"
            )

    def __len__(self) -> int:
        return len(self.positions)

    def explains(self, detections: gpd.GeoDataFrame) -> np.ndarray:
        """How far each detection stands from the nearest register entry, NaN where none is near.

        In the register's own CRS, which the detections are reprojected into rather than the
        other way round: the register is the fixed thing here, and reprojecting it per scene
        would move the positions the whole point of this file is that they do not move.
        """
        if len(detections) == 0 or self.positions.empty:
            return np.full(len(detections), np.nan)

        placed = detections.to_crs(self.crs) if detections.crs != self.crs else detections
        apart = np.hypot(
            placed.geometry.x.to_numpy()[:, None] - self.positions["x"].to_numpy()[None, :],
            placed.geometry.y.to_numpy()[:, None] - self.positions["y"].to_numpy()[None, :],
        )
        nearest = apart.min(axis=1)
        return np.where(nearest <= self.tolerance_m, nearest, np.nan)

    def mark(self, detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Set `status` to `structure` on every detection a register entry explains.

        Applied after the matching rather than before it, and that order is deliberate. A
        detection that a declared vessel explains *and* a register entry explains is a vessel
        moored at a structure or a mast a transponder was wrongly placed on, and the AIS match is
        the stronger evidence — it names an MMSI. So `matched` survives and the distance is
        recorded anyway, which is the only way anyone would ever notice such a case.

        Every run gets the column, including a run with no register: a layer whose schema depends
        on whether a stage was configured is a layer that cannot be stacked with the one beside
        it. `embedder.attach` keeps the same promise for the same reason.
        """
        marked = detections.copy()
        distance = self.explains(detections)
        marked["structure_distance_m"] = distance
        marked.loc[np.isfinite(distance) & (marked["status"] != MATCHED), "status"] = STRUCTURE
        return marked

    def write(self, path: Path) -> None:
        """Write the register whole, or leave what was there untouched.

        A CSV, and that is a decision rather than a default: this is the one artefact of the
        level a person is expected to open, read, and correct against a chart. The CRS travels in
        a column so that a file moved away from the config that produced it still says what its
        metres are.
        """
        stored = self.positions[list(ENTRY)].copy()
        stored["crs"] = self.crs
        stored["tolerance_m"] = self.tolerance_m
        with atomically(path) as partial:
            stored.to_csv(partial, index=False)

    @classmethod
    def read(cls, path: Path) -> "Register":
        """The register at `path`, refusing one that disagrees with itself about its own metres."""
        stored = pd.read_csv(path)
        for column in ("crs", "tolerance_m"):
            if stored[column].nunique() != 1:
                raise ValueError(
                    f"{path.name} states {stored[column].nunique()} different values of "
                    f"{column}; the register is one list of positions in one frame"
                )

        return cls(
            positions=stored[list(ENTRY)],
            crs=str(stored["crs"].iloc[0]),
            tolerance_m=float(stored["tolerance_m"].iloc[0]),
        )


def without_a_register(detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The same layer a register would have produced, from a run that has none.

    The column is present and empty rather than absent. A run configured with no register has
    excluded nothing, which is a different statement from a run that cannot say — and both are
    different from a layer whose columns depend on which stages happened to be switched on.
    """
    marked = detections.copy()
    marked["structure_distance_m"] = np.full(len(detections), np.nan)
    return marked


def reduction(detections: gpd.GeoDataFrame, *, searched: bool) -> str:
    """What the exclusion cost this run, as the line it prints.

    The number the ticket asks to be quantified, computed off the layer that was written rather
    than off a counter kept during the run: what someone opening the output can reproduce is the
    only honest version of this figure.

    `searched` is passed in rather than read back off the layer, for the reason `cli._verdict`
    takes its own count that way. A run whose every detection was a structure has no `dark` row
    and no `unsearched` row left to be read, so the layer cannot say which verdict the excluded
    rows were spared — and guessing would have this line report a scene with no AIS behind it as
    a scene full of dark vessels that were caught in time.
    """
    excluded = int((detections["status"] == STRUCTURE).sum())
    if not excluded:
        return "no detection stood at a registered fixed structure"

    verdict = DARK if searched else UNSEARCHED
    remaining = int((detections["status"] == verdict).sum())
    return (
        f"{excluded} detection(s) excluded as fixed structures, leaving {remaining} {verdict}: "
        f"without the register this run would have reported {remaining + excluded}"
    )
