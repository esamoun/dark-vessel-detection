"""Published positions of the fixed offshore structures in a box, and what that publication is.

Issue #14 asks that the structures this project identifies be verified against known offshore
wind farm locations. A verification is worth exactly what its reference is worth, so this module
is mostly about saying what the reference is.

**The source is OpenStreetMap**, through the Overpass API. It is not the authoritative one: the
authority for Danish turbines is Energistyrelsen's Stamdataregister, which as of 2026-08-27 is
published through a map viewer rather than as a file this could fetch. OSM's own limits are
documented by OSM and carried into the file this writes rather than left for a reader to
discover — of the 92 turbines it records at Anholt, 91 carry `note=position only approximate`,
and the farm is documented as having 111.

Both of those are stated in `docs/decisions.md`, 2026-08-27, together with the argument that
survives them: an approximate list and an independent radar archive agreeing to half a pixel is
evidence about both, and the direction the incompleteness runs in can be measured rather than
assumed.

Structures, not turbines. A wind farm has a transformer platform in it, and a platform is a
fixed structure by every argument that makes a mast one — at Anholt it is the single most
persistent object in the whole archive after the masts, and a reference containing only turbines
would have reported it as the method's one false alarm.

This is the one module here that needs a network, in the way `data/dma.py` and `data/gee_export.py`
do. What it writes is a small CSV, kept in the repository, so nothing downstream of it ever needs
one.
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from darkvessel.data.area import Bounds
from darkvessel.detect.checkpoints import atomically

OVERPASS = "https://overpass-api.de/api/interpreter"

# What is asked for. Turbines are nodes; a transformer platform is drawn as a way and comes back
# through `out center`, which is why the query asks for both and the parser reads either.
QUERY = """[out:json][timeout:{timeout}];
(
  nwr["generator:source"="wind"]({south},{west},{north},{east});
  nwr["power"="substation"]["location"="platform"]({south},{west},{north},{east});
);
out center tags;
"""

# What a reference row records. `lon` and `lat` in the frame the source publishes in, never
# reprojected on the way in: this file is a record of what somebody else said, and a conversion
# performed here would make it a record of what this project did to what somebody else said.
# `approximate` carries the source's own caveat per row, because it is per row.
COLUMNS = ("lon", "lat", "kind", "approximate", "osm_id")


@dataclass(frozen=True)
class Known:
    """Published structure positions, and the frame they were published in."""

    positions: pd.DataFrame
    source: str

    def __len__(self) -> int:
        return len(self.positions)

    def placed(self, crs: str) -> pd.DataFrame:
        """The same positions as `x` and `y` in `crs`, for comparing against a register.

        Reprojected here rather than on the way into the file, and once per use rather than
        stored, so there is never a moment when two columns of one row disagree about where a
        turbine is.
        """
        from pyproj import Transformer

        into = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = into.transform(
            self.positions["lon"].to_numpy(dtype=float),
            self.positions["lat"].to_numpy(dtype=float),
        )
        placed = self.positions.copy()
        placed["x"] = x
        placed["y"] = y
        return placed

    def inside(self, bounds: Bounds) -> "Known":
        """Only the structures inside a rectangle, in the degrees the rectangle is stated in.

        The archive is clipped to this rectangle, so a structure outside it was never imaged and
        counting it as one this method failed to find would be counting a miss against water
        nobody looked at.
        """
        within = self.positions[
            self.positions["lon"].between(bounds.west, bounds.east)
            & self.positions["lat"].between(bounds.south, bounds.north)
        ]
        return Known(positions=within.reset_index(drop=True), source=self.source)

    def write(self, path: Path) -> None:
        stored = self.positions[list(COLUMNS)].copy()
        stored["source"] = self.source
        with atomically(path) as partial:
            stored.to_csv(partial, index=False)

    @classmethod
    def read(cls, path: Path) -> "Known":
        """The reference at `path`. A file with no rows is a real answer, not a broken file.

        The Kattegat lane's reference is empty, and that is the control this whole level rests
        on: nothing published stands in that box. It has to read back as an empty reference
        rather than as a missing one, or the box that proves the method raises instead.
        """
        stored = pd.read_csv(path)
        source = str(stored["source"].iloc[0]) if len(stored) else OVERPASS
        return cls(positions=stored[list(COLUMNS)], source=source)


def fetch(bounds: Bounds, timeout_s: int = 90) -> Known:
    """Every fixed offshore structure OpenStreetMap records inside `bounds`.

    One request, no key, no account. The response is parsed for a position and three facts about
    it and everything else is dropped: a reference file that carried the manufacturer and the
    light characteristic of every mast would be a file nobody checks, and what a verification
    needs is coordinates and their stated accuracy.
    """
    query = QUERY.format(
        timeout=timeout_s,
        south=bounds.south,
        west=bounds.west,
        north=bounds.north,
        east=bounds.east,
    )
    request = urllib.request.Request(
        OVERPASS,
        data=urllib.parse.urlencode({"data": query}).encode("ascii"),
        headers={"User-Agent": "darkvessel (offshore structure reference)"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s + 30) as response:  # noqa: S310
        answered = json.loads(response.read().decode("utf-8"))

    return Known(positions=_rows_of(answered["elements"]), source=OVERPASS)


def _rows_of(elements: list[dict]) -> pd.DataFrame:
    """One row per element that has a position, whether it was drawn as a node or as a shape."""
    rows = []
    for element in elements:
        centre = element.get("center") or element
        if centre.get("lat") is None or centre.get("lon") is None:
            continue
        tags = element.get("tags", {})
        rows.append(
            {
                "lon": float(centre["lon"]),
                "lat": float(centre["lat"]),
                "kind": "turbine" if tags.get("generator:source") == "wind" else "platform",
                # The source's own words, kept as a flag rather than paraphrased. It is the
                # single most important thing this file has to say about itself.
                "approximate": "approximate" in str(tags.get("note", "")),
                "osm_id": f"{element['type']}/{element['id']}",
            }
        )
    return (
        pd.DataFrame(rows, columns=list(COLUMNS)).sort_values(["lat", "lon"]).reset_index(drop=True)
    )
