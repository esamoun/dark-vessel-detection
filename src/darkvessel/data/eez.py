"""The exclusive economic zones a detection can stand in, and what that publication is.

Issue #16 asked where the dark candidates stand, and one of its four variables came back empty on
every row: Earth Engine's public catalogue carries no EEZ boundaries, so the column read
`unavailable` on all 189 detections and the README said so rather than hiding it. The conclusion
drawn at the time — that the boundaries would have to be ingested into Earth Engine as a table
asset — does not follow. An EEZ is a polygon and membership is a point-in-polygon test, which
needs no catalogue, no credentials and no network once the polygons are on the disk.

**The source is Marine Regions**, the Flanders Marine Institute's Maritime Boundaries Geodatabase,
through its WFS. The layer is `Exclusive Economic Zones (200 NM) (v12, world, 2023)`, quoted from
the service's own GetCapabilities rather than from a page about it. It is CC-BY 4.0, transcribed
from the `LICENSE_EEZ_v12.txt` that ships with the geodatabase rather than from a page about that
either.

**What this module deliberately does not do is keep them.** That same licence file asks, in as many
words, that users "not make our products available for download elsewhere and to always refer to
marineregions.org for the most up-to-date products and services". CC-BY would permit a clipped
copy in this repository and that request is a courtesy rather than a licence term, which is
exactly why it is worth honouring rather than arguing with. So the boundaries are written under
`data/`, where this project already puts other people's bulk — the Sentinel-1 archive and 21 GB of
Danish AIS are both fetched and both ignored by git. What it costs is one command after a clone,
and until it is run the column reads `unavailable`, which is the honest state and is the word that
exists for it.

The provenance travels *inside* the file rather than in a commit message, because the file is the
only thing that will still exist on somebody's disk in a year.

**And what the boundaries do not settle is quoted rather than hedged.** From the same file: "VLIZ
expresses no opinion about the legal state neither of any country, territory or area nor
concerning its delimitation, frontier or borders. The data has no legal value whatsoever." A zone
on a detection says which side of a published line it fell, and nothing else — not who has
jurisdiction over that vessel, and not whether anyone was entitled to be there. That sentence
travels on every row rather than living in a README somebody may not have open.

Two things about this service that are cheaper to read here than to rediscover:

**The bbox filter selects on bounding boxes, not on polygons.** Asked for a rectangle in the
Kattegat it returns six zones, four of which are nowhere near it: the Russian and the Alaskan EEZ
wrap the antimeridian, so their bounding boxes cover most of the northern hemisphere. The
intersection against the real geometry is done here, and a run that trusted the server's filter
would clip the Kattegat out of the Bering Sea.

**WFS 1.1.0 is specified to take a bbox in the axis order of its CRS**, which for EPSG:4326 is
latitude first. This service takes longitude first. Asked the other way round it does not fail —
it answers, with the exclusive economic zone of Yemen.
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box as rectangle

from darkvessel.data.area import Bounds
from darkvessel.detect.checkpoints import atomically

MARINE_REGIONS = "https://geo.vliz.be/geoserver/MarineRegions/wfs"
EEZ_LAYER = "MarineRegions:eez"

DEGREES = "EPSG:4326"

# Two layers, because the file has to answer two different questions. `zones` is which water
# belongs to whom; `coverage` is how much water this file was ever asked about. Without the
# second, a detection outside the fetched rectangle is indistinguishable from one on the high
# seas — the exact conflation between "no zone here" and "not looked at here" that the column's
# two words exist to keep apart.
ZONES_LAYER = "zones"
COVERAGE_LAYER = "coverage"

# Transcribed from LICENSE_EEZ_v12.txt as distributed with the geodatabase, not paraphrased from
# a page about it. Written onto every row of the file, so that a copy on somebody's disk in a year
# can still say what it is and what it is not.
LICENCE = "CC-BY 4.0 - https://creativecommons.org/licenses/by/4.0/"
CITATION = (
    "Flanders Marine Institute (2023). Maritime Boundaries Geodatabase: Maritime Boundaries and "
    "Exclusive Economic Zones (200NM), version 12. https://doi.org/10.14284/632"
)
# The publisher's own limits on what these polygons mean, quoted. This is what answers "what does
# this variable not settle": not this project hedging, but the people who drew the lines saying
# what they drew.
TERMS = (
    "Developed solely for scientific, educational and research purposes; not for legal, economic "
    "or navigational use. VLIZ expresses no opinion about the legal state of any country, "
    "territory or area nor concerning its delimitation, frontier or borders. The data has no "
    "legal value whatsoever."
)

ATTRIBUTION = "Flanders Marine Institute (VLIZ), Marine Regions - https://marineregions.org"

PROVENANCE = ("source", "layer", "retrieved_at", "licence", "citation", "terms", "attribution")


@dataclass(frozen=True)
class Zones:
    """The boundaries, and the rectangle they were asked for.

    The second is not bookkeeping. `zones` alone cannot tell a position outside the fetch from a
    position on the high seas, and those are the two words this variable is made of.
    """

    zones: gpd.GeoDataFrame
    covers: Bounds

    def __len__(self) -> int:
        return len(self.zones)

    def column(self, field: str) -> str:
        """The column a config asked for, matched without regard to case.

        Marine Regions' shapefile spells these in capitals and its WFS answers in lower case, so
        a config naming `SOVEREIGN1` — as this project's did, taken from the shapefile — would
        otherwise match nothing and every detection would come back as unnamed water.
        """
        for name in self.zones.columns:
            if name.lower() == field.lower():
                return name
        raise ValueError(
            f"{field!r} is not a field of these boundaries; they carry "
            f"{sorted(name for name in self.zones.columns if name != 'geometry')}"
        )

    def names(self, field: str) -> list[str]:
        """The distinct zone names in this file, under the field a config names."""
        return sorted({str(value) for value in self.zones[self.column(field)]})

    def write(self, path: Path) -> None:
        """Both layers, into one GeoPackage, whole or not at all."""
        path.parent.mkdir(parents=True, exist_ok=True)
        covered = gpd.GeoDataFrame(
            {name: [self.zones[name].iloc[0] if len(self.zones) else ""] for name in PROVENANCE},
            geometry=[rectangle(*self.covers.as_rectangle())],
            crs=DEGREES,
        )
        with atomically(path, keep_suffix=True) as partial:
            self.zones.to_file(partial, layer=ZONES_LAYER, driver="GPKG")
            covered.to_file(partial, layer=COVERAGE_LAYER, driver="GPKG")


def fetch(bounds: Bounds, *, source: str = MARINE_REGIONS, layer: str = EEZ_LAYER) -> Zones:
    """Every zone whose polygon actually meets this rectangle, clipped to it.

    Clipped rather than kept whole for the reason the request is bounded at all: the Danish EEZ is
    104 229 km2 and the question is which side of a line 17 km of Kattegat falls on. What is kept
    is the answer to that question and no more of somebody else's product than it takes.
    """
    wanted = rectangle(*bounds.as_rectangle())
    answered = gpd.GeoDataFrame.from_features(
        json.loads(_get(source, layer, bounds))["features"], crs=DEGREES
    )

    # The server filtered on bounding boxes. This is the filter that was meant.
    meeting = answered[answered.intersects(wanted)].copy()
    clipped = gpd.clip(meeting, wanted).reset_index(drop=True)
    clipped["source"] = source
    clipped["layer"] = layer
    clipped["retrieved_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    clipped["licence"] = LICENCE
    clipped["citation"] = CITATION
    clipped["terms"] = TERMS
    clipped["attribution"] = ATTRIBUTION
    return Zones(zones=clipped, covers=bounds)


def load(path: Path) -> Zones:
    """The boundaries back off the disk, with the rectangle they were fetched for."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; the EEZ boundaries are fetched rather than committed - "
            "Marine Regions asks not to be redistributed - so run `darkvessel eez` first"
        )
    zones = gpd.read_file(path, layer=ZONES_LAYER)
    west, south, east, north = gpd.read_file(path, layer=COVERAGE_LAYER).total_bounds
    return Zones(zones=zones, covers=Bounds(west=west, south=south, east=east, north=north))


def _get(source: str, layer: str, bounds: Bounds) -> str:
    query = urllib.parse.urlencode(
        {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": layer,
            "outputFormat": "application/json",
            "srsName": DEGREES,
            # Longitude first. See the note in the module docstring: the specification says
            # otherwise and this service does not, and the wrong order returns Yemen rather than
            # an error.
            "bbox": ",".join(str(value) for value in [*bounds.as_rectangle(), DEGREES]),
        }
    )
    with urllib.request.urlopen(f"{source}?{query}", timeout=120) as answer:
        return answer.read().decode()
