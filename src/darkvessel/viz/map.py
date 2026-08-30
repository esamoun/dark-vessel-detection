"""GeoJSON export for the portfolio web map.

Static output, no backend: matched detections in one layer, unmatched in another. The map is
the proof that the pipeline produces something real.

Two files, written together and read as one: a GeoJSON of every detection, which QGIS and
anything else opens, and an HTML page that draws it over a basemap. Nothing else — no service to
wake, no job to schedule, no build step. A demo that sleeps and takes forty seconds to answer is
worse than no demo, and it is worse in the specific way that matters here: the thing being shown
is that the chain found something, and a spinner is indistinguishable from having found nothing.

Three decisions in here would each pass review as correct and publish a false statement:

**The export is reprojected.** The chain works in EPSG:25832 because a match tolerance in degrees
is not a tolerance. GeoJSON is WGS84 by specification and carries no way to say otherwise, so a
layer written as it stands puts the northern Kattegat off the coast of Ghana while the terminal
reports 189 detections written.

**The page carries its own data.** It could fetch the GeoJSON beside it in three lines. Opened
from a disk — which is how anyone checks a page before it is published — the browser's origin
rules refuse that read and the map draws an empty sea. So the same collection is inlined into the
page as well as written beside it, and `test_the_page_and_the_geojson_beside_it_hold_the_same
_detections` is what stops the two drifting apart.

**`unsearched` is not `dark`.** `fusion/match.py` keeps those apart because a run with nothing to
search that called its detections dark would report a sea full of undeclared vessels. This page
is the one output where that error is read by people with no way to check it, so the third status
is drawn in its own colour rather than folded into either of the others.

What is deliberately not published is the MMSI. A matched detection is a vessel that declared
itself, and naming it on a public page adds nothing to the demo — the finding is the detections
nobody declared. The identifier stays in the GeoPackage, where an analyst who needs it has it.

Leaflet is vendored, under `viz/vendor/`, and copied out beside the page. It was on a CDN, pinned
by version and by subresource hash, and a pin stops a script being substituted while doing nothing
about it being absent — at which point the page keeps its table and loses its map. The basemap
tiles are the one thing left that this repository does not hold, and they cannot be: a basemap is
a tile server by definition. Without them the page draws its detections on an empty ground, with
the coordinates, the dates and the scenes all still on it.
"""

import base64
import hashlib
import html
import json
import math
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED

# GeoJSON's one coordinate reference system. Named rather than assumed, because the whole of the
# reprojection below is a claim about which CRS the file is in.
WGS84 = "EPSG:4326"

GEOJSON_NAME = "detections.geojson"
PAGE_NAME = "index.html"

# Six decimal places is a tenth of a metre at this latitude, against detections placed to the
# nearest 10 m pixel. Enough precision to be honest and not enough to suggest the position is
# better known than it is; it also halves the file.
PLACES = 6

# How many decimals each measurement is published to. A match distance of 136.51302541327746 m
# is a claim about femtometres made of a detection placed to the nearest 10 m pixel, and a reader
# has no way to tell a digit that means something from one that fell out of a float. Rounded here
# rather than in the page, so that the GeoJSON somebody downloads carries the same statement the
# page does. Anything not named keeps its value as it stands.
PRECISION = {
    "tolerance_m": 1,
    "score": 4,
    "match_distance_m": 1,
    "length_m": 1,
    "azimuth_shift_m": 1,
    "distance_to_shore_m": 0,
    "depth_m": 1,
}

# What travels to the page, in the order a popup reads them. `status`, `acquired_at`, `scene` and
# `tolerance_m` are the ticket's four; the rest are what makes a row believable rather than
# decorative — a match at 120 m inside a 200 m radius is a different claim from one at 199 m.
# A column the layer does not carry is skipped rather than invented, so the synthetic run's
# output, which never met the contextual sampling, exports the same way as the archive's.
PUBLISHED = (
    "status",
    "acquired_at",
    "scene",
    "tolerance_m",
    "score",
    "match_distance_m",
    "length_m",
    "position_basis",
    "azimuth_shift_m",
    "distance_to_shore_m",
    "depth_m",
)

# The palette the figures in docs/figures already use, so that a bar in the analysis and a dot on
# the map are the same colour for the same thing. Blue and red rather than green and red: the
# most common colour blindness is on that axis, and the two classes here are the whole point of
# the page.
COLOURS = {
    MATCHED: "#1f6feb",
    DARK: "#e5534b",
    UNSEARCHED: "#8b949e",
}

# Leaflet, vendored, and copied out beside the page it is asked for.
#
# It was on a CDN, pinned by version and by subresource hash, which stops it being *substituted*
# and does nothing about it being *absent*. The page is meant to be published once and left
# alone, and the CARTO basemap below is the standing evidence that a third party's terms outlive
# nobody's attention: it started answering every tile with "API KEY REQUIRED" and the page went
# on loading. What that costs is 188 KB of somebody else's code in this repository, under its own
# BSD licence, which nothing will ever update — stated here rather than discovered by whoever
# next reads a security advisory about it.
LEAFLET = Path(__file__).resolve().parent / "vendor" / "leaflet-1.9.4"

# What those files have to be, written in the subresource-integrity form leafletjs.com publishes
# so the pin can be checked against upstream by eye. Verified here, at the moment the page is
# written, rather than by an `integrity` attribute in the page: served from beside the page these
# are same-origin files, and an attribute a viewer resolved differently would fail closed to a
# blank map — the exact failure vendoring them was meant to remove.
LEAFLET_CHECKSUMS = {
    "leaflet.css": "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=",
    "leaflet.js": "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=",
}

# OpenStreetMap's own tiles, which need no account and no key.
#
# The first version of this page used CARTO's Positron, which is the quieter basemap and the
# better backdrop for a scatter of points over water. It renders, today, as a grey field with
# "API KEY REQUIRED" written across it in every tile — a page that loads, draws its detections
# in the right places, and is worthless. That is the whole failure mode issue #8 exists to avoid,
# and it arrived through a dependency rather than through a server of ours, which is the part
# worth remembering: "no backend" is not the same as "nothing can go dark".
#
# So the basemap is the one with no gate in front of it. Its usage policy asks for light traffic
# and attribution, and a static page of one study area is what that policy is for.
TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

# The order the classes are drawn in, and it is not alphabetical: matched first, so that a dark
# candidate standing on top of one is the dot that survives. The page exists to show the second
# class, and a finding hidden under a dot the reader is not looking for is a finding nobody has.
#
# Taken from `fusion/match.py`'s own names and handed to the page as data rather than written out
# again in the script. Written out again, renaming a status there would put every marker in the
# wrong group, draw nothing at all, and fail no test in this repository.
ORDER = (UNSEARCHED, MATCHED, DARK)

# The frame the opening zoom is computed against. Not a measurement of anybody's browser — it is
# the size this page is laid out for, and the zoom that follows from it is a whole number that
# errs towards showing more water rather than less.
NOMINAL_FRAME = (1024, 450)
_TILE = 256
MIN_ZOOM = 3
MAX_ZOOM = 19

# Where the map looks at a collection with nothing in it. Only ever reached by an empty layer —
# any detection at all fits the view to the detections — so it is not a study area, it is the
# water this chain runs over. A page that opened on the null island instead would read as a
# georeferencing fault rather than as a run that found nothing.
EMPTY_VIEW = ((57.6, 11.15), 9)

# Room around the fitted bounds, in pixels, so a detection on the edge of the box does not sit
# half under the frame.
FIT_PADDING = 28

# How the statuses are named to a reader who has not read `fusion/match.py`.
LABELS = {
    MATCHED: "Matched a declared position",
    DARK: "Dark candidate",
    UNSEARCHED: "Not searched",
}


@dataclass(frozen=True)
class Summary:
    """What the page says about itself, taken from the collection it is drawing.

    Derived from the exported features rather than from the layer, so that the caption and the
    dots cannot describe two different runs: the numbers in the header are counted off the same
    bytes that were written to disk.
    """

    detections: int
    counts: dict[str, int]
    acquisitions: int
    first: datetime | None
    last: datetime | None
    tolerances: tuple[float, ...]

    def span(self) -> str:
        """The dates these detections were acquired between, or nothing where none carried one.

        One expression, because the terminal's caption and the page's are exactly the pair this
        module exists to keep from drifting apart.
        """
        if not (self.first and self.last):
            return ""
        return f"{self.first.date().isoformat()} to {self.last.date().isoformat()}"

    def lines(self) -> list[str]:
        """The same sentences the page carries, printed by the command that writes it."""
        span = f", {self.span()}" if self.span() else ""
        found = ", ".join(
            f"{count} {LABELS[status].lower()}"
            if status not in (MATCHED, DARK)
            else f"{count} {status}"
            for status, count in self.counts.items()
        )
        radius = (
            f" at a tolerance of {_metres(self.tolerances)}"
            if self.tolerances
            else " with nothing to match against"
        )
        return [
            f"{self.detections} detections over {_acquisitions(self.acquisitions)}{span}",
            f"  {found}{radius}",
        ]


def collection(detections: gpd.GeoDataFrame) -> dict[str, Any]:
    """The detections as a GeoJSON FeatureCollection, in longitude and latitude.

    A dictionary rather than a string, so that the page and the file are serialised from one
    object and a caller can assert on it without parsing anything.
    """
    if detections.crs is None:
        raise ValueError(
            "these detections carry no CRS, and GeoJSON is longitude and latitude by "
            "specification; there is nothing to reproject from"
        )
    in_degrees = (
        detections
        if CRS.from_user_input(detections.crs).equals(CRS.from_epsg(4326))
        else detections.to_crs(WGS84)
    )

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        round(point.x, PLACES),
                        round(point.y, PLACES),
                    ],
                },
                "properties": {
                    name: _scalar(row[name], PRECISION.get(name))
                    for name in PUBLISHED
                    if name in in_degrees.columns
                },
            }
            for (_, row), point in zip(in_degrees.iterrows(), in_degrees.geometry, strict=True)
        ],
    }


def summarise(exported: dict[str, Any]) -> Summary:
    """Count what the collection holds, in the order the statuses are worth reading."""
    properties = [feature["properties"] for feature in exported["features"]]
    statuses = [row.get("status") for row in properties]
    times = sorted(
        datetime.fromisoformat(row["acquired_at"])
        for row in properties
        if row.get("acquired_at") is not None
    )
    tolerances = {
        float(row["tolerance_m"]) for row in properties if row.get("tolerance_m") is not None
    }

    return Summary(
        detections=len(properties),
        counts={
            status: statuses.count(status)
            for status in (MATCHED, DARK, UNSEARCHED)
            if status in statuses
        },
        acquisitions=_acquisitions_in(properties),
        first=times[0] if times else None,
        last=times[-1] if times else None,
        tolerances=tuple(sorted(tolerances)),
    )


def opening_view(exported: dict[str, Any]) -> tuple[tuple[float, float], int]:
    """Where the map should be looking when it opens, worked out here rather than in a browser.

    Leaflet's `fitBounds` is a function of the frame it is given, and on the published page that
    frame was not what it appeared to be: the fit came back at the maximum zoom, street level over
    the right coordinates, with every detection outside the view. Nothing threw. It read as a sea
    where nothing was found, which is the one wrong answer this page must never give.

    The centre and the zoom are properties of the detections, so they are computed from the
    detections — once, here, where the result can be asserted — and the page opens on them. The
    browser may still improve on it once it knows its own size; it can no longer make it worse.

    Web Mercator, and the zoom is floored so that a rounding error shows more water rather than
    cutting a detection off the edge.
    """
    features = exported["features"]
    if not features:
        return EMPTY_VIEW

    longitudes = [feature["geometry"]["coordinates"][0] for feature in features]
    latitudes = [feature["geometry"]["coordinates"][1] for feature in features]
    centre = ((min(latitudes) + max(latitudes)) / 2, (min(longitudes) + max(longitudes)) / 2)

    # The same padding the browser's own fit leaves, so the two agree on a zoom rather than
    # disagreeing by one and making the page jump when the frame is finally measured.
    width = NOMINAL_FRAME[0] - 2 * FIT_PADDING
    height = NOMINAL_FRAME[1] - 2 * FIT_PADDING
    span_x = max(max(longitudes) - min(longitudes), 1e-6) / 360.0
    span_y = max(_mercator(max(latitudes)) - _mercator(min(latitudes)), 1e-6)
    zoom = min(
        math.floor(math.log2(width / (_TILE * span_x))),
        math.floor(math.log2(height / (_TILE * span_y))),
    )
    return centre, max(MIN_ZOOM, min(MAX_ZOOM, zoom))


def _mercator(latitude: float) -> float:
    """A latitude as a fraction of the Web Mercator world, which is not linear in degrees."""
    radians = math.radians(latitude)
    return 0.5 - math.log(math.tan(math.pi / 4 + radians / 2)) / (2 * math.pi)


def page(exported: dict[str, Any], *, title: str) -> str:
    """One self-contained HTML file: the collection, a basemap, a legend and a table.

    The table is not decoration. Everything the ticket asks the page to show — the acquisition
    date, the scene the detection came from, the radius it was searched at — is in it as plain
    HTML, so a reader who never clicks a marker, or who arrives with scripting turned off, or who
    reaches the page on the morning a tile server is down, still has the four facts in front of
    them. What Leaflet adds is where the detections are, which is the one thing a table cannot
    say.
    """
    summary = summarise(exported)

    return _filled(
        _TEMPLATE,
        {
            "title": html.escape(title),
            "lede": _lede(summary),
            "legend": _legend(summary),
            "rows": _rows(exported),
            "leaflet": LEAFLET.name,
            "tiles": TILES,
            "tile_attribution": TILE_ATTRIBUTION,
            "colours": json.dumps(COLOURS),
            "labels": json.dumps(LABELS),
            "order": json.dumps(list(ORDER)),
            "fallback": json.dumps(UNSEARCHED),
            "opening_centre": json.dumps(list(opening_view(exported)[0])),
            "opening_zoom": json.dumps(opening_view(exported)[1]),
            "fit_padding": json.dumps(FIT_PADDING),
            # `<` escaped rather than trusted: the collection goes inside a script element, and
            # the scene identifiers in it come off somebody else's product.
            "data": json.dumps(exported, separators=(",", ":")).replace("<", "\\u003c"),
        },
    )


def write(exported: dict[str, Any], *, out: Path, title: str) -> list[Path]:
    """The GeoJSON and the page beside it, returned in the order they were written.

    Takes the collection rather than the layer, which is the shape `concentration.write` has and
    for the same reason: the caller computes the artefact once and both files are serialised from
    that one object, so the page and the export cannot be two renderings of the same run.
    """
    out.mkdir(parents=True, exist_ok=True)

    geojson_path = out / GEOJSON_NAME
    geojson_path.write_text(json.dumps(exported, indent=2) + "\n")

    page_path = out / PAGE_NAME
    page_path.write_text(page(exported, title=title))

    return [geojson_path, page_path, _leaflet_beside(page_path)]


def _leaflet_beside(page_path: Path) -> Path:
    """Copy the vendored Leaflet next to the page, having checked it is what it claims to be.

    Checked rather than trusted, because the whole argument for carrying 188 KB of somebody
    else's code in this repository is that the page then depends on nothing that can change
    without anyone noticing — and a vendored file nobody verifies is a file nobody notices
    changing. A mismatch stops the write; it does not publish a page around it.
    """
    for name, expected in LEAFLET_CHECKSUMS.items():
        found = _checksum(LEAFLET / name)
        if found != expected:
            raise ValueError(
                f"{LEAFLET / name} hashes to {found}, not the pinned {expected}; this is not the "
                "Leaflet this page was built against, and the page is not written around it"
            )

    beside = page_path.parent / LEAFLET.name
    shutil.copytree(LEAFLET, beside, dirs_exist_ok=True)
    return beside


def _checksum(path: Path) -> str:
    """A file's SHA-256, in the form a subresource-integrity attribute would carry it."""
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256-" + base64.b64encode(digest).decode()


def map_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What the map is asked for, read out of a config file.

    The same shape as the other `*_request_from` functions in `cli.py`, and here for the reason
    they are there: what can be checked without doing the work is checked before the work starts.

    The layer to draw is the archive's if the config has one, and the single run's otherwise. An
    accumulated layer is 49 acquisitions against one, and a map of one acquisition's six
    detections is a screenshot rather than a demonstration — so the archive wins where both are
    named, and `map.detections` overrides both for anyone who wants a particular file.
    """
    settings = config.get("map", {})
    named = (
        settings.get("detections")
        or config.get("archive", {}).get("detections")
        or config.get("run", {}).get("output")
    )
    if named is None:
        raise ValueError(
            "nothing to map: this config names neither map.detections, archive.detections nor "
            "run.output, and the page is drawn from a layer the chain has already written"
        )
    if "out" not in settings:
        raise ValueError(
            "map.out is missing: the page and its GeoJSON have to be written somewhere, and this "
            "is not a path to guess at inside somebody's repository"
        )

    return {
        "detections": (relative_to / str(named)).resolve(),
        "out": (relative_to / str(settings["out"])).resolve(),
        "title": str(settings.get("title", config.get("area", {}).get("name", "Detections"))),
    }


def _scalar(value: Any, places: int | None = None) -> Any:
    """One cell, as something a JSON parser in a browser will accept.

    `json.dumps` writes a bare `NaN` for a missing float and does not consider that an error.
    It is not JSON, every browser refuses the whole file over it, and one dark detection — which
    has no match distance by definition — is enough to empty the map while the export reports
    success.
    """
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and places is not None:
        return round(value, places)
    return value


def _metres(values: Iterable[float]) -> str:
    """A radius, or every radius an accumulated layer was matched at.

    Two runs can be accumulated into one layer at two tolerances. Printing the first would state
    a radius most of the detections were not searched at, which is the kind of caption that
    survives every review because it is a number and it is nearly right.
    """
    written = [
        f"{value:.0f} m" if float(value).is_integer() else f"{value:.1f} m" for value in values
    ]
    if len(written) <= 1:
        return "".join(written)
    return f"{', '.join(written[:-1])} and {written[-1]}"


def _acquisitions(count: int) -> str:
    return "1 acquisition" if count == 1 else f"{count} acquisitions"


def _acquisitions_in(properties: list[dict[str, Any]]) -> int:
    """How many passes of the satellite these detections came from.

    Counted off the scene identifiers where the layer carries them, and off the acquisition
    instants where it does not. `archive-run` puts a scene on every row because an accumulated
    layer cannot be read without one; a single run's output has no such column, and reporting
    "0 scenes" for a page drawing one acquisition is a caption that is wrong in the direction of
    looking like a bug in the chain rather than in the sentence.
    """
    scenes = {row.get("scene") for row in properties if row.get("scene") is not None}
    if scenes:
        return len(scenes)
    return len({row.get("acquired_at") for row in properties if row.get("acquired_at") is not None})


def _lede(summary: Summary) -> str:
    span = (
        f", {summary.first.date().isoformat()} to {summary.last.date().isoformat()}"
        if summary.first and summary.last
        else ""
    )
    sentences = [
        f"<strong>{summary.detections}</strong> detections over "
        f"<strong>{_acquisitions(summary.acquisitions)}</strong> of Sentinel-1{span}."
    ]
    if summary.tolerances:
        sentences.append(
            f"<strong>{summary.counts.get(MATCHED, 0)}</strong> matched a position declared "
            f"over AIS within <strong>{_metres(summary.tolerances)}</strong> of where the radar "
            f"drew them; <strong>{summary.counts.get(DARK, 0)}</strong> did not."
        )
    if UNSEARCHED in summary.counts:
        sentences.append(
            f"{summary.counts[UNSEARCHED]} were never searched — no declarations were supplied "
            "for their acquisition, which is a different statement from finding none."
        )
    return " ".join(sentences)


def _legend(summary: Summary) -> str:
    """A swatch per status, and the checkbox that hides it.

    Only the statuses the layer actually holds. A legend entry for a class with nothing in it
    reads as a class with nothing found in it.
    """
    return "".join(
        f'<label class="key"><input type="checkbox" data-status="{status}" checked>'
        f'<span class="dot" style="background:{COLOURS[status]}"></span>'
        f'{html.escape(LABELS[status])} <span class="count">{count}</span></label>'
        for status, count in summary.counts.items()
    )


def _rows(exported: dict[str, Any]) -> str:
    """Every detection as a table row, oldest acquisition first.

    Written into the file rather than built by script, so that the four facts the ticket asks for
    are on the page whether or not anything runs.
    """
    ordered = sorted(
        enumerate(feature["properties"] for feature in exported["features"]),
        key=lambda pair: (pair[1].get("acquired_at") or "", pair[1].get("status") or ""),
    )
    return "".join(
        f'<tr data-index="{index}" data-status="{html.escape(str(row.get("status", "")))}">'
        f'<td><span class="dot" style="background:'
        f'{COLOURS.get(row.get("status"), COLOURS[UNSEARCHED])}"></span>'
        f"{html.escape(LABELS.get(row.get('status'), str(row.get('status'))))}</td>"
        f"<td>{html.escape(_when(row.get('acquired_at')))}</td>"
        f'<td class="scene">{html.escape(str(row.get("scene") or "—"))}</td>'
        f"<td>{_number(row.get('tolerance_m'), unit='m')}</td>"
        f"<td>{_number(row.get('match_distance_m'), unit='m')}</td>"
        f"<td>{_number(row.get('score'), places=3)}</td>"
        "</tr>"
        for index, row in ordered
    )


def _when(acquired_at: str | None) -> str:
    if not acquired_at:
        return "—"
    moment = datetime.fromisoformat(acquired_at)
    return moment.strftime("%Y-%m-%d %H:%M")


def _number(value: Any, *, unit: str = "", places: int = 0) -> str:
    """One cell of the table, or an em dash where the layer holds nothing.

    The precision is a parameter rather than something read off the unit: a distance to the metre
    and a score to three places are two decisions, and inferring one from the other hides both
    behind whether a caller passed an empty string.
    """
    if value is None:
        return "—"
    return f"{float(value):.{places}f} {unit}".strip()


def _filled(template: str, values: dict[str, str]) -> str:
    """Substitute every `{{name}}` in one pass.

    One pass rather than a chain of `.replace()` calls, so that a value carrying the syntax — a
    title with braces in it, a scene identifier — cannot be re-expanded by a later substitution.
    The CSS and the tile URL hold single braces, which this does not touch.
    """
    return re.sub(r"\{\{(\w+)\}\}", lambda match: values[match.group(1)], template)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<link rel="stylesheet" href="{{leaflet}}/leaflet.css">
<style>
  :root { --ink: #24292f; --muted: #57606a; --line: #d0d7de; --bg: #ffffff; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 26px; margin: 0 0 8px; letter-spacing: -0.01em; }
  p.lede { margin: 0 0 20px; max-width: 76ch; color: var(--ink); }
  p.note { margin: 16px 0 0; max-width: 76ch; color: var(--muted); font-size: 13.5px; }
  .legend { display: flex; flex-wrap: wrap; gap: 18px; margin: 0 0 12px; }
  .key { display: inline-flex; align-items: center; gap: 7px; font-size: 14px; cursor: pointer; }
  .key .count { color: var(--muted); font-variant-numeric: tabular-nums; }
  .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block;
         border: 1px solid rgba(255,255,255,0.85); flex: none; }
  #map { height: 65vh; min-height: 380px; border: 1px solid var(--line); border-radius: 6px;
         background: #eef1f4; }
  #map.absent { display: flex; align-items: center; justify-content: center; height: 120px;
                min-height: 0; color: var(--muted); font-size: 14px; }
  h2 { font-size: 17px; margin: 36px 0 6px; }
  .scroll { max-height: 420px; overflow: auto; border: 1px solid var(--line);
            border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  thead th { position: sticky; top: 0; background: #f6f8fa; text-align: left;
             border-bottom: 1px solid var(--line); padding: 8px 10px; font-weight: 600; }
  td { padding: 6px 10px; border-bottom: 1px solid #eaeef2; white-space: nowrap; }
  tbody tr:hover { background: #f6f8fa; cursor: pointer; }
  td:first-child { display: flex; align-items: center; gap: 7px; }
  td.scene { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
             color: var(--muted); }
  td:nth-child(n+4) { text-align: right; font-variant-numeric: tabular-nums; }
  .popup dt { color: var(--muted); font-size: 11.5px; text-transform: uppercase;
              letter-spacing: 0.04em; margin-top: 6px; }
  .popup dd { margin: 0; font-size: 13px; }
  .popup .scene { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 11px; word-break: break-all; }
</style>
</head>
<body>
<div class="wrap">
  <h1>{{title}}</h1>
  <p class="lede">{{lede}}</p>
  <div class="legend">{{legend}}</div>
  <div id="map"></div>
  <h2>Every detection</h2>
  <div class="scroll">
    <table>
      <thead>
        <tr><th>Status</th><th>Acquired (UTC)</th><th>Scene</th><th>Tolerance</th>
            <th>Match distance</th><th>Score</th></tr>
      </thead>
      <tbody>{{rows}}</tbody>
    </table>
  </div>
  <p class="note"><em>Dark</em> is a claim about evidence, not a verdict: no position declared
  over AIS, interpolated to the instant of acquisition and moved to where the radar would have
  drawn a vessel travelling at that speed, stood within the tolerance above. Every reason a
  detection can be undeclared that is not an undeclared vessel &mdash; a fishing boat under the
  AIS carriage threshold, a gap in the national archive, a fixed structure &mdash; is still on
  the table.</p>
  <p class="note">Written by <code>darkvessel map</code> from the layer the chain produced. This
  page is a file: no backend, no scheduled job, nothing to wake up. The detections are embedded
  in it and also sit beside it as
  <a href="detections.geojson">detections.geojson</a>, which QGIS opens directly.</p>
</div>
<script src="{{leaflet}}/leaflet.js"></script>
<script>
const DETECTIONS = {{data}};
const COLOURS = {{colours}};
const LABELS = {{labels}};
// The statuses, in the order they are drawn, and the one an unrecognised status falls back to.
// Handed down from `fusion/match.py` rather than written out again here: renaming a status there
// and leaving a literal behind in this script would put every marker in the wrong group, or in
// no group at all, and would fail no test in the repository.
const ORDER = {{order}};
const FALLBACK = {{fallback}};

(function () {
  // Built after the page has loaded, and that is the fix rather than a nicety.
  //
  // Leaflet measures its container once, when the map is constructed, and every later
  // correction works from that measurement. Run inline, this script can execute before
  // `leaflet.css` has arrived and before layout has settled — over a CDN it reliably does —
  // and the map is then built against a frame that is not the frame the reader will see. The
  // markers enter the SVG renderer as degenerate paths, `M0 0`, and stay there: 188 of 189
  // detections invisible on a map that is otherwise perfect, at the right place, at the right
  // scale, with a clean console. The identical file served from a local disk drew all 189.
  //
  // Three attempts to correct the measurement afterwards — re-measuring, re-measuring on
  // `load`, following the frame with a ResizeObserver — all looked right and all shipped that
  // page. Nothing after construction rebuilds the renderer's own bounds. So construction
  // waits instead.
  function start() {
    if (typeof L === 'undefined') {
      // Leaflet is the one thing on this page that comes from somewhere else. If it does not
      // arrive, say so where the map would have been instead of dying at the next line: every
      // detection is in the table below, with the date, the scene and the radius it was searched
      // at, so the page is diminished rather than empty.
      const frame = document.getElementById('map');
      frame.className = 'absent';
      frame.textContent = 'The map library did not load. Every detection is listed below.';
      return;
    }

    const map = L.map('map', { scrollWheelZoom: false });

    // The view is set before anything is added to the map, and that order is the fix rather than a
    // tidiness. A Leaflet layer added to a map that has no view yet is projected against no
    // projection: the markers went into the renderer as degenerate paths — `M0 0`, the value its
    // SVG renderer writes for a shape it considers outside the frame — and stayed that way after
    // the view arrived. 188 of 189 detections were invisible on a map that was otherwise correct,
    // at the right place and the right scale, with a clean console.
    map.setView({{opening_centre}}, {{opening_zoom}});

    L.tileLayer('{{tiles}}', { maxZoom: 19, attribution: '{{tile_attribution}}' }).addTo(map);

    const groups = {};
    const markers = [];

    function esc(value) {
      return String(value === null || value === undefined ? '\u2014' : value)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function metres(value) {
      return value === null || value === undefined ? '\u2014' : value.toFixed(0) + ' m';
    }

    function popup(properties) {
      const rows = [
        ['Status', esc(LABELS[properties.status] || properties.status)],
        ['Acquired (UTC)', esc((properties.acquired_at || '').replace('T', ' ').slice(0, 16))],
        ['Scene', '<span class="scene">' + esc(properties.scene) + '</span>'],
        ['Match tolerance', metres(properties.tolerance_m)],
        ['Match distance', metres(properties.match_distance_m)],
        ['Declared length', metres(properties.length_m)],
        ['Detector score', properties.score === null ? '\u2014' : properties.score.toFixed(3)]
      ];
      return '<dl class="popup">' + rows.map(function (row) {
        return '<dt>' + row[0] + '</dt><dd>' + row[1] + '</dd>';
      }).join('') + '</dl>';
    }

    function groupFor(status) {
      return groups[status] || groups[FALLBACK];
    }

    ORDER.forEach(function (status) { groups[status] = L.layerGroup().addTo(map); });

    DETECTIONS.features.forEach(function (feature, index) {
      const properties = feature.properties;
      const position = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
      const marker = L.circleMarker(position, {
        radius: 6,
        weight: 1.5,
        color: '#ffffff',
        fillColor: COLOURS[properties.status] || COLOURS[FALLBACK],
        fillOpacity: 0.92
      }).bindPopup(popup(properties));
      marker.addTo(groupFor(properties.status));
      markers[index] = marker;
    });

    // Leaflet measures its container once, when the map is constructed, and caches that size. This
    // frame is sized in `vh`, so a browser that has not settled its viewport by the time this
    // script runs reports a height of zero — and `fitBounds` against a viewport of zero returns
    // the *maximum* zoom rather than failing. The map then opens at street level, over exactly the
    // right coordinates, with every detection off the edge of the frame. Nothing throws and the
    // console stays clean. It looks like a map of a sea where nothing was found, which is the one
    // wrong answer this page must never give.
    //
    // Measuring once more is not enough, and that was the first attempt: by the time `load` fires
    // the frame can still be zero. What works is measuring again whenever the frame actually
    // changes size, which is also what makes the page survive a phone rotating and a desktop
    // window being dragged wider.
    // The view above is computed from the detections when this page is written, not derived here
    // from a frame whose size the browser may not yet know. `fitBounds` against a frame that is not
    // what it appears to be returns the *maximum* zoom rather than failing, and the page then shows
    // street level over the right coordinates with every detection outside it — no error, no
    // console output, indistinguishable from a run that found nothing. Two attempts to make that
    // measurement reliable both looked correct and both shipped exactly that page.
    //
    // So the map opens on the view that was worked out from the data, and the browser is only ever
    // allowed to improve on it.
    let touched = false;
    map.on('zoomstart dragstart', function () { touched = true; });

    function refine() {
      if (touched || !markers.length) { return; }
      map.invalidateSize();
      const size = map.getSize();
      // A frame this small is not a frame; fitting to it is what produced the failure above.
      if (size.x < 200 || size.y < 200) { return; }
      map.fitBounds(L.featureGroup(markers).getBounds(), {
        padding: [{{fit_padding}}, {{fit_padding}}]
      });
    }

    refine();
    window.addEventListener('load', refine);
    if (window.ResizeObserver) {
      new ResizeObserver(refine).observe(document.getElementById('map'));
    }

    document.querySelectorAll('.key input').forEach(function (box) {
      box.addEventListener('change', function () {
        const group = groups[box.dataset.status];
        if (!group) { return; }
        if (box.checked) { map.addLayer(group); } else { map.removeLayer(group); }
      });
    });

    document.querySelectorAll('tbody tr').forEach(function (row) {
      row.addEventListener('click', function () {
        const marker = markers[Number(row.dataset.index)];
        if (!marker) { return; }
        map.setView(marker.getLatLng(), Math.max(map.getZoom(), 12));
        marker.openPopup();
      });
    });
  }

  if (document.readyState === 'complete') {
    start();
  } else {
    window.addEventListener('load', start);
  }
})();
</script>
</body>
</html>
"""
