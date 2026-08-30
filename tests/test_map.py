"""The chain's one output that is not for an analyst, and the ways it would lie quietly.

A GeoPackage that is wrong opens in QGIS beside the imagery and looks wrong. A web page that is
wrong looks like a web page. Every test here holds a decision that would pass review as
"correct" and publish a false statement about the sea.

*The export is in longitude and latitude.* The chain works in EPSG:25832 and GeoJSON is WGS84 by
specification, with no room in the format to say otherwise. Written unprojected, the northern
Kattegat lands off the coast of Ghana — or, for a viewer that clamps to valid degrees, nowhere at
all, and the terminal still reports 189 detections written.

*A measurement nobody made is null.* A NaN is not JSON and `json.dumps` writes the bare literal
`NaN` without complaining. Every parser in every browser then refuses the whole file, so the map
is empty while the export says it succeeded.

*Unsearched is not dark.* `fusion/match.py` keeps those two apart for the reason it states: a run
with no declarations that called its detections dark would publish a sea full of undeclared
vessels. The web map is the one place that error reaches people who cannot check it.

*The page carries its own data.* Opened from a disk, a page that fetches a sibling GeoJSON is
refused by the browser's own origin rules and renders an empty basemap. This demo has to survive
being double-clicked, which is what "no backend" means when nobody is watching.
"""

import base64
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from shapely import Point

from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED
from darkvessel.viz.map import (
    COLOURS,
    EMPTY_VIEW,
    GEOJSON_NAME,
    LEAFLET,
    LEAFLET_CHECKSUMS,
    NOMINAL_FRAME,
    ORDER,
    PAGE_NAME,
    PUBLISHED,
    collection,
    map_request_from,
    opening_view,
    page,
    summarise,
    write,
)

WORKING_CRS = "EPSG:25832"

# A point in the middle of the Kattegat study box, in the CRS the chain works in, and the
# longitude and latitude that it is. The trip between these two is the whole job of the export,
# and it is the one thing here no viewer would report as an error.
EASTING, NORTHING = 628_000.0, 6_388_000.0
LONGITUDE, LATITUDE = 11.142818, 57.616169

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml"

ACQUIRED = datetime(2026, 8, 9, 5, 31, 24, tzinfo=UTC)
SCENE = "S1C_IW_GRDH_1SDV_20260809T053124_20260809T053149_008914_011AED_6908"


def layer(
    statuses: tuple[str, ...] = (MATCHED, DARK),
    *,
    scene: str = SCENE,
    acquired_at: datetime = ACQUIRED,
    tolerance_m: float = 200.0,
    crs: str = WORKING_CRS,
) -> gpd.GeoDataFrame:
    """A classified layer of the shape `fusion/match.py` hands on, one row per status.

    Built by hand rather than by running the chain, so that a test about the export cannot pass
    because the pipeline happened to be in a good mood.
    """
    searched = [status != UNSEARCHED for status in statuses]
    return gpd.GeoDataFrame(
        {
            "score": [0.95 for _ in statuses],
            "status": list(statuses),
            "mmsi": ["219000000" if status == MATCHED else None for status in statuses],
            "length_m": [180.0 if status == MATCHED else np.nan for status in statuses],
            "match_distance_m": [120.0 if status == MATCHED else np.nan for status in statuses],
            "tolerance_m": [tolerance_m if ok else np.nan for ok in searched],
            "position_basis": ["interpolated" if s == MATCHED else None for s in statuses],
            "azimuth_shift_m": [180.0 if s == MATCHED else np.nan for s in statuses],
            "acquired_at": [acquired_at for _ in statuses],
            "scene": [scene for _ in statuses],
            "distance_to_shore_m": [31_669.5 for _ in statuses],
            "depth_m": [-38.0 for _ in statuses],
        },
        geometry=[Point(EASTING + 400.0 * i, NORTHING) for i, _ in enumerate(statuses)],
        crs=crs,
    )


def properties(exported: dict) -> list[dict]:
    return [feature["properties"] for feature in exported["features"]]


def test_the_export_is_longitude_and_latitude_whatever_the_chain_was_working_in() -> None:
    """GeoJSON is WGS84 and carries no way to say it is anything else.

    The chain works in metres because a tolerance in degrees is not a tolerance. Handing that
    straight to a browser puts the Kattegat in the Atlantic, and nothing in the file objects.
    """
    exported = collection(layer((MATCHED,)))

    longitude, latitude = exported["features"][0]["geometry"]["coordinates"]
    assert longitude == pytest.approx(LONGITUDE, abs=1e-5)
    assert latitude == pytest.approx(LATITUDE, abs=1e-5)


def test_a_layer_already_in_degrees_is_left_where_it_is() -> None:
    """Reprojection is conditional on the CRS rather than unconditional: a second trip through
    the same transformer would move a correct point off the water and look like the first one."""
    exported = collection(layer((MATCHED,), crs="EPSG:4326"))

    longitude, latitude = exported["features"][0]["geometry"]["coordinates"]
    assert (longitude, latitude) == pytest.approx((EASTING, NORTHING), abs=1e-6)


def test_a_measurement_nobody_made_is_null_and_never_the_nan_that_no_browser_accepts() -> None:
    """`json.dumps` writes `NaN` by default, which is not JSON. A single dark detection — which
    has no match distance, by definition — is enough to make the whole file unparseable."""
    text = json.dumps(collection(layer((MATCHED, DARK))))

    reparsed = json.loads(text, parse_constant=_refuse)
    dark = properties(reparsed)[1]
    assert dark["match_distance_m"] is None
    assert dark["length_m"] is None
    assert "NaN" not in text


def _refuse(constant: str) -> None:
    raise AssertionError(f"{constant} is not JSON, and no browser will parse a file holding it")


def test_an_unsearched_detection_is_not_published_as_a_dark_one() -> None:
    """The distinction `fusion/match.py` exists to keep, carried to the one output that is read
    by people who cannot open the layer and check."""
    exported = collection(layer((MATCHED, DARK, UNSEARCHED)))

    assert [row["status"] for row in properties(exported)] == [MATCHED, DARK, UNSEARCHED]
    assert COLOURS[UNSEARCHED] != COLOURS[DARK]
    assert COLOURS[MATCHED] != COLOURS[DARK]


def test_every_feature_carries_the_date_the_scene_and_the_radius_it_was_searched_at() -> None:
    """The three facts the ticket asks the page to show. On the feature rather than in a caption,
    because a dark detection means nothing without the radius that produced it and the page is
    not the place that rule stops applying."""
    exported = collection(layer((MATCHED, DARK)))

    for row in properties(exported):
        assert row["acquired_at"] == ACQUIRED.isoformat()
        assert row["scene"] == SCENE
        assert row["tolerance_m"] == 200.0


def test_a_measurement_is_published_to_the_precision_it_was_made_at() -> None:
    """136.51302541327746 m is a statement about femtometres, made of a detection placed to the
    nearest 10 m pixel. A reader of the file has no way to tell a digit that means something from
    one that fell out of a float, so the rounding happens in the export rather than in the page:
    the GeoJSON somebody downloads has to carry the same claim the page does."""
    detections = layer((MATCHED,))
    detections.loc[0, "match_distance_m"] = 136.51302541327746
    detections.loc[0, "score"] = 0.9810964465141296

    row = properties(collection(detections))[0]

    assert row["match_distance_m"] == 136.5
    assert row["score"] == 0.9811


def test_the_basemap_needs_no_account_and_no_key() -> None:
    """The first version of this page drew CARTO's Positron, which now answers every tile with
    "API KEY REQUIRED" printed across it: a page that loads, places its detections correctly and
    is worthless. "No backend" is not the same as "nothing can go dark", and the only guard
    against that is a basemap with no gate in front of it."""
    rendered = page(collection(layer((MATCHED,))), title="Kattegat")

    assert "tile.openstreetmap.org" in rendered
    assert "cartocdn" not in rendered
    assert "apikey" not in rendered.lower()


def test_the_mmsi_of_a_vessel_that_declared_itself_stays_in_the_geopackage() -> None:
    """A matched vessel is a vessel that did everything right. Naming it on a public page adds
    nothing to the demo — the finding is the detections nobody declared — so the identifier is
    not among the properties published, and the layer in outputs/ still holds it."""
    exported = collection(layer((MATCHED,)))

    assert "mmsi" not in PUBLISHED
    assert "mmsi" not in properties(exported)[0]
    assert "219000000" not in json.dumps(exported)


def test_a_property_cannot_close_the_script_tag_it_is_embedded_in() -> None:
    """The data is inlined into the page, so a scene identifier is a string in a script element.
    Escaped here rather than trusted: the identifiers come off somebody else's product."""
    rendered = page(collection(layer((MATCHED,), scene="</script><b>x")), title="x")

    assert "</script><b>" not in rendered.split("<script")[-1]
    assert rendered.count("</script>") == rendered.count("<script")


def test_the_page_carries_its_own_detections_rather_than_fetching_them() -> None:
    """Double-clicked from a disk, a page that fetched its sibling GeoJSON would be refused by
    the browser's origin rules and draw an empty sea. That is the failure mode the ticket's "no
    backend" is really about, and it looks exactly like a working page."""
    exported = collection(layer((MATCHED, DARK)))

    rendered = page(exported, title="Kattegat")

    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert GEOJSON_NAME not in rendered.split("<script")[-1]
    assert SCENE in rendered


def test_the_page_states_the_date_range_the_scene_count_and_the_tolerance_without_a_click() -> None:
    """Not only in a popup. A reader who never clicks a marker, or who arrives with scripting
    off, still has to be able to read what was searched and how far."""
    detections = layer((MATCHED, DARK))
    detections.loc[1, "acquired_at"] = datetime(2026, 6, 1, 17, 1, 2, tzinfo=UTC)
    detections.loc[1, "scene"] = "S1C_IW_GRDH_1SDV_20260601T170037_20260601T170102_x_y_z"

    rendered = page(collection(detections), title="Kattegat")

    assert "2026-06-01" in rendered
    assert "2026-08-09" in rendered
    assert "200 m" in rendered
    assert "2 acquisitions" in rendered


def test_the_summary_counts_what_was_found_and_over_how_much_water() -> None:
    detections = layer((MATCHED, MATCHED, DARK))
    detections.loc[2, "scene"] = "another-scene"
    detections.loc[2, "acquired_at"] = datetime(2026, 6, 1, 17, 1, 2, tzinfo=UTC)

    summary = summarise(collection(detections))

    assert summary.detections == 3
    assert summary.counts == {MATCHED: 2, DARK: 1}
    assert summary.acquisitions == 2
    assert summary.first.date().isoformat() == "2026-06-01"
    assert summary.last.date().isoformat() == "2026-08-09"
    assert summary.tolerances == (200.0,)


def test_a_single_run_that_names_no_scene_is_one_acquisition_and_not_none() -> None:
    """`archive-run` puts a scene identifier on every row because an accumulated layer cannot be
    read without one. A single run's output has no such column, and "0 scenes" over a page
    drawing one acquisition reads as a fault in the chain rather than in the sentence."""
    detections = layer((MATCHED, DARK)).drop(columns=["scene"])

    assert summarise(collection(detections)).acquisitions == 1


def test_two_runs_at_two_radii_report_both_rather_than_one_of_them() -> None:
    """An accumulated layer can hold acquisitions matched at different tolerances. Printing the
    first one would state a radius that most of the detections were not searched at."""
    detections = layer((MATCHED, DARK))
    detections.loc[1, "tolerance_m"] = 300.0

    assert summarise(collection(detections)).tolerances == (200.0, 300.0)


def test_the_page_and_the_geojson_beside_it_hold_the_same_detections(tmp_path: Path) -> None:
    """Two files written from one layer, and nothing but this test stops them drifting: a page
    regenerated without its export, or the other way round, is a map of one run captioned with
    the numbers of another."""
    written = write(collection(layer((MATCHED, DARK))), out=tmp_path, title="Kattegat")

    assert written == [tmp_path / GEOJSON_NAME, tmp_path / PAGE_NAME, tmp_path / LEAFLET.name]
    exported = json.loads((tmp_path / GEOJSON_NAME).read_text())
    rendered = (tmp_path / PAGE_NAME).read_text()
    assert json.dumps(exported, sort_keys=True) in _embedded(rendered)


def _embedded(rendered: str) -> str:
    """The page's inlined collection, re-serialised in the same order as the file's."""
    opening = "const DETECTIONS = "
    start = rendered.index(opening) + len(opening)
    end = rendered.index(";\n", start)
    payload = json.loads(rendered[start:end].replace("\\u003c", "<"))
    return json.dumps(payload, sort_keys=True)


def test_the_export_opens_as_a_geojson_that_geopandas_reads_back(tmp_path: Path) -> None:
    """The first acceptance criterion, checked by something other than the writer of the file."""
    write(collection(layer((MATCHED, DARK))), out=tmp_path, title="Kattegat")

    read_back = gpd.read_file(tmp_path / GEOJSON_NAME)

    assert read_back.crs.to_epsg() == 4326
    assert list(read_back["status"]) == [MATCHED, DARK]


def test_the_statuses_reach_the_script_as_data_rather_than_as_literals() -> None:
    """`fusion/match.py` owns these three words. Written out a second time in the page's script,
    renaming one there would put every marker in the wrong layer group — or in none — draw a map
    that is subtly or completely wrong, and fail nothing in this repository."""
    rendered = page(collection(layer((MATCHED, DARK))), title="Kattegat")
    script = rendered.split("<script")[-1]

    assert f"const ORDER = {json.dumps(list(ORDER))};" in script
    assert f"const FALLBACK = {json.dumps(UNSEARCHED)};" in script
    assert "['unsearched', 'matched', 'dark']" not in script
    assert "groups.unsearched" not in script


def test_a_page_whose_map_library_never_arrives_still_carries_its_detections() -> None:
    """Leaflet is the one script here that comes from somewhere else. Unguarded, its absence
    throws at the first `L.map(...)` and takes the table's click handlers down with it — so the
    page loses the thing it can still do because of the thing it cannot."""
    rendered = page(collection(layer((MATCHED, DARK))), title="Kattegat")
    script = rendered.split("<script")[-1]

    assert "typeof L === 'undefined'" in script
    assert script.index("typeof L === 'undefined'") < script.index("L.map(")


def test_a_title_carrying_the_template_syntax_is_not_expanded_into_the_page() -> None:
    """Substitution happens once, over the whole template. Done as a chain of replacements, a
    title is put in before the data is, and a title reading `{{data}}` would be handed the whole
    collection — which no reader would report as anything but a strange caption."""
    rendered = page(collection(layer((MATCHED,))), title="{{data}} {{legend}}")

    assert "&lbrace;" not in rendered  # nothing clever; the braces survive as themselves
    assert "<h1>{{data}} {{legend}}</h1>" in rendered


def test_the_view_the_map_opens_on_holds_every_detection() -> None:
    """The property that matters, asserted on the numbers rather than on the browser: at the
    centre and zoom this page ships, every detection is inside the frame it is laid out for.

    It exists because the opposite shipped twice. `fitBounds` against a frame that is not what it
    appears to be returns the *maximum* zoom instead of failing, and the published page then
    showed street level over exactly the right coordinates with all 189 detections outside the
    view — no error, no console output, and indistinguishable from a sea where nothing was found.
    """
    exported = collection(layer((MATCHED, DARK)))
    (latitude, longitude), zoom = opening_view(exported)

    width, height = NOMINAL_FRAME
    scale = 256 * 2**zoom
    for feature in exported["features"]:
        east, north = feature["geometry"]["coordinates"]
        assert abs(east - longitude) / 360.0 * scale <= width / 2
        assert abs(_mercator(north) - _mercator(latitude)) * scale <= height / 2


def _mercator(latitude: float) -> float:
    radians = math.radians(latitude)
    return 0.5 - math.log(math.tan(math.pi / 4 + radians / 2)) / (2 * math.pi)


def test_the_map_is_built_only_once_the_page_has_finished_loading() -> None:
    """Leaflet measures its container once, at construction, and every later correction works
    from that measurement. Run inline, this script can execute before `leaflet.css` has arrived
    and before layout has settled — over a CDN it reliably does — and the markers then enter the
    renderer as degenerate paths and stay there. 188 of 189 detections were invisible on a page
    whose bytes were identical to one that drew all 189 from a local disk.

    Three attempts to correct the measurement after construction all looked right and all shipped
    that page. Nothing after construction rebuilds the renderer's own bounds, so construction is
    what waits.
    """
    rendered = page(collection(layer((MATCHED, DARK))), title="Kattegat")
    script = rendered.split("<script")[-1]

    assert "function start()" in script
    assert "document.readyState === 'complete'" in script
    assert "window.addEventListener('load', start)" in script
    assert script.index("function start()") < script.index("L.map(")


def test_the_map_opens_on_the_computed_view_before_it_measures_anything() -> None:
    """The browser is allowed to improve the view and never to produce it. A frame too small to
    be a frame is refused rather than fitted to, which is the shape the failure took twice."""
    rendered = page(collection(layer((MATCHED, DARK))), title="Kattegat")
    script = rendered.split("<script")[-1]

    assert script.index("map.setView(") < script.index("map.fitBounds(")
    # Before anything is added to it, too. A layer added to a map with no view is projected
    # against no projection, and 188 of 189 markers stayed degenerate after the view arrived.
    assert script.index("map.setView(") < script.index("L.tileLayer(")
    assert script.index("map.setView(") < script.index("L.circleMarker(")
    assert "if (size.x < 200 || size.y < 200) { return; }" in script


def test_the_map_measures_its_frame_again_before_it_fits_the_detections_into_it() -> None:
    """Leaflet caches its container's size at construction. Handed a height of zero — a layout
    that has not settled when the script runs — `fitBounds` returns the maximum zoom, and the map
    opens at street level over the right coordinates with every detection outside the frame. It
    throws nothing. It looks like a sea where nothing was found.

    Caught on the published URL after rendering correctly from a local server every time, which
    is why the fit is also repeated on `load` rather than only made later.
    """
    rendered = page(collection(layer((MATCHED, DARK))), title="Kattegat")
    script = rendered.split("<script")[-1]

    assert script.index("map.invalidateSize()") < script.index("map.fitBounds(")
    # Measuring once more is not enough — that was the first attempt at this, and the published
    # page still opened at street level. The refinement follows the frame instead, and stops as
    # soon as the reader takes hold of the map rather than arguing with them.
    assert "window.addEventListener('load', refine)" in script
    assert "ResizeObserver" in script
    assert "if (touched || !markers.length) { return; }" in script


def test_an_empty_collection_opens_on_water_rather_than_on_the_null_island() -> None:
    """Only ever reached by a layer with nothing in it — any detection fits the view to the
    detections. A map that opened at 0N 0E would read as a georeferencing fault rather than as a
    run that found nothing, which is the more expensive of the two misreadings."""
    rendered = page(collection(layer(())), title="Kattegat")

    (latitude, longitude), zoom = EMPTY_VIEW
    assert f"map.setView([{latitude}, {longitude}], {zoom});" in rendered


def test_the_page_asks_for_no_host_but_the_one_serving_the_basemap() -> None:
    """Leaflet was on a CDN. A pin stops a script being substituted and does nothing about it
    being absent, and an absent Leaflet leaves a page with a table where its map was. The tiles
    are the one host that cannot be vendored — a basemap is a tile server by definition — and
    everything else now travels with the page."""
    rendered = page(collection(layer((MATCHED, DARK))), title="Kattegat")

    hosts = set(re.findall(r"https?://([^/\"' )]+)", rendered))

    assert hosts == {"tile.openstreetmap.org", "www.openstreetmap.org"}
    assert "unpkg" not in rendered
    assert '<script src="leaflet-1.9.4/leaflet.js">' in rendered


def test_leaflet_travels_beside_the_page_rather_than_being_fetched(tmp_path: Path) -> None:
    write(collection(layer((MATCHED,))), out=tmp_path, title="Kattegat")

    beside = tmp_path / LEAFLET.name
    assert (beside / "leaflet.js").exists()
    assert (beside / "leaflet.css").exists()
    # The CSS asks for these three by relative path; without them a default marker or a layers
    # control added later would draw as a broken image rather than as an icon.
    assert (beside / "images" / "marker-icon.png").exists()
    assert (beside / "images" / "layers.png").exists()
    # Somebody else's code, under somebody else's licence, which travels with it.
    assert (beside / "LICENSE").read_text().startswith("BSD 2-Clause License")


def test_the_vendored_leaflet_is_the_one_the_page_was_built_against() -> None:
    """The pinned hashes, against the bytes in the repository. The whole argument for carrying
    188 KB of somebody else's code here is that the page then depends on nothing that can change
    unnoticed, and a vendored file nobody checks is a file nobody notices changing."""
    for name, expected in LEAFLET_CHECKSUMS.items():
        digest = hashlib.sha256((LEAFLET / name).read_bytes()).digest()
        assert "sha256-" + base64.b64encode(digest).decode() == expected


def test_a_vendored_file_that_is_not_what_it_claims_stops_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mismatch is refused rather than published around. The page would still have rendered."""
    monkeypatch.setitem(LEAFLET_CHECKSUMS, "leaflet.js", "sha256-somethingelse")

    with pytest.raises(ValueError, match="not the pinned"):
        write(collection(layer((MATCHED,))), out=tmp_path, title="Kattegat")


def test_the_published_site_sends_its_root_to_the_map_the_config_writes() -> None:
    """`docs/` is what GitHub Pages serves, and the map is one directory inside it, so the root
    of the site is a 404 without this. Checked against `map.out` rather than against a hardcoded
    path: move where the page is written and a redirect still pointing at the old place would be
    a link that resolves to nothing, which is the state this file exists to end.
    """
    docs = Path(__file__).resolve().parents[1] / "docs"
    out = map_request_from(yaml.safe_load(SHIPPED_CONFIG.read_text()), SHIPPED_CONFIG.parent)["out"]
    target = out.relative_to(docs)

    root = (docs / "index.html").read_text()

    assert f'href="{target}/"' in root
    assert f'content="0; url={target}/"' in root


def test_the_shipped_config_maps_the_archive_rather_than_the_single_scene() -> None:
    """49 acquisitions and 189 detections against one acquisition and six. The single-scene run
    is what `configs/pipeline.yaml` has and the archive is what this box is for."""
    config = yaml.safe_load(SHIPPED_CONFIG.read_text())

    request = map_request_from(config, SHIPPED_CONFIG.parent)

    assert request["detections"].name == "kattegat-lane-archive.gpkg"
    assert request["out"].is_absolute()


def test_a_config_with_only_a_single_scene_run_maps_that() -> None:
    relative_to = Path(__file__).resolve().parent / "configs"

    request = map_request_from(
        {"run": {"output": "../outputs/detections.gpkg"}, "map": {"out": "../docs/map"}},
        relative_to,
    )

    assert request["detections"] == relative_to.parent / "outputs" / "detections.gpkg"


def test_a_config_naming_no_layer_at_all_is_refused_before_anything_is_written() -> None:
    with pytest.raises(ValueError, match="nothing to map"):
        map_request_from({"map": {"out": "../docs/map"}}, Path(__file__).resolve().parent)
