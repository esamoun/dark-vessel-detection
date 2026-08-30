"""Whose water a detection is standing in, and the three ways that answer goes quietly wrong.

The variable this covers was empty on every row for three days — `unavailable` on all 189
detections — because the catalogue that answers the other three carries no boundaries. What is
tested here is the replacement, and the failures it can have are not the failures a raster
sampler has.

*A zone that does not reach the box is dropped.* The service filters on bounding boxes, so a
rectangle in the Kattegat comes back with the Russian and the Alaskan EEZ attached: their
polygons wrap the antimeridian and their bounding boxes cover most of the hemisphere. Kept, they
would clip to nothing or, worse, to a sliver, and detections would be assigned water they are
thousands of kilometres from.

*The high seas and an unfetched sea are different sentences.* The first says the position is
outside every zone, which is an answer. The second says this file was never asked about that
water. The column has carried two words for that distinction since the sampling was written, and
a fetch clipped to a rectangle introduces a third case — a detection outside the rectangle — that
belongs on the `unavailable` side and would land silently on the other.

*Two zones claiming one position is the finding, not a tie to break.* Boundaries do overlap;
Marine Regions carries `Joint regime` polygons for exactly that. A rule that took the first, or
the smallest, would be this repository quietly deciding a maritime boundary.
"""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon
from shapely.geometry import box as rectangle

from darkvessel.context.gee_layers import EEZ, HIGH_SEAS, UNAVAILABLE
from darkvessel.context.zones import CLAIMED_BY, attach, zones_request_from
from darkvessel.data import eez as source
from darkvessel.data.area import Bounds
from darkvessel.data.eez import LICENCE, PROVENANCE, Zones, fetch, load

WORKING_CRS = "EPSG:25832"

# The study box, and the two waters it is split between. The real one straddles the Denmark and
# Sweden boundary, which is what makes this variable worth answering rather than constant.
BOX = Bounds(west=11.0, south=57.55, east=11.3, north=57.70)
WEST = rectangle(10.0, 57.0, 11.15, 58.0)
EAST = rectangle(11.15, 57.0, 12.0, 58.0)
# A polygon whose bounding box meets nothing here. It stands for what the service actually
# returns beside the two above.
ELSEWHERE = Polygon([(50.0, 12.0), (54.0, 12.0), (54.0, 16.0), (50.0, 16.0)])


def answer(*zones: tuple[str, Polygon]) -> str:
    """A canned WFS reply, in the shape and the letter case the service answers in."""
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": polygon.__geo_interface__,
                    "properties": {"sovereign1": name, "geoname": f"{name} EEZ", "mrgid": index},
                }
                for index, (name, polygon) in enumerate(zones)
            ],
        }
    )


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch):
    """The service, stood in for. Nothing in this file reaches a network."""

    def serve(reply: str) -> list[dict]:
        sent: list[dict] = []

        def _get(source_url: str, layer: str, bounds: Bounds) -> str:
            sent.append({"source": source_url, "layer": layer, "bounds": bounds})
            return reply

        monkeypatch.setattr(source, "_get", _get)
        return sent

    return serve


def detections_at(*positions: tuple[float, float]) -> gpd.GeoDataFrame:
    """Detections in degrees, with the three sampled columns already on them."""
    return gpd.GeoDataFrame(
        {
            "score": [0.95] * len(positions),
            "distance_to_shore_m": [31_669.5] * len(positions),
            "depth_m": [-38.0] * len(positions),
            "fishing_hours": [57.7] * len(positions),
        },
        geometry=gpd.points_from_xy([x for x, _ in positions], [y for _, y in positions]),
        crs="EPSG:4326",
    )


def boundaries(*zones: tuple[str, Polygon], covers: Bounds = BOX) -> Zones:
    return Zones(
        zones=gpd.GeoDataFrame(
            {"sovereign1": [name for name, _ in zones]},
            geometry=[polygon for _, polygon in zones],
            crs="EPSG:4326",
        ),
        covers=covers,
        provenance=dict.fromkeys(PROVENANCE, "fixture"),
    )


class TestFetchingThem:
    def test_a_zone_whose_polygon_never_reaches_the_box_is_dropped(self, served) -> None:
        served(answer(("Denmark", WEST), ("Sweden", EAST), ("Yemen", ELSEWHERE)))

        fetched = fetch(BOX)

        assert fetched.names("sovereign1") == ["Denmark", "Sweden"]

    def test_the_boundaries_are_clipped_to_the_rectangle_they_were_asked_for(self, served) -> None:
        """The Danish EEZ is 104 229 km2 and the question is which side of a line 17 km of
        Kattegat falls on. What is kept is the answer and no more of somebody else's product."""
        served(answer(("Denmark", WEST), ("Sweden", EAST)))

        fetched = fetch(BOX)

        assert fetched.zones.total_bounds == pytest.approx(BOX.as_rectangle(), abs=1e-9)

    def test_the_licence_and_what_it_does_not_settle_travel_on_every_row(self, served) -> None:
        """The file is the only thing that will still be on somebody's disk in a year. "The data
        has no legal value whatsoever" is the publisher's sentence, not this project's hedge."""
        served(answer(("Denmark", WEST)))

        fetched = fetch(BOX)

        assert list(fetched.zones["licence"]) == [LICENCE]
        assert "no legal value whatsoever" in fetched.zones["terms"].iloc[0]
        assert "10.14284/632" in fetched.zones["citation"].iloc[0]
        assert fetched.zones["retrieved_at"].iloc[0].startswith("20")

    def test_the_bounding_box_is_sent_longitude_first(self, monkeypatch) -> None:
        """WFS 1.1.0 is specified to take EPSG:4326 latitude first. This service takes longitude
        first, and asked the other way round it does not fail — it returns Yemen."""
        asked: list[str] = []

        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return answer(("Denmark", WEST)).encode()

        def urlopen(url: str, timeout: float = 0):
            asked.append(url)
            return Reply()

        monkeypatch.setattr(source.urllib.request, "urlopen", urlopen)
        fetch(BOX)

        assert "bbox=11.0%2C57.55%2C11.3%2C57.7%2CEPSG%3A4326" in asked[0]

    def test_the_file_remembers_how_much_water_it_was_asked_about(
        self, served, tmp_path: Path
    ) -> None:
        """Without it, a detection outside the fetch cannot be told from one on the high seas."""
        served(answer(("Denmark", WEST), ("Sweden", EAST)))
        path = tmp_path / "eez.gpkg"

        fetch(BOX).write(path)
        read_back = load(path)

        assert read_back.names("sovereign1") == ["Denmark", "Sweden"]
        assert read_back.covers.as_rectangle() == pytest.approx(BOX.as_rectangle(), abs=1e-9)

    def test_a_fetch_that_found_no_zone_still_says_where_it_came_from(
        self, served, tmp_path: Path
    ) -> None:
        """The file most in need of explaining itself is the one with nothing in it. Read off the
        first row, its provenance would be blank exactly then, because there is no first row."""
        served(answer(("Yemen", ELSEWHERE)))
        path = tmp_path / "eez.gpkg"

        fetch(BOX).write(path)
        read_back = load(path)

        assert len(read_back) == 0
        assert read_back.provenance["licence"] == LICENCE
        assert "no legal value whatsoever" in read_back.provenance["terms"]

    def test_a_reference_that_was_never_fetched_says_how_to_fetch_it(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="darkvessel eez"):
            load(tmp_path / "never-fetched.gpkg")

    def test_a_field_named_in_capitals_matches_a_service_that_answers_in_lower_case(self) -> None:
        """The shipped config says SOVEREIGN1, taken from the shapefile. The WFS answers
        `sovereign1`, and a strict match would name no water at all on every row."""
        assert boundaries(("Denmark", WEST)).column("SOVEREIGN1") == "sovereign1"

    def test_a_field_that_is_not_there_says_which_ones_are(self) -> None:
        with pytest.raises(ValueError, match="sovereign1"):
            boundaries(("Denmark", WEST)).column("SOVREIGN1")


class TestStandingInThem:
    def test_a_detection_inside_a_zone_carries_its_name(self) -> None:
        zoned = attach(
            detections_at((11.05, 57.6), (11.25, 57.6)),
            boundaries(("Denmark", WEST), ("Sweden", EAST)),
            field="SOVEREIGN1",
        )

        assert list(zoned[EEZ]) == ["Denmark", "Sweden"]

    def test_a_detection_in_covered_water_and_no_zone_is_on_the_high_seas(self) -> None:
        """An answer, not the absence of one."""
        zoned = attach(
            detections_at((11.25, 57.6)),
            boundaries(("Denmark", WEST)),
            field="SOVEREIGN1",
        )

        assert zoned.loc[0, EEZ] == HIGH_SEAS

    def test_a_detection_beyond_what_was_fetched_is_unavailable_and_not_the_high_seas(
        self,
    ) -> None:
        """The third case a clipped fetch introduces. Landing on the other side of this would
        publish "outside every zone" about water nobody looked at."""
        zoned = attach(
            detections_at((11.05, 57.6), (14.0, 57.6)),
            boundaries(("Denmark", WEST), covers=BOX),
            field="SOVEREIGN1",
        )

        assert list(zoned[EEZ]) == ["Denmark", UNAVAILABLE]
        assert HIGH_SEAS not in list(zoned[EEZ])

    def test_two_zones_claiming_one_position_are_both_reported(self) -> None:
        """Boundaries overlap, which is why Marine Regions carries joint-regime polygons at all.
        Reported rather than resolved: a tie-break here is this repository deciding a maritime
        boundary in a sort order."""
        overlapping = boundaries(
            ("Denmark", rectangle(11.0, 57.5, 11.2, 57.7)),
            ("Sweden", rectangle(11.1, 57.5, 11.3, 57.7)),
        )

        zoned = attach(detections_at((11.15, 57.6)), overlapping, field="SOVEREIGN1")

        assert zoned.loc[0, EEZ] == f"Denmark{CLAIMED_BY}Sweden"

    def test_the_three_sampled_variables_are_left_exactly_as_they_were_found(self) -> None:
        """The whole reason this is not part of `darkvessel context`: filling one column must not
        require an Earth Engine account to refill the other three."""
        detections = detections_at((11.05, 57.6))

        zoned = attach(detections, boundaries(("Denmark", WEST)), field="SOVEREIGN1")

        for column in ("distance_to_shore_m", "depth_m", "fishing_hours", "score"):
            assert list(zoned[column]) == list(detections[column])

    def test_detections_are_placed_in_degrees_whatever_the_chain_was_working_in(self) -> None:
        """The chain works in EPSG:25832 and the boundaries are published in degrees. A join in
        the wrong frame puts every detection outside every polygon and calls the lot high seas."""
        placed = detections_at((11.05, 57.6)).to_crs(WORKING_CRS)

        zoned = attach(placed, boundaries(("Denmark", WEST)), field="SOVEREIGN1")

        assert zoned.crs == placed.crs
        assert zoned.loc[0, EEZ] == "Denmark"


class TestWhatTheConfigAsksFor:
    def test_the_shipped_config_names_a_file_outside_the_repository(self) -> None:
        """Marine Regions asks not to be redistributed, so the boundaries land where the scenes
        and the AIS archive land: under data/, which git ignores."""
        config = Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml"
        import yaml

        request = zones_request_from(yaml.safe_load(config.read_text()), config.parent)

        assert request["reference"].parts[-3:] == ("data", "eez", "kattegat-lane.gpkg")
        assert request["field"] == "SOVEREIGN1"
        assert request["margin_m"] == 5000

    def test_a_config_naming_no_reference_is_refused(self) -> None:
        with pytest.raises(ValueError, match="context.eez.reference"):
            zones_request_from({"context": {"eez": {}}}, Path("."))
