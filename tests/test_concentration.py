"""Where the dark candidates concentrate, and how firm each statement about that is.

Every test here is a decision the analysis makes, held so that reverting it fails rather than
producing a slightly different number in a table nobody re-derives. That bar exists because this
is the level whose output is prose: a defect in `detect/` is a wrong box in an image and a defect
here is a sentence in the README that reads exactly as correct.

The four that matter most, and would each survive review as "correct" without a test:

*The interval is resampled over scenes, not over detections.* 189 detections came from 50
acquisitions, and two detections of one acquisition share a sea, a morning and often a vessel.
Treating them as 189 independent trials narrows every interval by about a third, and a finding
that exists only inside that third is an artefact of the arithmetic.

*A band is cut on every detection, not on the dark ones.* Quantiles taken over the dark subset
define the bins by the thing being measured, and the rate per bin then tends to flat by
construction.

*A value nobody could sample is not a band.* The rule `context/gee_layers.py` is built on,
carried through to the analysis that reads its columns: an unsampled variable is reported
unavailable, never as a null result, because "we looked and found nothing" and "we could not
look" are different sentences and only one of them is true of the EEZ.

*Wilson, not the normal approximation.* Bands here carry 40-odd detections and rates near zero,
where a symmetric interval leaves [0, 1] and prints a negative probability.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from shapely import Point

from darkvessel.analysis.concentration import (
    EEZ,
    SPREAD_SEEDS,
    Band,
    concentrate,
    cut,
    interval_over,
    monte_carlo_spread,
    spearman,
    svg,
    wilson,
)
from darkvessel.cli import main
from darkvessel.context.gee_layers import UNAVAILABLE
from darkvessel.detect.geo import DETECTIONS_LAYER, write_detections
from darkvessel.fusion.match import DARK, MATCHED

WORKING_CRS = "EPSG:25832"


def detections(
    *,
    status: list[str],
    scene: list[str],
    shore: list[float] | None = None,
    depth: list[float] | None = None,
    effort: list[float] | None = None,
    eez: list[str] | None = None,
    sea_level: list[float] | None = None,
    sea_spread: list[float] | None = None,
) -> gpd.GeoDataFrame:
    """A layer of the shape `archive-run` writes, with only the columns the analysis reads.

    Positions are a line of points 500 m apart in the working CRS. Nothing about them matters to
    the analysis, which reads columns rather than geometry, and they are real coordinates anyway
    so that a layer written to disk in a test opens in the same tools the real one does.
    """
    count = len(status)
    ones = [1.0] * count
    return gpd.GeoDataFrame(
        {
            "score": [0.96] * count,
            "status": status,
            "scene": scene,
            "distance_to_shore_m": ones if shore is None else shore,
            "depth_m": [-35.0] * count if depth is None else depth,
            "fishing_hours": ones if effort is None else effort,
            "eez": [UNAVAILABLE] * count if eez is None else eez,
            "sea_level_db": [-20.0] * count if sea_level is None else sea_level,
            "sea_spread_db": [2.0] * count if sea_spread is None else sea_spread,
            "geometry": [Point(620_000.0 + 500.0 * n, 6_390_000.0) for n in range(count)],
        },
        crs=WORKING_CRS,
    )


def band_of(result, variable: str, index: int) -> Band:
    (profile,) = [each for each in result.profiles if each.variable == variable]
    return profile.bands[index]


def profile_of(result, variable: str):
    (profile,) = [each for each in result.profiles if each.variable == variable]
    return profile


class TestTheInterval:
    """Wilson over the rows, and a bootstrap over the acquisitions the rows came from."""

    def test_the_interval_stays_inside_zero_and_one(self):
        """A band that saw no dark detection cannot have a negative rate as its lower bound.

        The normal approximation gives 0 +/- 0 here, which is the other failure: an interval of
        zero width around zero reads as certainty from five detections. Wilson is asymmetric and
        gives a band with room in it.
        """
        low, high = wilson(0, 5)
        assert low == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < high < 1.0

        low, high = wilson(5, 5)
        assert 0.0 < low < 1.0
        assert high == pytest.approx(1.0, abs=1e-9)

    def test_an_empty_band_knows_nothing_rather_than_everything(self):
        """No detections is not a rate of zero; both bounds are missing."""
        low, high = wilson(0, 0)
        assert np.isnan(low) and np.isnan(high)

    def test_the_interval_widens_when_the_scenes_disagree(self):
        """The guard on the whole method: resampling scenes, not rows.

        Twelve acquisitions, six of which are entirely dark and six entirely declared. Every row
        is one of 12 coin flips wearing eight faces, not 96 independent trials, and an interval
        that does not know that is far too tight. Revert `interval_over` to resample rows and
        this fails: the row-wise interval on 96 trials at 0.5 is about +/- 0.10, and the true
        scene-wise one is half the range 0 to 1.
        """
        scenes = [f"scene-{n:02d}" for n in range(12)]
        dark = np.array([n < 6 for n in range(12) for _ in range(8)])
        clusters = np.array([scene for scene in scenes for _ in range(8)])

        low, high = interval_over(dark, clusters, draws=2000, seed=0)
        by_row_low, by_row_high = wilson(int(dark.sum()), dark.size)

        assert high - low > 2 * (by_row_high - by_row_low)

    def test_the_scenes_are_resampled_whole(self):
        """A scene drawn twice contributes both its detections twice, not two random rows.

        With every scene internally unanimous, any bootstrap draw can only ever produce a rate
        that is a multiple of 1/12. A row-wise resample would produce rates off that grid almost
        every draw, so the bounds landing on it is the evidence that whole acquisitions moved.
        """
        dark = np.array([n < 6 for n in range(12) for _ in range(8)])
        clusters = np.array([f"scene-{n:02d}" for n in range(12) for _ in range(8)])

        low, high = interval_over(dark, clusters, draws=2000, seed=0)

        twelfths = np.arange(13) / 12
        assert np.isclose(twelfths, low, atol=1e-9).any()
        assert np.isclose(twelfths, high, atol=1e-9).any()

    def test_the_same_seed_gives_the_same_interval(self):
        """A README number that moves between runs of the same command is not a measurement."""
        dark = np.array([n % 3 == 0 for n in range(60)])
        clusters = np.array([f"scene-{n // 4:02d}" for n in range(60)])

        assert interval_over(dark, clusters, draws=500, seed=7) == interval_over(
            dark, clusters, draws=500, seed=7
        )
        assert interval_over(dark, clusters, draws=500, seed=7) != interval_over(
            dark, clusters, draws=500, seed=8
        )


class TestTheBands:
    """How a continuous variable is cut, and what happens to the rows it has no value for."""

    def test_the_bands_are_cut_on_every_detection_not_on_the_dark_ones(self):
        """Quantiles over the dark subset would define the bins by the thing being measured.

        Here the dark detections all sit at the bottom of the range. Cut on them, the four bands
        span 1 to 4 and every declared detection falls in the last one; cut on all 40, the edges
        are the population's quartiles and the top band is above 30.
        """
        values = np.array([float(n) for n in range(1, 41)])
        dark = values <= 4

        edges = cut(values, bands=4)

        assert edges[0] == pytest.approx(1.0)
        assert edges[-1] == pytest.approx(40.0)
        assert edges[2] > 15.0, "the middle edge is the population's median, not the dark one's"
        assert not dark[values > edges[2]].any()

    def test_a_variable_that_never_moves_is_one_band_not_four(self):
        """ETOPO1 over a small box can return one depth. Four bands of it would be three empty
        bands and a table that implies a gradient nobody measured."""
        edges = cut(np.array([-35.0] * 20), bands=4)
        assert len(edges) == 2

    def test_a_value_nobody_could_sample_is_not_a_band(self):
        """The rule `context/gee_layers.py` is built on, carried into the analysis.

        Two of six detections were never sampled for depth. They are counted as unmeasured and
        left out of the bands; folded in as a zero they would be a detection at the waterline,
        and folded in as a band of their own they would be a depth range that does not exist.
        """
        layer = detections(
            status=[DARK, MATCHED, MATCHED, DARK, MATCHED, MATCHED],
            scene=[f"s{n}" for n in range(6)],
            depth=[-40.0, -38.0, np.nan, -33.0, -31.0, np.nan],
        )

        profile = profile_of(concentrate(layer, draws=200, seed=0), "depth_m")

        assert profile.measured == 4
        assert profile.unmeasured == 2
        assert sum(band.total for band in profile.bands) == 4

    def test_a_variable_with_no_values_at_all_is_unavailable_not_flat(self):
        """ "We looked and found nothing" and "we could not look" are different sentences.

        A profile with no measured rows carries no bands and says so. Reported as a flat
        distribution it would be a finding about water nobody sampled.
        """
        layer = detections(
            status=[DARK, MATCHED, MATCHED, MATCHED],
            scene=[f"s{n}" for n in range(4)],
            effort=[np.nan] * 4,
        )

        profile = profile_of(concentrate(layer, draws=200, seed=0), "fishing_hours")

        assert not profile.available
        assert profile.bands == ()
        assert any("unavailable" in line for line in profile.lines())

    def test_the_bands_partition_the_measured_rows(self):
        """Every sampled detection lands in exactly one band. A row that fell through a gap
        between two edges would quietly leave the denominator of a rate."""
        rng = np.random.default_rng(3)
        count = 60
        layer = detections(
            status=[DARK if n % 4 == 0 else MATCHED for n in range(count)],
            scene=[f"s{n // 3:02d}" for n in range(count)],
            shore=list(rng.uniform(21_000.0, 32_000.0, count)),
        )

        profile = profile_of(concentrate(layer, draws=200, seed=0), "distance_to_shore_m")

        assert sum(band.total for band in profile.bands) == count
        assert sum(band.dark for band in profile.bands) == sum(
            1 for each in layer.status if each == DARK
        )


class TestTheZones:
    """The EEZ, which was empty on every row until #35 answered it without Earth Engine.

    It is the one categorical variable here, and everything that makes a banded variable
    trustworthy has to hold for it too: a rate, an interval resampled over acquisitions rather
    than over rows, and a comparison that reports overlap rather than a difference.
    """

    def test_an_unfetched_eez_is_reported_unavailable_rather_than_as_a_finding(self):
        layer = detections(
            status=[DARK, MATCHED, MATCHED, MATCHED],
            scene=[f"s{n}" for n in range(4)],
        )

        result = concentrate(layer, draws=200, seed=0)
        (zones,) = [each for each in result.categories if each.variable == EEZ]

        assert not zones.available
        assert [zone.name for zone in zones.zones] == [UNAVAILABLE]
        assert any("unavailable" in line for line in zones.lines())

    def test_a_sampled_zone_carries_a_rate_and_an_interval_like_every_band(self):
        """Counts alone are not a finding. "A larger share of what is here is undeclared" is a
        claim about a difference, and a difference between two counts with no interval around
        either is arithmetic rather than evidence."""
        layer = detections(
            status=[DARK, MATCHED, MATCHED, DARK],
            scene=[f"s{n}" for n in range(4)],
            eez=["Denmark", "Denmark", "Sweden", "Denmark"],
        )

        result = concentrate(layer, draws=200, seed=0)
        (zones,) = [each for each in result.categories if each.variable == EEZ]
        denmark, sweden = zones.zones

        assert zones.available
        assert (denmark.name, denmark.total, denmark.dark) == ("Denmark", 3, 2)
        assert (sweden.name, sweden.total, sweden.dark) == ("Sweden", 1, 0)
        assert denmark.rate == pytest.approx(2 / 3)
        assert denmark.estimated
        assert 0.0 <= denmark.interval[0] <= denmark.interval[1] <= 1.0

    def test_a_zone_whose_detections_came_from_one_acquisition_has_no_interval(self):
        """The property that says the interval is resampled over acquisitions and not over rows.
        One morning is one draw however many detections came out of it, and no uncertainty is
        measurable from it — which is reported, rather than being handed back as a narrow band."""
        layer = detections(
            status=[DARK, MATCHED, DARK, MATCHED],
            scene=["s0", "s0", "s1", "s2"],
            eez=["Denmark", "Denmark", "Sweden", "Sweden"],
        )

        result = concentrate(layer, draws=200, seed=0)
        (zones,) = [each for each in result.categories if each.variable == EEZ]
        denmark, sweden = zones.zones

        assert not denmark.estimated
        assert sweden.estimated

    def test_two_zones_are_reported_as_overlapping_rather_than_as_a_difference(self):
        """The comparison every finding on this page is stated against, applied to words instead
        of to a range. Two rates that differ inside their intervals are not a finding."""
        layer = detections(
            status=[DARK, MATCHED, MATCHED, DARK, MATCHED, MATCHED],
            scene=[f"s{n}" for n in range(6)],
            eez=["Denmark"] * 3 + ["Sweden"] * 3,
        )

        result = concentrate(layer, draws=200, seed=0)
        (zones,) = [each for each in result.categories if each.variable == EEZ]

        assert zones.comparable
        assert zones.separations == ()
        assert any("no concentration established" in line for line in zones.lines())

    def test_a_zones_interval_follows_its_name_and_not_its_place_in_the_list(self):
        """Seeded by position, adding a zone that sorts early would move the interval of every
        zone after it — including ones this project has already published. `_profile` is safe
        because a band's index is stable; a name sorted alphabetically is not."""
        rows = {
            "status": [DARK, MATCHED, MATCHED, DARK, MATCHED, MATCHED],
            "scene": [f"s{n}" for n in range(6)],
        }
        without = detections(**rows, eez=["Denmark"] * 3 + ["Sweden"] * 3)
        # "Belgium" sorts before both, so under a positional seed it displaces them.
        with_belgium = detections(
            status=[*rows["status"], MATCHED],
            scene=[*rows["scene"], "s6"],
            eez=["Denmark"] * 3 + ["Sweden"] * 3 + ["Belgium"],
        )

        def sweden(layer):
            (zones,) = [
                each
                for each in concentrate(layer, draws=200, seed=0).categories
                if each.variable == EEZ
            ]
            return next(zone for zone in zones.zones if zone.name == "Sweden")

        assert sweden(with_belgium).interval == sweden(without).interval


class TestTheConfound:
    """The sea state, which the archive-wide run put on the row so it could be looked at."""

    def test_the_confound_is_measured_over_scenes_not_over_detections(self):
        """One acquisition has one sea, however many detections came out of it.

        Correlating 189 rows against a value repeated per scene weights each acquisition by its
        own detection count — which is the very quantity under test on one side of the
        correlation. Twelve scenes here carry between one and four detections each, and the
        per-scene answer is the one reported.
        """
        counts = [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4]
        scene = [f"s{index:02d}" for index, many in enumerate(counts) for _ in range(many)]
        levels = [-30.0 + index for index, many in enumerate(counts) for _ in range(many)]
        layer = detections(
            status=[DARK if index % 2 else MATCHED for index in range(len(scene))],
            scene=scene,
            sea_level=levels,
        )

        sea = concentrate(layer, draws=200, seed=0).sea

        assert sea.scenes == 12

    def test_spearman_is_the_rank_correlation_and_needs_no_scipy(self):
        """Monotone but not linear: Pearson on the values gives 0.87 and the rank answer is 1."""
        rising = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spearman(rising, rising**4) == pytest.approx(1.0)
        assert spearman(rising, -(rising**4)) == pytest.approx(-1.0)

    def test_a_correlation_over_one_scene_is_missing_rather_than_zero(self):
        """A single acquisition has no spread to correlate against. Nothing is known, and zero
        would read as "the sea state does not matter", which is a claim."""
        layer = detections(status=[DARK, MATCHED, MATCHED], scene=["s0"] * 3)

        sea = concentrate(layer, draws=200, seed=0).sea

        assert np.isnan(sea.rate_against_level)


class TestWhatIsRefused:
    def test_a_layer_without_a_scene_column_is_refused(self):
        """`run` writes no `scene` column and `archive-run` does. Without it the interval has no
        cluster to resample and would silently fall back to the row-wise one this module exists
        to avoid, so it is an error naming the command that produces the right layer."""
        layer = detections(status=[DARK, MATCHED], scene=["s0", "s1"]).drop(columns="scene")

        with pytest.raises(ValueError, match="archive-run"):
            concentrate(layer, draws=200, seed=0)

    def test_a_layer_with_no_detections_is_refused(self):
        layer = detections(status=[], scene=[])

        with pytest.raises(ValueError, match="no detections"):
            concentrate(layer, draws=200, seed=0)


class TestTheFigure:
    def test_the_bands_run_left_to_right_in_the_order_they_are_reported(self):
        """Stated here as well as in the module, because a figure drawn the other way round is
        the analysis's own opposite and looks entirely correct. The nearest-shore band is the
        leftmost bar, and the test reads the x coordinates back out of the markup."""
        rng = np.random.default_rng(11)
        count = 60
        layer = detections(
            status=[DARK if n < 15 else MATCHED for n in range(count)],
            scene=[f"s{n // 3:02d}" for n in range(count)],
            shore=sorted(rng.uniform(21_000.0, 32_000.0, count)),
        )
        result = concentrate(layer, draws=200, seed=0)

        drawing = svg(profile_of(result, "distance_to_shore_m"))

        centres = [float(part.split('"')[0]) for part in drawing.split('data-band-x="')[1:]]
        assert centres == sorted(centres)
        assert len(centres) == len(profile_of(result, "distance_to_shore_m").bands)

    def test_the_figure_carries_the_interval_and_not_only_the_rate(self):
        """A bar chart of four rates with no whiskers is the figure this analysis must not
        produce: every one of its bands overlaps every other and the picture would deny it."""
        rng = np.random.default_rng(12)
        count = 60
        layer = detections(
            status=[DARK if n % 4 == 0 else MATCHED for n in range(count)],
            scene=[f"s{n // 3:02d}" for n in range(count)],
            depth=list(rng.uniform(-49.0, -31.0, count)),
        )

        drawing = svg(profile_of(concentrate(layer, draws=200, seed=0), "depth_m"))

        assert drawing.count("data-interval") == 4


class TestTheCommand:
    def test_the_command_writes_a_report_and_a_figure_per_variable(self, tmp_path: Path):
        """End to end over a written layer, with no network anywhere in it."""
        rng = np.random.default_rng(5)
        count = 80
        layer = detections(
            status=[DARK if rng.random() < 0.3 else MATCHED for _ in range(count)],
            scene=[f"s{n // 4:02d}" for n in range(count)],
            shore=list(rng.uniform(21_000.0, 32_000.0, count)),
            depth=list(rng.uniform(-49.0, -31.0, count)),
            effort=list(rng.uniform(20.0, 90.0, count)),
            sea_level=[-30.0 + (n // 4) for n in range(count)],
        )
        detections_path = tmp_path / "archive.gpkg"
        write_detections(layer, detections_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "archive": {"detections": "archive.gpkg"},
                    "analysis": {
                        "bands": 4,
                        "draws": 200,
                        "seed": 0,
                        "report": "analysis.json",
                        "figures": "figures",
                    },
                }
            )
        )

        assert main(["analyse", "--config", str(config_path)]) == 0

        report = json.loads((tmp_path / "analysis.json").read_text())
        assert report["detections"] == count
        assert report["scenes"] == 20
        assert {each["variable"] for each in report["profiles"]} == {
            "distance_to_shore_m",
            "depth_m",
            "fishing_hours",
        }
        for each in report["profiles"]:
            assert (tmp_path / "figures" / f"{each['figure']}").exists()

    def test_the_command_says_where_to_look_when_the_run_has_not_been_made(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "archive": {"detections": "missing.gpkg"},
                    "analysis": {"report": "analysis.json", "figures": "figures"},
                }
            )
        )

        with pytest.raises(FileNotFoundError, match="archive-run"):
            main(["analyse", "--config", str(config_path)])

    def test_the_shipped_config_declares_where_the_analysis_writes(self):
        """The same check `test_context.py` makes of the context keys: the config in the
        repository is the one the README's commands run, so its keys are asserted rather than
        assumed."""
        shipped = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml").read_text()
        )

        analysis = shipped["analysis"]
        assert analysis["bands"] >= 2
        assert analysis["draws"] >= 1000
        assert analysis["report"].endswith(".json")
        assert Path(analysis["figures"]).name == "figures"


class TestTheLayerItReads:
    def test_the_layer_name_is_the_one_the_chain_writes(self, tmp_path: Path):
        """Read back through the same constant the run writes, rather than the first layer of
        the file, so a GeoPackage that grows a second layer does not change the answer."""
        layer = detections(
            status=[DARK, MATCHED, MATCHED, MATCHED],
            scene=[f"s{n}" for n in range(4)],
        )
        path = tmp_path / "archive.gpkg"
        write_detections(layer, path)

        assert gpd.read_file(path, layer=DETECTIONS_LAYER).shape[0] == 4


class TestWhatCouldNotBeMeasured:
    """The distinction the module is built on, at the three places it was nearly lost.

    Each of these was a real defect in the first cut of `concentration.py`, found in review: the
    code was correct about the water and wrong about its own silence, which is the only kind of
    defect a stage whose output is prose can have.
    """

    def test_a_band_from_one_acquisition_says_so_rather_than_printing_nan(self):
        """Every band's detections come from a single scene, so no interval exists at all.

        Reverting this reports "every interval overlaps every other; no concentration
        established" over four `nan%` bounds — the module's own cardinal error, "we looked and
        found nothing" printed where the truth is "we could not look".
        """
        layer = detections(
            status=[DARK, MATCHED, MATCHED, MATCHED, DARK, MATCHED],
            scene=["s0", "s0", "s1", "s1", "s2", "s2"],
            shore=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )

        profile = profile_of(concentrate(layer, bands=3, draws=200, seed=0), "distance_to_shore_m")

        assert not profile.comparable
        assert not any(band.estimated for band in profile.bands)
        printed = profile.lines()
        assert not any("nan" in line for line in printed)
        assert any("no interval could be estimated" in line for line in printed)
        assert not any("no concentration established" in line for line in printed)

    def test_a_column_that_is_not_on_the_layer_is_reported_unavailable(self):
        """Not silently dropped from the report.

        A layer that never went through `darkvessel context` carries none of these columns.
        Filtering the missing ones out answers "where do they concentrate against fishing
        effort?" by not mentioning fishing effort, which reads as though it had been asked.
        """
        layer = detections(
            status=[DARK, MATCHED, MATCHED, MATCHED],
            scene=[f"s{n}" for n in range(4)],
        ).drop(columns="fishing_hours")

        result = concentrate(layer, draws=200, seed=0)

        profile = profile_of(result, "fishing_hours")
        assert not profile.available
        assert profile.measured == 0
        assert any("unavailable" in line for line in profile.lines())
        assert "fishing_hours" in {each["variable"] for each in result.as_dict()["profiles"]}

    def test_a_sea_that_never_moves_is_not_blamed_on_the_acquisition_count(self):
        """Twenty acquisitions is not "too few"; their sea level simply never varied.

        The correlation is missing either way, and the sentence explaining why has to be the
        true one — a reader told there were too few acquisitions would go and fetch more.
        """
        layer = detections(
            status=[DARK, MATCHED, MATCHED, MATCHED] * 5,
            scene=[f"s{n:02d}" for n in range(20)],
        )

        printed = concentrate(layer, draws=200, seed=0).sea.lines()

        assert any("does not vary" in line for line in printed)
        assert not any("too few" in line for line in printed)

    def test_two_acquisitions_are_still_too_few_to_rank(self):
        """The other branch of the same sentence, so that neither can be deleted unnoticed."""
        layer = detections(status=[DARK, MATCHED], scene=["s0", "s1"], sea_level=[-20.0, -14.0])

        printed = concentrate(layer, draws=200, seed=0).sea.lines()

        assert any("too few" in line for line in printed)


class TestTheMonteCarloErrorOfTheBounds:
    """The interval has an interval, and the page says so rather than printing past it.

    Added after review measured what the config had asserted: the first cut claimed the
    percentiles were "stable to well under the digit the README prints", and they are not — they
    move about half a point at the shipped draw count, which is the digit. The claim was the
    defect, not the draw count, and raising the draws is not the fix: the error falls only as the
    square root and still moves the printed digit at fifty thousand. So it is measured, reported
    in the committed run, and the page states the resolution to read its own bounds at.
    """

    @staticmethod
    def clustered(*, scenes: int = 40, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
        """Detections whose dark flag is a property of the acquisition, as the archive's are.

        Drawing each row's flag independently understates the spread badly — the real layer has
        29 of 49 acquisitions carrying no dark detection at all, and it is that clumping the
        resample has to move around. A fixture without it would make this test pass on a bootstrap
        that was not clustering at all.
        """
        rng = np.random.default_rng(seed)
        sizes = rng.integers(1, 9, size=scenes)
        per_scene = rng.random(scenes) < 0.35
        clusters = np.concatenate([[f"s{n:02d}"] * size for n, size in enumerate(sizes)])
        dark = np.concatenate(
            [np.full(size, flag) for size, flag in zip(sizes, per_scene, strict=True)]
        )
        return dark, clusters

    def test_the_bounds_move_when_only_the_seed_changes(self):
        """The measurement the README quotes. Zero here would mean it was never taken."""
        dark, clusters = self.clustered()

        low_spread, high_spread = monte_carlo_spread(dark, clusters, draws=4000, seeds=12)

        assert low_spread > 0.0 and high_spread > 0.0

    def test_more_draws_shrink_the_error_without_removing_it(self):
        """Holds the paragraph's claim that raising the draw count is not the fix.

        Twelve times the draws buys a visibly tighter figure and does not reach zero, which is
        why the page states a reading resolution instead of chasing one.
        """
        dark, clusters = self.clustered()

        cheap = max(monte_carlo_spread(dark, clusters, draws=2000, seeds=12))
        dear = max(monte_carlo_spread(dark, clusters, draws=24000, seeds=12))

        assert dear < cheap
        assert dear > 0.0

    def test_the_command_reports_the_error_beside_the_interval(self):
        """In the committed run and on the terminal, so the README quotes an artefact.

        `docs/runs/analysis-archive.json` is where the figure on the page comes from, the same
        way the ladder's verdict is read out of `docs/runs/` rather than retyped.
        """
        rng = np.random.default_rng(4)
        count = 120
        layer = detections(
            status=[DARK if rng.random() < 0.25 else MATCHED for _ in range(count)],
            scene=[f"s{n // 3:02d}" for n in range(count)],
        )

        result = concentrate(layer, draws=2000, seed=0)

        assert all(spread > 0.0 for spread in result.monte_carlo)
        written = result.as_dict()
        assert written["monte_carlo"] == list(result.monte_carlo)
        assert written["monte_carlo_seeds"] == SPREAD_SEEDS
        assert any("whole-percent resolution" in line for line in result.lines())

    def test_the_error_is_a_range_over_consecutive_seeds_and_does_not_wander(self):
        """A figure quoted on the page cannot itself move between two runs of the command."""
        dark, clusters = self.clustered()

        assert monte_carlo_spread(dark, clusters, draws=2000, seeds=8) == monte_carlo_spread(
            dark, clusters, draws=2000, seeds=8
        )

    def test_the_shipped_config_does_not_claim_a_stability_it_has_not_got(self):
        """The defect itself, held in the file it was written in.

        The comment beside `draws` asserted the percentiles were stable to well under the printed
        digit. They are not, and prose claiming a property the tests do not hold is the same
        defect as code that does not hold it.
        """
        shipped = (
            Path(__file__).resolve().parents[1] / "configs" / "kattegat-lane.yaml"
        ).read_text()

        assert "stable to well under the digit" not in shipped
        assert "Monte Carlo error" in shipped
