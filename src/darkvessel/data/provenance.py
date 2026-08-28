"""Which acquisition a detection came out of, and what the sea was doing in it.

One run over one scene never needed either. There was one acquisition, its name was on the
command line, and its sea state was the sea state of the only answer. An archive-wide run
accumulates fifty acquisitions into one layer, and both facts stop being properties of the run
and become properties of the row.

**The scene name is provenance.** `embed/archive.py` makes the same argument about crops: a
neighbour that cannot be pointed at on a map or in an acquisition is not evidence of anything.
A detection in a merged layer with no acquisition on it cannot be checked, cannot be opened
again, and cannot be excluded when the scene it came from turns out to have a problem.

**The sea state is a confound, recorded so it can be tested rather than assumed away.** The
chain applies one fixed window between decibels and amplitude, calibrated on the sea of a single
scene, and `amplitude.fit_window` states why that must not be refitted per scene: refitted, a
score threshold stops meaning the same thing from one acquisition to the next. What that
argument buys is comparability, and what it costs is that a scene whose sea sits well away from
the calibrated one is scored at an operating point nobody chose.

Across the Kattegat archive the measured sea spans tens of decibels while the window is anchored
at one value, so the count of dark detections per scene could be partly an artefact of wind
rather than of undeclared traffic. That is a real possibility and the wrong answer is to correct
for it silently. Recorded on the row, it is a variable the spatial analysis can regress its
distributions against and report on; absent, it is a confound nobody can see.

The estimator is `amplitude.sea_level`, deliberately, rather than a mean written here. It is the
project's one statement of where a scene's sea stands — robust, because a scene contains ships
and a ship stands forty decibels above the water — and it is the same estimator the window was
fitted with. Two ways of measuring the sea would eventually disagree, and the disagreement would
look like weather.
"""

import geopandas as gpd

from darkvessel.data.scene import Scene
from darkvessel.detect.amplitude import sea_level

# Which acquisition the detection came out of. The product's file stem, which is what a reader
# needs to open the scene again.
SCENE = "scene"
# Where that scene's sea stood, and how much it varied, by the estimator the stretch was fitted
# with. In the units of the scene: decibels for a Sentinel-1 product, and the 0..1 amplitude the
# synthetic fixture is written in — which is why these are read alongside `scene` rather than
# compared across products.
SEA_LEVEL = "sea_level_db"
SEA_SPREAD = "sea_spread_db"

PROVENANCE = (SCENE, SEA_LEVEL, SEA_SPREAD)


def attach_provenance(detections: gpd.GeoDataFrame, scene: Scene) -> gpd.GeoDataFrame:
    """Put the acquisition and its sea state on every detection of that acquisition.

    Constant down the column, because they are properties of the scene rather than of the
    detection — the same shape `classify` writes `declarations_searched` in, and for the same
    reason: a merged layer is read one row at a time, and a fact that has to be looked up
    elsewhere to interpret a row is a fact that will not be.

    The sea is measured once per scene here rather than per detection. It is a reduction over the
    whole image, so measuring it per row would be the same number computed as many times as there
    are ships in it.

    A scene with no name writes an empty one rather than being refused. A scene built in memory
    is what every test and the synthetic fixture use, and a column that only a run from disk
    could fill would make the schema depend on where the pixels came from — which is the promise
    `without_context` and `without_a_register` both make in the other direction.
    """
    level, spread = sea_level(scene.image)

    carried = detections.copy()
    carried[SCENE] = scene.name if scene.name is not None else ""
    carried[SEA_LEVEL] = level
    carried[SEA_SPREAD] = spread
    return carried
