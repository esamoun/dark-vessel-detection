# Decision log

Why each choice was made, dated, with the reasoning that produced it. Append; do not rewrite
history. When a decision turns out to be wrong, add a new entry that supersedes the old one
rather than editing it.

---

## 2026-08-12 — Study area: Danish waters

**Decision.** Danish waters, using open AIS archives from the Danish Maritime Authority.

**Why.** Three reasons converge. Sentinel-1 revisit over Europe is excellent because Copernicus
treats it as a priority observation zone, so acquisitions are frequent and regular — US coastal
revisit is markedly more irregular since the loss of Sentinel-1B. The Danish AIS archive is raw,
daily and needs no registration, which means the whole ingestion chain is mine rather than
inherited from a preprocessed product. And the traffic is dense and varied: cargo, coastal
fishing, leisure, plus enough offshore wind to guarantee a real false-positive problem.

**Rejected.** US Marine Cadastre (weaker SAR revisit), Global Fishing Watch API (activity data
already processed — less of the chain is mine), Norwegian waters (viable alternative, kept in
reserve).

---

## 2026-08-12 — Training runs on cloud free tiers, not locally

**Decision.** Kaggle for long training runs, Colab for exploration and demonstration. Nothing is
trained locally.

**Why.** The development machine is an 8 GB M1 MacBook Air. A single Sentinel-1 GRD product is
larger than the disk headroom that existed before this project started, and 8 GB of unified
memory shared with the OS cannot train a detector. Kaggle provides guaranteed weekly GPU hours
against Colab's best-effort allocation, and hosts the labelled SAR datasets directly, so training
data never transits the local disk.

**Consequences, accepted deliberately.** Training subset is scoped and documented rather than
exhaustive; tiles are small; sessions are short and resumable; checkpointing is written from the
first epoch rather than added after the first lost run. Full-scene demonstration covers one or two
Sentinel-1 scenes, not a region.

---

## 2026-08-12 — Build in levels, publish at each one

**Decision.** Detector → full-scene chain → AIS fusion → spatial analysis, in that order, with
each level published before the next is started.

**Why.** The failure mode for a project like this is not abandonment, it is accumulation: scope
grows quietly and nothing is finished at any checkpoint. Publishing each level means there is
always something complete to show, whatever date someone looks.

**Arbitration rule.** If time runs short, cut model performance — never chain completeness. A
mediocre detector inside a complete, honest pipeline is worth more than an excellent detector
that stops at test-set metrics.

---

## 2026-08-12 — AlphaEarth / Satellite Embedding rejected for this project

**Decision.** Google's Satellite Embedding dataset is not used here. Embeddings are learned from
detection crops instead, by self-supervised contrastive training.

**Why.** The Satellite Embedding product is an *annual* composite. A vessel occupies a location
for minutes; an annual composite averages transient objects away by construction. Using it to
find ships would misrepresent what the product is, and the target audience knows the product well
enough to notice.

**Where the embedding idea does apply.** Over detection crops, where it separates vessels from
fixed offshore structures without additional labelling, supports similarity search across the
archive, and flags anomalies. Kept in reserve for a future land-based project, where an annual
embedding is exactly the right tool.

---

## 2026-08-13 — The chain is built end to end before anything in it is good

**Decision.** The first working thing is a walking skeleton: a synthetic scene, a threshold on
bright pixels standing in for the detector, matching against the nearest AIS report in time, and
a GeoPackage out. Every stage is present and none of them is yet good.

**Why.** The three failure modes that matter here are silent: a georeferencing fault puts
plausible detections in the wrong place, a tiling fault double-counts, a matching fault invents
dark vessels. None of them announces itself, and all of them are cheap to catch while the chain
is still trivial and impossible to catch once a model's errors are mixed in. Building the chain
first also means the seam is fixed by something that runs, rather than designed in the abstract.

**Consequence.** The detector is a parameter of the pipeline, not an import inside it, so the
whole chain runs with no weights, no GPU and no network. That is what makes the seam test
possible, and it is the load-bearing design decision of the project.

---

## 2026-08-13 — Matching is against the nearest report in time, and that is wrong on purpose

**Decision.** The skeleton matches each detection to the AIS report nearest in time to the
acquisition, taken as it stands. No interpolation along the track.

**Why.** Interpolation is Level 3, and putting it in now would mean building the interesting
part before the wiring around it was proven. The naive version establishes the seam that the
real one drops into.

**The risk, stated rather than discovered later.** A vessel at 12 knots moves some 370 m in a
minute. Against a report a few minutes old, a declared vessel can fall outside any sane
tolerance and be reported dark. Every dark result from this level of the chain is an artefact of
that, not evidence, and the naive matching is recorded in the config as
`interpolate_ais_to_acquisition: false` so a run cannot silently claim otherwise.

---

## 2026-08-13 — Match tolerance provisionally 200 m

**Decision.** `match_tolerance_m: 200` in the pipeline config, and the tolerance is written into
every output row rather than left in the config file.

**Why 200.** It is a placeholder of the right order — Sentinel-1 GRD geolocation error is metres
to tens of metres and AIS position error is small, so the tolerance is dominated by how far a
vessel travels between its last report and the acquisition. The number that belongs here comes
from that analysis and cannot be derived until AIS interpolation exists. It is provisional and
labelled as such.

**Why it travels with the results.** "Dark" is a claim about what was searched. A detection
marked dark at 200 m and one marked dark at 2 km are different claims, and a reader who has only
the layer cannot tell them apart unless the radius is in the row.

---

## 2026-08-13 — torch and Earth Engine are extras, not dependencies

**Decision.** The package's required dependencies are the chain's: numpy, rasterio, geopandas
and friends. `torch`/`torchvision` move to a `detector` extra and `earthengine-api` to a `gee`
extra.

**Why.** The acceptance condition for the chain is that it runs with no weights, GPU or network.
A hard dependency on torch contradicts that at install time: a reader cloning the repository to
see the pipeline work would pull two gigabytes of CUDA wheels to run a threshold on bright
pixels. It also matters locally — the development machine has 8 GB and limited disk.

---

## 2026-08-13 — A scene outside the working CRS is refused, not reprojected

**Decision.** The run declares its working CRS in the config. If the scene is not in it, the
command fails with an error naming both CRSs.

**Why not just use the scene's CRS.** The match tolerance is a distance in metres, compared
against coordinate distances. Given a scene in degrees, 200 becomes 200 degrees and every
detection is matched — or, with a small tolerance, every detection goes dark. Nothing crashes
and the output looks entirely plausible. This is the same class of fault as a georeferencing
error, and it is caught the same way: loudly, at the boundary.

**Why not reproject silently.** Reprojecting radar amplitude resamples it, which changes what
the detector sees. That is a decision about the run, and it belongs to whoever configured it.

---

## 2026-08-13 — CI installs with pip, and runs the same two make targets as a laptop

**Decision.** GitHub Actions runs `make lint` and `make test` on every push and pull request, as
two separate jobs, installing with `pip install -e ".[dev]"` rather than building the conda
environment in `environment.yml`.

**Why the same make targets.** If CI runs a different command from the one in the README, there
are two definitions of "the tests pass" and they drift apart quietly. The Makefile is the single
definition; CI is one more caller of it.

**Why pip and not conda.** `environment.yml` exists for a specific local problem — GDAL and the
geospatial stack are painful to build on macOS. On Linux the same libraries are wheels and
install in seconds, so conda would buy minutes of environment solving and nothing else. It also
makes CI verify the claim the README actually makes to a reader: that a clean machine with pip
gets a working chain. The chain runs with no weights, no GPU and no network, which is what makes
this possible at all — a run needing Earth Engine credentials or a checkpoint could not be a
required check on a public repository.

**Why lint and tests are separate jobs.** In one job the lint step runs first and a formatting
slip stops the tests from running at all, so a red badge says nothing about whether the code
works. Separately, the checks list names which of the two broke.

**Why the tests run on two Python versions.** `requires-python` claims 3.11 and `environment.yml`
pins it, but the development machine runs 3.13. Testing only one of them leaves the other an
untested claim.

**What that costs, recorded now rather than discovered later.** rasterio and numpy have both
moved to `>= 3.12`, so the 3.11 leg does not install the same dependencies as the 3.13 one — it
backsolves to rasterio 1.4.4 where 3.13 gets 1.5.1. It resolves today. When it stops resolving,
the leg will go red for a reason that has nothing to do with the code, and the answer at that
point is to raise the floor to 3.12 in `pyproject.toml` and `environment.yml` — not to widen the
matrix or pin dependencies to hold 3.11 open. Lint runs on 3.11 only; ruff's result does not
depend on the interpreter.

---

## 2026-08-13 — Cross-tile duplicates are prevented by ownership, not removed by proximity

**Decision.** The scene is partitioned into cores, one per tile, and a tile reports only the
detections standing in its own core. There is no merge step, no distance threshold and no
non-maximum suppression over the assembled detections.

**Why not merge by proximity.** The obvious scheme is to run every tile, pool the detections and
collapse any two that land within a few pixels of each other. It needs a radius, and there is no
radius that is right: too small and a target seen slightly differently by two tiles survives
twice; too large and two vessels moored side by side become one. The chain already asserts, in
`match.py`, that two hulls 60 m apart are two vessels — a merge radius wide enough to be safe
against clipped centroids would quietly contradict that. Worse, both failures are silent. A
merged pair and a duplicated target both produce a count that looks entirely plausible.

Ownership has no such parameter. Every position in the scene lies in exactly one core, so the
count is right by construction rather than by tuning, and the property is testable as a property:
walk every position of every tile and assert each is claimed exactly once.

**What it rests on, stated because it is a real constraint on the config.** The overlap must be
at least as wide as the largest target the detector will report. A core stops half an overlap
short of its tile's edge, so a target centred in a core is at least half an overlap from that
edge and is seen whole by the tile that owns it. A neighbouring tile may see the same target cut
in half and report a centroid displaced towards its own interior — that view is discarded,
because a clipped centroid cannot move far enough to land in the neighbour's core. At 10 m pixels
a vessel is a handful of pixels across and any sane overlap satisfies this; a very long ship on a
scene tiled tightly would not, and that is a config error rather than a code path.

**The other half of the rule: the last tile is pulled back against the edge of the scene.** A
scene is not a whole number of strides. Rather than a runt tile at the far edge, or a tile hanging
over it, the last tile starts at `extent - size` and therefore overlaps its predecessor by more
than the configured amount. Every tile is then the same size — the shape a detector is trained on
— at the cost of reading a strip twice, which ownership makes harmless.

**Consequence, taken deliberately.** Detections are returned in scene row-major order rather than
in the order the tiles happened to produce them. Tiling is a property of the hardware the
detector runs on; two runs of the same scene at different tile sizes return the same answer, and
a test asserts it.

**The shipped config drops from 512/64 to 144/32, and that is not a claim about Sentinel-1.** A
tile of about 512 px is the figure that follows from a detector's memory, and it is what a real
scene will be run at. The scene `configs/pipeline.yaml` actually points at is the 256 px synthetic
one, which a 512 px tile swallows whole — so the shipped command would demonstrate tiling by never
tiling. 144/32 cuts that scene into four tiles meeting where the fixture stands a target. The
number follows the scene in the config, and when the config points at a Sentinel-1 scene it will
follow that instead.

**Why a test reads that config file.** It is the one config the suite does not write for itself,
which makes it the one place a value can be widened back without a test noticing — the same
class of gap as the `.gitignore` in docs/failures.md, where every check ran against something
other than what was shipped. `test_the_shipped_config_still_cuts_the_synthetic_scene_across_a_target`
closes it: it asserts that, at the tiling that file declares, more than one tile sees the
fixture's boundary target.

---

## 2026-08-13 — A real scene arrives clipped, in one response, and its georeferencing is never recomputed

**Decision.** `darkvessel export` asks Earth Engine for one acquisition, already clipped to the
area of interest and reprojected into the working CRS, and takes back a single GeoTIFF. The file
is opened for update only to add the metadata the pixels do not carry — acquisition time, scene
id, polarisations, orbit pass. The transform and CRS Earth Engine wrote are left untouched.

**Why not export to Drive.** A batch export has no size limit and would take a whole swath, but
it splits a run in two: launch a task, wait, download, then run the chain. The direct download
answers in one call, which keeps "a run is one command against one config file" true, and it is
bounded — which is the point. A Sentinel-1 GRD product is two orders of magnitude past what one
response carries, so no full product can arrive by this path even by mistake. The area shipped in
`configs/anholt.yaml` is about 15 km square and came back as 1582 x 1498 px in VV and VH, 33 MB,
which at 512/64 is sixteen tiles with real seams between them rather than a single tile
pretending. Drive
remains the escape hatch for a whole swath, and is not built until something needs one.

**Why the transform is taken as it stands.** Rebuilding one from the bounding box and the pixel
size looks entirely reasonable and is the same class of fault as the tiling and CRS ones: it
never crashes, it just puts every detection somewhere else. The one thing this module must not
have is an opinion about where the pixels are.

**Why the request is refused before it is sent.** An area past the limit is refused locally, with
the arithmetic shown, rather than by Earth Engine after the wait — and its message would not say
which of the three numbers to change. A test reads `configs/anholt.yaml` and runs it through the
command's own parsing, so a mistyped key in the one config that needs credentials is caught in a
second rather than by someone who has already authenticated.

**What is not tested, stated rather than implied.** Whether Earth Engine's filters select what
this code believes they select is a claim about a live API and cannot be made here. The client is
kept to filtering, reading metadata and fetching bytes — it decides nothing — so that what is
untestable is also as small as possible. It is verified once, by hand, on the first real export.

---

## 2026-08-13 — A run with nothing to match against reports `unsearched`, not `dark`

**Decision.** `ais: null` is a valid run. Its detections come back with status `unsearched` and
no tolerance, rather than `dark` at the configured radius.

**Why.** The first real scene runs before real AIS exists — ingesting Danish archives is the
next level. `classify` previously marked every detection dark when handed no declarations, which
is the most confident wrong answer this chain is capable of producing: a GeoPackage of a thousand
"dark vessels" over the Kattegat, opening in QGIS looking exactly like a finding. "Dark" is a
claim about what was searched, and with no AIS supplied nothing was.

**The distinction that matters.** An empty AIS slice is not the same as no AIS slice. An empty
slice is a search that ran and returned nothing, and its detections are honestly dark. `None` is
no search at all. The config spells the absence out as `ais: null` rather than allowing it by
omission, so a run cannot arrive here by forgetting a key.

---

## 2026-08-13 — The synthetic scene is placed at sea, not at a round number

**Decision.** The synthetic fixture moves from (500000, 6150000) to (639000, 6282000) in
EPSG:25832 — from farmland near Vejen to open water in the Kattegat, inside the area
`configs/anholt.yaml` fetches a real acquisition over.

**What was wrong with the old origin.** Nothing, arithmetically. 500000 is the central meridian
of UTM zone 32N and 6150000 is a round northing; both were placeholders, and every detection
landed exactly where the transform said. But the transform said mainland Jutland. Dragged onto a
basemap in QGIS — the first thing anyone does with the output, and an acceptance criterion of the
real-scene ticket — the demonstration showed four vessels in a field.

**Why it matters more than it looks.** The claim this repository makes is that the output opens
in QGIS and lands where it should. A reader checking that claim against the shipped demo would
have found it false, and would have had no way to tell a placeholder origin from a georeferencing
fault — which is precisely the failure the georeferencing tests exist to rule out. A fixture that
cannot be distinguished from the bug it is meant to disprove is worth moving.

**How it was moved.** By a uniform shift, applied to the fixture and to every hand-derived
literal in the seam test at once. The tests assert ground coordinates worked out from the
transform rather than recomputed by the code, so an inconsistent shift fails them; passing after
the move is what says the shift was uniform.

---

## 2026-08-13 — A product's nodata is a hole, and a hole is not a target

**Decision.** `Scene.from_geotiff` turns anything the file declares as nodata into NaN before the
image reaches a detector.

**What happened.** The first real Sentinel-1 scene run through the chain returned 126 detections.
Twelve of them were not vessels. Earth Engine writes masked pixels as a fill value and declares
that value as nodata; the export took 0 for the fill, and 6.2% of the scene was fill. Read
plainly, 0 is just a number — and on a scene in dB, where the sea sits near -14 dB, it is
brighter than any vessel in the image. The threshold detector duly found three "targets" of
72100, 38955 and 36428 pixels.

**Why this is the interesting kind of bug.** Nothing crashed and nothing warned. The count was
plausible, the detections carried scores and coordinates like any other, and the largest of them
would have looked, in QGIS, like an unusually large vessel rather than a hole in the product. It
is the same family as a georeferencing fault or a double-counted target: an answer that is wrong
without being suspicious. It also could not have been found on the synthetic scene, which has no
holes because it was written by us — the first real product was always going to be the thing that
surfaced it.

**Why NaN rather than a mask.** Every comparison against NaN is false, so a hole cannot exceed
any threshold a detector picks — including a detector written later that never considered nodata
at all. A masked array would work today and would depend on each future detector remembering to
honour it.

**What it also confirmed.** Those three blobs were far wider than the 64 px overlap, and came back
duplicated across tiles — exactly as the ownership scheme's stated precondition says they must.
The scheme held; the input broke the condition it is documented to require.

---

## 2026-08-13 — Vessels are placed at the acquisition instant, and never extrapolated past their track

**Supersedes** *Matching is against the nearest report in time, and that is wrong on purpose*.

**Decision.** Before anything is compared, each vessel is placed at the acquisition timestamp by
linear interpolation between the two AIS reports either side of it. Where the acquisition is not
bracketed — the track ends before the radar looks, or begins after it, or the vessel reported
once — the nearest report is used as it stands and the row says so. Nothing is extrapolated.
`fusion.interpolate_ais_to_acquisition` is gone from the configs.

**Why the config key went.** It recorded, in a file, a claim about the whole run. Every match now
carries `position_basis` — `interpolated` or `reported` — and `position_age_s`, so the claim is
made per vessel, in the layer someone opens, rather than globally in a file they may not read. A
per-row statement is strictly stronger, and keeping the key would have meant maintaining a code
path whose only purpose is to give the answer this entry supersedes.

**Why not extrapolate.** Prolonging a track past its last observation needs a course and a speed,
and the archive's position reports carry neither, so both would be derived from two earlier points
and projected forward. That manufactures a position where no measurement exists. It is the same
class of confidently wrong answer as matching against a stale report, and harder to see, because
the output looks like a placement rather than a fallback. A vessel that cannot be placed keeps its
nearest report and is marked; it stays in the search, because a vessel that declared itself and
cannot be placed is still a vessel that declared itself, and dropping it would publish it as dark
— the exact fault this level exists to remove.

**The gap ceiling, provisionally 600 s.** A straight line between two reports is a claim that the
vessel held one course and one speed between them, and the wider the bracket the more confident
the claim looks while being worth less. `fusion.interpolation_max_gap_s` is the widest bracket a
line may span; past it the nearest report is used instead. 600 s is an upper bound of the right
order rather than a derived figure — a class A vessel underway reports every few seconds and one
at anchor every three minutes, so a bracket wider than ten minutes is a hole in the archive rather
than a reporting cadence. The number an error analysis would give is smaller: a vessel altering
course inside a ten-minute gap leaves the chord by far more than the 200 m tolerance. Measuring
chord error against real Danish tracks is what settles it, and that arrives with the level that
ingests them. Provisional, and labelled as such, like the tolerance it sits beside.

**What this does not fix.** The tolerance is still 200 m and still provisional. Interpolation was
the prerequisite for deriving it: the tolerance was dominated by how far a vessel travels between
its last report and the acquisition, and that distance is no longer part of the error budget for
an interpolated position — what remains is how far the vessel departed from the straight line
between its two reports. `position_age_s` is what says how much room there was to depart: it is
the gap to the nearest of the two bracketing reports, not zero. Deriving a tolerance from it needs
real tracks, not a synthetic fixture.

**Why the gap ceiling is not written into the output rows, when the tolerance is.** They look
alike and are not. Without the tolerance, `dark` cannot be read at all — the radius *is* the
claim. `reported` can be read without the ceiling: the row says the position is an observation
from another moment and `position_age_s` says how far away that moment was, which is what decides
whether to trust the match. The ceiling is a threshold the run was configured with, like the tile
size, and it belongs with the config rather than in every row.

**Reports that cannot be placed at all are refused, not skipped.** Grouping by MMSI drops a report
that has none without a word, and a missing timestamp compares false against the acquisition in
both directions and falls out of the bracket just as quietly. Either way a declaration disappears
on its way to the matching, and a declaration that disappears is a detection published as dark —
this level's own fault, reintroduced by the mechanism meant to remove it. Raw Danish archives do
contain such rows, so this will fire on the first real slice. That is the intended place for it to
fire: cleaning raw AIS is the ingestion level's decision, and the alternative — pooling
unidentified reports under one key — would draw a track between two different ships.

**What the fixture gained.** A vessel under way, reporting 900 m west of its target three minutes
before the acquisition and 600 m east of it two minutes after. Neither report is within any sane
tolerance of where the radar imaged it; the interpolated position lands on it. Every vessel in the
fixture had until now been standing still, and a fixture whose vessels do not move cannot tell a
chain that interpolates from one that does not.
