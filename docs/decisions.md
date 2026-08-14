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

---

## 2026-08-13 — A day of Danish AIS is streamed and filtered, never stored

**Decision.** `darkvessel ais` inflates the daily archive off the network a chunk at a time,
filters each chunk to the study area and the window, and discards it. Nothing is written to disk
except the few hundred reports that survive. `zipfile` is not used.

**Why.** The archive for the acquisition this project runs on is 662 MB compressed and 3.3 GB of
CSV. It holds 26 366 160 position reports for the day, of which 415 stand inside a 25 km box in
the half hour around the acquisition. `zipfile` reads the central directory at the end of a file and
therefore needs somewhere seekable, which means the whole 662 MB on disk before the first row can
be parsed. A daily archive holds exactly one member, and the local header at the front of the
stream says everything needed to inflate it, so the response is decompressed as it arrives.

This is the same constraint the Sentinel-1 export is built around, arriving from the other side
of the chain: the development machine has 8 GB and limited disk, and both halves of the fusion
level are two orders of magnitude larger than what they contribute to the answer. The whole day
crosses the network in under 40 seconds and never exists anywhere at once.

**What is not used, deliberately.** The uncompressed size in the local header. A zip written in
streaming mode records zero there and puts the true size in a descriptor after the data.
Inflation stops when the deflate stream says it has ended, which is true either way.

**The one check that is worth its cost.** The member's name has to name the day the URL asked
for. A server that answers every path with the same file, or a naming convention that changes
under us, otherwise produces an archive whose reports all fall outside the window — an empty
slice, which is a search that ran and found nothing, and every detection in the scene is then
honestly and wrongly dark.

**Where the archive lives.** `dma.dk` sends a reader to `aisdata.ais.dk`, which is a directory
listing in front of an S3 bucket; the files are fetched from the bucket endpoint itself,
`http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com`, which is the string `ARCHIVE_HOST` holds.
Not `web.ais.dk/aisdata`, which is what every published example points at and what the Danish
Maritime Authority's own older pages linked to: its certificate expired in June 2025 and the host
now resets the connection after the request. A URL that has moved once will move again, so it is
a named constant rather than a string in a function.

---

## 2026-08-13 — What cleaning removes from raw AIS, and why the two errors are not symmetric

**Decision.** Five rules, applied in that order, each counting what it removed: reports that are
not a vessel, reports without a nine-digit identifier, exact duplicates, two positions for one
vessel at one instant, and positions the rest of the vessel's own track cannot reach. The counts
are printed by the command and are part of what the slice claims.

**Why every rule is counted.** A slice is a claim about which vessels declared themselves, and it
is only as good as what was thrown away on the way to it. The first real run removed 237 of 415
reports — 57% — almost all of them exact duplicates. A cleaning step nobody can audit is a filter
that quietly decides what the answer is.

**The asymmetry that decides how wide each rule is.** A declaration wrongly removed is a
detection published as a dark vessel: a finding this project would be reporting, and the exact
fault the fusion level exists to remove. A declaration wrongly kept is a match that explains
nothing — quieter, and recoverable by anyone reading the row. So the filters are wide wherever
they can be, and removal is confined to reports that cannot be part of any track.

**Base stations and aids to navigation are not vessels.** The archive carries every transmitter in
Danish waters: of the first 1.18 million rows, 83 192 were base stations and 26 896 aids to
navigation. Both are real transmitters at real positions and neither is a ship, so neither can
turn a detection into a declared one — a buoy explaining a radar target would read in the output
exactly like a vessel that declared itself. Fixed structures standing in a radar scene are the
detector's problem, and this project already knows it has one: the wind farm is in the frame on
purpose.

**Two positions at one instant are a contradiction, not a duplicate.** An exact repeat is the
archive saying the same thing twice and one copy is kept. Two different positions for the same
vessel at the same second are a pair in which at most one is true, and nothing available here
says which; keeping either is a coin toss that the output then presents as an observation. Both
go, and the vessel keeps whatever else it reported, so it stays in the search.

**An outlier is judged against the median of its own track, not against the report before it.**
Two obvious rules fail, and both fail quietly. Walking a track forward and dropping whatever the
last kept report cannot reach anchors the whole track on its first position, so one bad report at
the start takes the vessel's whole slice with it. Judging each report against its immediate
neighbours cannot tell a spurious report from a good one whose only neighbour is spurious: three
reports with a jump in the middle are each unreachable from the one beside them, and the rule
removes all three — two of which were the evidence. That was caught by a test, after the
neighbour rule had been written and looked right. The median moves for no single report.

**The reach is per report, and the first version of it was dead code.** That version allowed
every report what a vessel at `ais.max_speed_kn` covers across the whole window — 55 km at the
shipped settings, against a searched area whose diagonal is 35 km. Nothing that survived the
spatial filter could exceed it, so the rule could not fire and its zero in the report was
guaranteed rather than observed. A count that is structurally zero is worse than no count: it
reads as a clean archive. The reach is now what the vessel could have covered between *that
report's* time and the middle of its own track, which for a spurious position among dense reports
is seconds, plus a floor at the order of the match tolerance — below that radius a displaced
report cannot change any verdict, so there is nothing to gain by removing it.

---

## 2026-08-13 — `dark` carries the number of declarations it was measured against

**Decision.** Every classified detection now carries `declarations_searched` alongside
`tolerance_m`, and the run says it out loud. A run that searched no declarations at all says so
in a third line of its verdict.

**Why.** The first real Danish slice this project ingested was empty — no vessel declared itself
inside that scene at that instant — and the chain did exactly what it was designed to do: an
empty slice is a search that ran and returned nothing, so all 115 detections came back honestly
dark. Opened in QGIS, that is a hundred and fifteen dark vessels over the Kattegat, which reads
as the headline finding of the project. Nothing in the layer distinguished it from one.

**Why this is not the `unsearched` distinction again.** `unsearched` says no declarations were
supplied. This says declarations were supplied and none of them was here, which is a different
claim and a true one — the reasoning that makes an empty slice honestly dark still holds. What
was missing was not correctness but legibility: a reader has the radius and now has what the
radius was applied to, and 115 dark against 0 declarations is unmistakable at a glance.

---

## 2026-08-14 — The study area is measured, and it moves onto the shipping lane

**Partly supersedes** *Study area: Danish waters*, in the part that chose Anholt for its wind
farm. Everything else in that entry stands, and the false-positive evidence the wind farm
produced is kept.

**Decision.** The study area moves from the Anholt box (11.15–11.40 E, 56.58–56.71 N) to
11.00–11.30 E, 57.55–57.70 N — 17.4 x 17.3 km of open water in the northern Kattegat, on the
approach to Skagen. The rectangle is chosen by `darkvessel survey`, which streams one day of
Danish AIS and ranks every rectangle of that size in the Kattegat.

**Why the old one had to go.** Anholt was chosen because turbines are bright point scatterers, so
the detector's false-positive problem would be visible in the very first real output. That
worked. It also put the box in quiet water off the Kattegat lane, and every acquisition over it
found the same thing: of 30 acquisitions between 21 June and 28 July 2026, 19 had no declared
vessel in the frame at all, and across the other 11 the largest vessel ever present was 15 m —
a pixel and a half at 10 m, and never a target this chain could match. The fusion level was
complete, tested and correct, and had nothing to fuse.

**What the ranking counts, and the three measures it had to beat.** Not reports: a ship alongside
a quay reports for twelve hours and one crossing at 14 knots is gone in twenty minutes, so report
counts measure dwell and transmit rate and a harbour wins. Not vessels over a day: that measures
throughput, and a rectangle one ship crosses at dawn scores like a lane, while an acquisition
arrives at a moment nobody chose. Not everything afloat: a rectangle chosen for how many 15 m
pleasure craft cross it is chosen on evidence the radar cannot see either way — which is exactly
how Anholt was chosen. What is counted is **distinct vessels of at least 100 m, under way,
standing inside the rectangle during a half hour** — the same half hour `darkvessel ais` fetches
— averaged over every half hour of the day including the empty ones.

The `under way` filter is load-bearing rather than tidy. Ranked on presence alone the winners are
the Frederikshavn approach and the Skagen anchorage, where twenty ships of 200 m sit waiting: all
large, all declared, all beside land, and no lane running through any of it.

**What the measurement said.** Over 2026-08-09, 29 718 190 position reports:

| Rectangle | Vessels ≥ 100 m under way, over the day | Mean in a half hour | Fewest | Empty half hours |
| --- | --- | --- | --- | --- |
| **11.00–11.30 E, 57.55–57.70 N** | **91** | **4.75** | **2** | **0** |
| 11.45–11.75 E, 57.25–57.40 N | 88 | 4.54 | 2 | 0 |
| 10.95–11.25 E, 57.55–57.70 N | 91 | 4.50 | 2 | 0 |
| 11.40–11.70 E, 57.25–57.40 N | 88 | 4.44 | 2 | 0 |
| 11.50–11.80 E, 57.25–57.40 N | 88 | 4.42 | 2 | 0 |
| 11.35–11.65 E, 57.25–57.40 N | 86 | 4.15 | 2 | 0 |
| 11.10–11.40 E, 57.50–57.65 N | 93 | 5.42 | 1 | 0 |
| 11.15–11.45 E, 57.50–57.65 N | 92 | 5.42 | 1 | 0 |
| 11.20–11.50 E, 57.50–57.65 N | 92 | 5.19 | 1 | 0 |
| 10.75–11.05 E, 57.65–57.80 N | 91 | 5.04 | 1 | 0 |

Ranked on the worst half hour before the mean: a rectangle that is empty at some point in the day
is a rectangle an acquisition can catch empty, and no average makes that acceptable. Three
rectangles score better on the mean and are passed over for it. Against Anholt's
largest-ever-15 m, this box holds five or six commercial ships at an arbitrary instant.

`darkvessel survey --config configs/survey.yaml` prints this table; the config carries the day,
the region, the box, the step, the window and both thresholds, so the ranking above is what that
one command reproduces rather than a number copied out of a notebook.

**What the choice gives up, in three parts.**

*The wind farm.* There are no turbines in the new box, so the detector's false-positive problem is
no longer standing in the frame of every run. It does not stop being a problem, and the evidence
already gathered over Anholt does not stop being evidence — but from here on the false positives
in a scene are sidelobes and sea clutter rather than a documented 111-turbine lattice.

*A polarisation.* Earth Engine answers a single download up to 48 MiB, and this rectangle at 10 m
came back from its grid at 57 MB in VV and VH. Something had to give. The area is the one thing
here that was measured and argued for; 20 m pixels would put a 100 m hull at five pixels and
halve the point of using SAR. So VV only, at 22 MB on disk. Nothing today reads VH — the chain
takes band 1 — but cross-polarised backscatter separates a hull from the sea better than VV does,
and a detector trained on both will need either a smaller rectangle or an export to Drive.

*Land, which is what was not given up.* The busiest cells in the Kattegat by any count are
coastal, and land is bright: a second false-positive problem on top of the one already documented.
Measured rather than eyeballed, because "there is no land in this box" is exactly the kind of
claim that is obvious and wrong. The top twenty rectangles the survey returned were reduced over
`NOAA/NGDC/ETOPO1` band `bedrock` in Earth Engine, at a scale of 1000 m, taking the mean and the
maximum elevation inside each — a one-arc-minute relief model is coarse against a 17 km box and
far finer than the question, which is whether any part of the coast is inside the rectangle at
all. Every one of the twenty is entirely at sea: the chosen box has a mean depth of 39 m and its
shallowest point is 31 m below the surface. The Øresund rectangles, which score comparably on
traffic, come back with a mean elevation *above* sea level and a 56 m hill inside them.

That check is a hand measurement against a live API and is not in the repository, for the reason
`gee_export.py` gives for the filters it does not test: what cannot be asserted offline is kept
small and written down rather than pretended at. What is written down is the dataset, the band,
the reducer and the scale, which is what someone needs to repeat it.

**Why one day, and why that is enough.** A lane is a feature of the traffic separation scheme, not
of the weather: the Kattegat routes carry the same ships in the same places every day. A day ranks
rectangles and is cheap enough that anyone can repeat the measurement. What a day cannot settle is
whether a particular acquisition catches anything, and that question has its own command.

---

## 2026-08-14 — The export ceiling is Earth Engine's own, and a sample costs nine bytes

**Supersedes** the size-guard reasoning in *A real scene arrives clipped, in one response, and its
georeferencing is never recomputed*. Everything else in that entry stands.

**Decision.** `MAX_REQUEST_BYTES` is 48 MiB, quoted from Earth Engine's own refusal.
`BYTES_PER_SAMPLE` is 9. The pixel count is estimated from all four corners of the area rather
than two.

**Why.** The ceiling was 64 MB, set on the argument that the real cap is a server-side detail that
would go stale in a comment, and calibrated against one area that came back fine. That argument is
sound and the number it produced was wrong in the only direction that matters. The first request
to approach it — this study area, in two polarisations — went out, waited, and came back with
`Total request size (57353670 bytes) must be less than or equal to 50331648 bytes`, which is the
entire failure the guard exists to prevent, performed by the guard.

**What the refusal was worth.** It states the limit exactly, so the ceiling is now a measurement.
It also states the size Earth Engine computed, and that number turned out to explain the second
error: the scene that eventually came back is 1845 x 1727 px, and 1845 x 1727 x 2 bands x 18 bytes
is 57 353 670 to the byte. Nine bytes per sample, not eight — float64, plus a byte of validity
mask alongside each. The estimate had been low by an eighth for every request this project has
ever made, and nothing had come close enough to the ceiling for it to show.

**The other eighth was the two corners.** A rectangle in degrees is not a rectangle in a projected
CRS: its edges bow, and the bounding box Earth Engine works to is wider than the box between two
opposite corners. Taking two corners understated the first study area by 6.5%. With four corners
and nine bytes, the estimate lands within 0.1% of the grid Earth Engine settled on for both areas
this project has exported, and a test pins it against all three observations — the area that came
back, the area that was refused, and the same area in one polarisation that came back.

**Why not a safety margin instead.** A fudge factor was the first fix written here, and it would
have held. It would also have left two wrong models in the code with a constant on top hiding
both, and the next area that failed would have failed for a third reason nobody could separate
from the first two. The margin came out once the arithmetic explained the measurement.

---

## 2026-08-14 — The detector trains on LS-SSDD-v1.0

**Decision.** Labels come from LS-SSDD-v1.0: 15 large Sentinel-1 IW acquisitions, VV, cut by the
dataset's authors into 9000 sub-images of 800 x 800, with ships labelled by SAR experts against
AIS and Google Earth.

**Why.** Its physics is this chain's physics. Same satellite, same 10 m pixel spacing, same
problem of a hull three or four pixels across against an enormous empty sea — and the same
labelling method as the thing being built here, AIS read against imagery. A detector trained on
it is learning what a ship looks like on the product this pipeline actually exports.

**Rejected.** HRSID, 5604 sub-images at 0.5–3 m: a 100 m ship there is a hundred pixels with a
visible superstructure, and the features that separate it from the sea are not the features
available at 10 m. SSDD, the classic small set, mixes Sentinel-1, RadarSat-2 and TerraSAR-X at
resolutions from 1 to 15 m, so a model fitted to it is fitted to an average of sensors that this
project only ever sees one of. xView3-SAR is the closest match of all — Sentinel-1 with
AIS-derived labels, and the public challenge this problem is known by — and it is terabytes.
Scoping a subset of it is a piece of work in its own right; it stays in reserve for the level
where the detector is the bottleneck rather than the chain.

**What it costs.** LS-SSDD ships VH only as the 15 large scenes, not as sub-images, so training
is VV alone. That happens to match what the chain exports — the Kattegat box comes back VV-only
against Earth Engine's 48 MiB ceiling — so nothing is lost today. It is also the reason both will
have to move together on the day cross-polarised backscatter is wanted.

---

## 2026-08-14 — The held-out split is drawn by scene, and only the training side is ever cut down

**Decision.** LS-SSDD's own split: sub-images cut from scenes 01–10 train, 11–15 are held out.
The training side keeps every tile carrying a ship plus one empty tile per ship-bearing tile; the
held-out side is scored entire, pure backgrounds and all.

**Why by scene.** Two 800 px cuts of one acquisition are not two independent samples. They share
a sea state, an incidence angle, a calibration and a speckle distribution, and a ship on the seam
between them is in both. A split drawn over sub-images would measure how well the model
recognises scenes it has already seen — a number that goes up and means nothing. Using the
dataset's own split rather than a fresh one also keeps the results comparable to published
baselines instead of only to themselves.

**Why the training side is cut down.** LS-SSDD is mostly open water by design, and on a free tier
the binding constraint is GPU hours. Dropping most of the empty tiles buys epochs. It is bounded
rather than total because a detector never shown open water will find ships in it, and the ratio
is written in the config so a run states how much sea it trained against.

**Why the held-out side is not.** The empty tiles are exactly where a false positive happens.
Scoring only over tiles known to contain a ship would report a precision the detector has not
earned, and it would be the easiest number in this project to publish by accident.

---

## 2026-08-14 — A detection is scored against the tolerance the fusion will apply to it

**Decision.** A detection counts as finding a ship when it lands within 200 m of that ship's
labelled centre — the fusion's own match tolerance, read into pixels through the resolution.
Not by intersection-over-union between boxes. Detections are matched in order of confidence and
claim a ship exclusively, and precision and recall are reported at a table of score thresholds
rather than at one.

**Why not overlap.** A 60 m vessel is six pixels at 10 m. A box two pixels out — a fifth of a
hull — already fails at half overlap, so an IoU score at this resolution mostly measures box
regression, and no part of this chain uses the box. What it uses is the point: a detection
becomes a ground coordinate and is compared against a declared position within a tolerance in
metres. Scoring the detector by the rule that will later be applied to it measures the thing that
matters downstream rather than a proxy for it.

**What it costs, stated rather than hidden.** 200 m is 20 pixels, which is generous against
labels that are pixel-accurate: a detection can be four hull-lengths off and still count. That is
the right rule for *this* chain and the wrong rule for comparing against a detection benchmark,
and the tolerance is in the config in metres so that anyone reading a precision can see how far a
hit was allowed to be.

**Why a table of thresholds.** A detector does not have a precision. It has a precision at a
confidence, and choosing the confidence is a decision about how much an inspection costs — made
later, by someone else, against a budget this repository knows nothing about.

---

## 2026-08-14 — What survives a session that is killed

**Decision.** A checkpoint is written under a temporary name and moved into place in one step.
The weights are written before the held-out split is scored, not after. The last two epochs are
kept and the rest deleted. Metrics go to a plain JSON file beside the weights. Mixed precision is
not used.

**Why the temporary name.** Free-tier sessions end when the provider says so, and a third of a
gigabyte of weights takes a while to reach the disk. A kernel stopped in the middle of that leaves
a truncated file under exactly the name the next session resumes from — so the run continues from
a state that was never valid, and nothing anywhere says so. `os.replace` makes a checkpoint either
whole or absent.

**Why weights before metrics.** An interrupted evaluation costs numbers that can be recomputed
from the checkpoint. An interrupted checkpoint costs an epoch that cannot.

**Why only two.** Kaggle gives 20 GB of working space against checkpoints of a third of a
gigabyte. Resuming needs the last one; the spare is cheap insurance. Choosing the *best* epoch
rather than the last is a different job, done later against the metrics file.

**Why metrics in JSON rather than inside the checkpoint.** The precision and recall are the
output of this level. Reading them should not require torch, a GPU or an unpickle.

**Why no mixed precision.** It is the obvious way to buy epochs on a T4, and it adds a loss scaler
whose state has to be saved and restored correctly — on a development machine that cannot run the
code path that uses it. An untested resume is a worse trade than a slower epoch. It belongs with
the rest of the small-target work, where there will be a GPU to test it on.

---

## 2026-08-14 — The detector is trained on 8-bit amplitude and the chain feeds it decibels

**Decision.** Recorded now, resolved at the level that swaps the trained detector into the chain.

**What the gap is.** LS-SSDD ships its sub-images as 8-bit JPEG, so the reader takes them as
amplitude in 0..1 and refuses anything else. What the chain exports is Sentinel-1 GRD from Earth
Engine in decibels, where the sea sits near −14 and the shipped run thresholds at 0. These are not
the same quantity, and the stretch that turned one into the other is not recorded in the dataset
and cannot be recovered from it.

**Why refuse rather than cast.** Handing a dB scene to a model fitted on 0..1 amplitude does not
crash and does not warn. It produces detections — plausible ones, in plausible places, with
scores — and the first sign of trouble would be a precision that made no sense three levels
later. The reader names the dtype it was given and says why it will not take it.

**What has to happen.** A documented mapping from calibrated dB to the range the model was fitted
on, chosen deliberately and tested on a scene where the answer is known by eye. That is the swap
ticket's work, and it is written down here so that it is a task rather than a discovery.

---

## 2026-08-14 — The first training run keeps the stock anchors, and that is the baseline

**Decision.** The shipped training config uses torchvision's own anchor sizes — 32 px upwards —
even though they are the wrong range for this data. Adapting them is the next detector ticket's
work, and `anchor_sizes` is a config key so that the adaptation is one line and a second run.

**Why not just fix them now.** They are wrong in a way that is easy to state: the smallest anchor
is 32 px, which at 10 m is a vessel 320 m long, longer than nearly everything in the training set.
The temptation is therefore to ship the fix with the first run. But the ticket that owns the
adaptation asks for each change to be *measured against the configuration before it, on the same
held-out split* — and shipping the fix unmeasured deletes the configuration before it. There would
be nothing left to compare against except a number nobody has.

**What it costs.** The first run's recall will be poor, possibly very poor, and the evening it
takes will buy a baseline rather than a detector. That is the arbitration rule this project
already wrote down: cut model performance, never chain completeness. A baseline that makes the
next change measurable is worth more than a better first number that makes it unmeasurable.

**The same argument does not apply to the channel count.** Repeating single-polarisation
amplitude across three channels is not an adaptation, it is the minimum required for a
three-channel backbone to accept the data at all. What the ticket means by an input stage adapted
to radar polarisation — a dual-polarisation stem trained as one — is still to come, and is not
what is here.

---

## 2026-08-14 — Where the annotations start counting is measured, not assumed

**Decision.** Whether the VOC boxes count their pixels from zero or from one is determined from
the boxes themselves at load time. `data.first_index` in the config overrides it and exists only
for a subset too small to settle the question.

**Why it matters more than it looks.** The two readings differ by one pixel. On a ship four
pixels across that is a quarter of the target, applied to every ship in the set, in the same
direction, and it is invisible: the boxes still land on ships, the loss still falls, and the
detector simply learns the wrong size of the only thing it is looking for.

**Why measured rather than assumed.** PASCAL VOC as originally published counts from one; sets
written with later tools frequently count from zero; LS-SSDD says neither. The first version of
this code assumed zero and checked the assumption by refusing any index that reached the image
size. That is a real check, but it fails in the wrong direction — on a 1-based set it stops the
run dead with no way past, and on a subset where no box happens to touch an edge it passes while
being wrong.

**What the evidence is.** An index of 0 cannot occur in a set counting from one, and an index
equal to the width cannot occur in a set counting from zero. Either one settles it, and over
9000 tiles cut from whole scenes there are always many. Both present at once means the
annotations are not all in one frame, which is refused. Neither present means the question cannot
be answered from the data, which is also refused — naming the setting that answers it, rather
than defaulting to the reading that happens to be more common.
