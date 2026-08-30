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

**What was done instead.** Corrected on 2026-08-17 — see docs/failures.md. Originally recorded as:
still to come. It is not: LS-SSDD is VV only and the scene this chain exports is VV only, so a
dual-polarisation stem has no data to be fitted on and none to be run on, on either side. What was
built instead is the single-channel stem, folded down from the pretrained three-channel kernels
and measured as rung 3 of the ladder — see the 2026-08-17 entry on the folded stem, below.

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

---

## 2026-08-16 — The window between decibels and amplitude: one end measured, one end swept

**Decision.** `−29.84 dB` maps to 0.0 and `+10.16 dB` to 1.0, fixed in
`configs/kattegat-lane.yaml` and applied by `amplitude.DecibelStretch`. This supersedes the open
question left on 2026-08-14.

**The end that was measured.** LS-SSDD's sea sits at **0.2000** in the 0..1 the reader produces.
Measured over 2,234 offshore held-out tiles — scenes 11 to 15, the published split — outside the
annotated boxes with a 4 px margin, tile by tile rather than pooled. Putting this scene's sea
(−21.84 dB, robust) at that value fixes the floor once the width is known.

Two corrections were needed to get that number, and both were found by the number being wrong.
The first measurement gave a sea of 0.235 ± 0.209 — a spread almost as large as its median, over
a histogram decaying monotonically to white. That is land: the held-out half is cut from whole
acquisitions and contains harbours and shoreline, and the dataset ships `test_inshore.txt` beside
`test_offshore.txt` to say so. Masking the annotated boxes removes ships, not coast. The second
was pooling: five acquisitions heaped together carry their differences in sea state and
calibration inside the spread.

**The end that could not be measured, and why.** Matching the *spread* of the two seas sets the
width of the window from how grainy each product is. LS-SSDD's sea has a relative spread near
0.8; the Sentinel-1 GRD this chain exports is near 0.27 in the same units. That ratio of three is
a difference in how many looks were averaged, not in how the bytes were made, and no statistic of
dispersion can separate the one from the other. Anchoring on ship brightness instead fails the
other way: with hulls at their 95th percentile it asks for a window of a hundred decibels, which
puts the whole scene in the bottom fifth of the range.

**How the width was chosen.** Swept from 25 to 60 dB and scored against what is known — the
vessels this scene's AIS declares — with the tolerance the fusion already applies.
`notebooks/sweep_window.py` reproduces it. Every width from 25 to 45 recovers all six hulls
standing in the frame; 50 and above start losing them. 40 dB is the middle of what works.

**What this is worth.** One free parameter, chosen on one scene, and reported on the same scene.
It is tuning on the evaluation, it is written here as that, and the numbers that carry weight
remain the held-out LS-SSDD table. What it is not is a number chosen by eye: the plateau has
measured edges and the choice sits between them.

---

## 2026-08-16 — The chain cuts at the size the model was scored at

**Decision.** `configs/kattegat-lane.yaml` tiles at 800/64 rather than 512/64, and
`cli.check_tile_size` refuses a run where the two disagree.

**Why.** The model is built with `min_size = max_size = 800`, so a 512 px tile would be resized
to 800 by the transform inside it. This project refuses to resample radar amplitude elsewhere in
as many words — `_check_working_crs` will not reproject a scene for the same reason — and a
silent resize inside a model is worse than a loud one at the boundary.

**Why not build the model at 512 instead.** It would also resample nothing: the network is fully
convolutional and its anchors are in input pixels, so hulls keep their native size. It was
rejected because the chain would then run at a scale the model has never been scored at, and this
ticket exists to measure the model's contribution rather than assume it. An unmeasured variable
introduced into exactly that comparison is the one thing it cannot afford.

**What made it possible.** `Tiling` returns all nine tiles of this 1727 x 1845 scene at exactly
800 x 800, with no short tile at the far edge. A 736 px edge tile would have been resized by the
transform this decision exists to keep idle, and the decision would have been worth nothing.

---

## 2026-08-16 — A nodata hole is filled at the sea and guarded afterwards

**Decision.** `DecibelStretch` fills NaN at `sea_db` before applying the window, and
`without_holes` discards any detection whose centre falls in a hole.

**Why two mechanisms.** `scene.py` writes nodata as NaN precisely because every comparison
against NaN is false, which immunises a threshold detector. A convolutional network has no such
immunity: one NaN propagates through every convolution that touches it and empties the tile, with
no crash and no warning. Six per cent of the first real scene is nodata.

Filling at the floor instead — the obvious choice, and simpler to state — would paint that six
per cent as a perfectly black patch with a hard edge, and a hard edge is a strong feature for a
detector. The risk moves from the hole to its outline rather than going away. Filling at the sea
leaves almost no contrast at the boundary.

The fill is not sufficient on its own, so the guard sits behind it and answers a different
question: not "what does a hole look like" but "may a hole be reported". Each can be removed
without silently disabling the other, which is why they are two functions and not one.

**What was rejected.** Synthetic speckle in the holes. It would put the fill fully inside the
training distribution, and it invents amplitude — which this project already refuses on the
augmentation side, where a contrast jitter is described as producing a ship made of a different
material.

---

## 2026-08-16 — Where the trained weights live, and what names them

**Decision.** `models/epoch-012.pt`, outside git. `.gitignore` already ignores `*.pt`, so no new
rule was needed. The repository carries the path, the provenance and the digest, not 330 MB.

**Provenance.** Epoch 12 of the training run of 2026-08-14, `configs/train.yaml` at seed
20260814, brought down from that notebook version's Kaggle output.
`sha256 396b0cc1b2d3886dfd027571f6357657bbd1062dac2eb11129ee39c9d0f3e467`. It carries the
optimiser state, which inference does not read.

Epoch 9 scored better — F1 0.817 against epoch 12's 0.807 — and `keep: 2` had deleted it before
the run finished. That is exactly the cost the "keep the last, not the best" decision was written
down to accept, and this is it being paid.

**What the checkpoint does not say.** What built it. `AnchorGenerator` holds no parameters and
`min_size`/`max_size` are attributes of the transform, so a state dict fitted under one set of
anchors loads into a model looking for another without a word. `train.py` now writes its build
block into every checkpoint; this one predates that, so `configs/kattegat-lane.yaml` restates the
values and `TrainedDetector` accepts a checkpoint with no block while refusing one that
disagrees.

---

## 2026-08-16 — The declaration is moved into the radar's frame, not the tolerance widened

**Decision.** Before matching, each declared position is displaced along the satellite's ground
track by `fusion/azimuth.py`, using the velocity the AIS track already gives. The match tolerance
stays at 200 m.

**Why not widen the tolerance.** It is the obvious repair and it is the wrong one. Widening to
600 m would buy back the four false accusations by making every match looser, so a genuinely
dark vessel passing within half a kilometre of a declared one would be quietly explained away. It
trades a false alarm for a miss, and it hides the physics behind a bigger number. Moving the
declaration keeps the tolerance meaning what it says: how far a detection may sit from where the
vessel *should have been drawn*.

**What the correction is made of.** The direction is the ground track, which follows from
Sentinel-1's inclination and the latitude, and the pass direction is already on every product
this chain exports — it just was not being read. The magnitude is slant range over platform
speed, times the sine that turns a ground velocity into a line-of-sight one.

**The part that is approximated, in the open.** Slant range needs the incidence angle, and the
product does not carry it. `fusion.azimuth.incidence_deg` declares it, defaulting to 38.5° — the
middle of an IW swath. Across the swath the constant runs from 50 to 90 seconds, so this is not a
detail: it is a fifth of the correction.

**What it recovers, measured.** On the Kattegat scene, matched vessels go from 2 to 5 of the 6
standing in frame. The sixth is not recoverable at any incidence angle in the swath: 34° gives 4,
38.5° and 43° give 5, and 46° over-corrects back to 4. So the residual is not in the magnitude —
it is in the bearing, or in where a detection's centroid falls, and six vessels with a
pixel-resolution peak finder cannot separate those. Recorded as unresolved rather than tuned
away: choosing the incidence angle that made the sixth match would be fitting the geometry to the
answer, which is the one thing this correction must not do.

**What is still owed.** Earth Engine publishes a per-pixel `angle` band on COPERNICUS/S1_GRD. The
export should record it, so a scene carries its own incidence angle instead of accepting the
middle of the swath. Written down here rather than left as a gap.

---

## 2026-08-17 — What counts as a rung helping, decided before anything ran

**Decision.** A rung of the small-target ladder is kept if its best F1 across the reported score
thresholds, at its final epoch, exceeds the previous kept rung's by **more** than the range of
that same statistic over the previous rung's last four epochs. A gain exactly equal to that range
is a rejection. `ladder.py` applies it; `configs/ladder.yaml` names the rungs it is applied to.

**Why a rule rather than a reading.** The baseline did not converge — see the failure log — and
its precision at a fixed threshold moved by a factor of three between adjacent epochs of a single
run. The gain a better anchor size buys is plausibly two or three points of F1, which is smaller
than that. Under those conditions "this change helped" is a sentence that can be written about
almost any pair of numbers, and the only defence is to fix what would count as help before seeing
any of it.

**Why the band is measured rather than assumed.** It is the previous rung's own dispersion over
its last four epochs, so a configuration that settles buys a tighter test for the next change and
one that does not pays for it. Nothing here is a claim about how much noise there ought to be.

**Why strictly greater.** A gain that only reaches the noise the previous rung was already showing
is noise. One character, and it is the difference between a ladder and a narration.

**The fallback, decided now rather than when it is needed.** If the cosine decay does not settle
the run — R1's band over its last four epochs stays the same order as R0's — the statistic becomes
the median over the last four epochs rather than the final one, the band stays the range, and the
finding that a decaying rate did not settle this configuration is recorded in the failure log.
Deciding this in advance is what stops it being an escape hatch.

**What this ticket cannot do about the deeper problem.** One run per rung, one seed. A rung whose
gain clears the band could still be a lucky draw, and only repeated seeds would settle that. It
would double a thirteen-hour budget on a free tier, and it is recorded here as the honest limit of
what these five numbers support rather than papered over.

---

## 2026-08-17 — The folded stem agrees with the repeat inside the tile, and not at its edge

**Decision.** Rung 3's single-channel stem is measured and used on the strength of one property:
`conv1`'s output agrees with the three-channel repeat's away from a three-pixel border, and every
parameter and buffer outside `conv1` — including which layers are trainable — is exactly the
repeat's own. It is not numerically identical to the repeat everywhere, and an earlier draft of
this project's own documents said it was; that claim was unqualified and wrong, and this entry is
the correction.

**Why the border disagrees.** `conv1` pads with three rings of zeros before it convolves. Under
the repeat stem the tile is normalised *before* the padding is added, so a padded zero sits in
normalised space and stands for raw amplitude `m_c` — ImageNet's per-channel mean, a different
value on each of the three channels. Under the folded stem the transform is the identity, so a
padded zero stands for a raw zero. No single padding value reconciles the two: it would have to
satisfy `v · A_k = B_k` for every output channel `k` at once, and those ratios differ per channel.
So the two stems agree wherever the kernel does not reach the tile's edge, and disagree by most of
the signal where it does — a boundary convention, not a difference in what either stem starts
from.

**What was measured.** On random weights, `conv1`'s output agrees to **1.5e-06** outside a
three-pixel margin at the 64 px tile `tests/test_model_stem.py` fixes — a border that a change to
the tile size is not expected to shrink, because it is a property of the kernel and the padding
rather than of the tile size. That expectation was checked once at 256 px too, in a scratch script
run during implementation and never committed, where the same margin held; that figure is recorded
here as what was observed then, not as something a test holds now. The mechanism was confirmed
rather than assumed: with the repeat's mean set to zero, a padded zero means the same raw value
under both conventions, and the two whole backbones — fifty layers, not just `conv1` — then agree
to **3.8e-04** on a signal of scale **142**, which is float32 accumulation and nothing more. That
the disagreement collapses to accumulation noise once the one thing that differs between the
conventions is equalised is what says the border gap is the padding and not some other scale
artefact in the fold arithmetic.

**What the property is used for.** Rung 3 measures what training does with one bank of kernels
folded down to one channel, rather than a different starting point. That claim only holds if the
folded stem and the repeat it replaces agree at initialisation everywhere the FPN's receptive
field actually looks — and `C5`'s receptive field is the whole tile, so a border difference does
not stay confined to the border; it reaches every level of the pyramid. The property above is
what is available instead: agreement inside the tile, a stated and measured disagreement at its
edge, and everything outside `conv1` identical including trainability. `tests/test_model_stem.py`
holds both halves at the one tile size it fixes, 64 px: agreement in the interior and, as its own
complement, a genuine disagreement at the border.

**What it costs.** The comparison rung 3 makes is not "the same model, one line different at
training time" quite as cleanly as the rest of the ladder claims for its own rungs — a
three-pixel border's worth of the tile starts from a different convention. It is recorded here
rather than smoothed over, because the alternative was a spec that stated a stronger property than
the code delivers.

---

## 2026-08-19 — The anchor census: what the pyramid actually matches, and how big a ship is

**Decision.** Rung 4 ships `rpn_batch_size_per_image: 32`, no longer provisional. Rung 2 keeps the
five pyramid levels it has. Both are settled by `notebooks/anchor_census.py`, run on a Kaggle CPU
session over the 1123 ship-bearing tiles of the training split — 3637 ships, no GPU quota spent.

**What it measured.**

| | positives per tile | rescue-only boxes | by pyramid level | realised fraction |
| --- | --- | --- | --- | --- |
| stock `((32,),(64,),(128,),(256,),(512,))` | mean 97.6, max 3098 | 3257 / 3637 | `{0: 109506, 1: 121}` | 0.168 |
| small `((4,),(8,),(16,),(32,),(64,))` | mean 3.6, max 81 | 3524 / 3637 | `{0: 469, 1: 1338, 2: 1507, 3: 662, 4: 57}` | 0.014 |

**Feature levels, which is what criterion 2 asks for.** Under the stock sizes three of the five
pyramid levels never match anything at all: every positive anchor but 121 of them sits on level 0.
Under the small sizes the matches spread across all five, with the bulk on levels 1 to 3 and level
4 still carrying 57. That is the argument for keeping five levels rather than trimming the coarse
ones, and it is a count rather than an inference from the stride arithmetic — which is the form the
issue asked the reasoning to take.

**The stock set's 97.6 positives per tile are an artefact, not a finding.** 90% of ships never
reach the 0.7 foreground threshold under it; they are matched only because
`allow_low_quality_matches` guarantees every box its best anchor. When a 16 px ship sits entirely
inside a 32 px anchor the overlap is `256/1024 = 0.25` for *every* anchor that contains it,
identically — so they tie at the maximum and the rescue rule forces all of them positive together.
One tile produced 3098 that way. The number is large because the anchors are wrong, not despite it.

**The prediction, and where it was wrong.** The census script recorded, before it ran, that the
realised positive fraction would be near 1% rather than the 50% ceiling. Under the configuration
rung 4 actually runs — small anchors — it is **1.4%**, and the sampler never approaches its cap of
128. Under the stock set it is **16.8%**, and the prediction is simply wrong there, for the reason
above: ties inflate the count. Recorded as wrong rather than quietly narrowed to the case that
held, which is this project's rule for its own predictions.

**Why 32.** With 3.6 positives to a tile the sampler's ceiling is idle, so what moves the realised
fraction is the batch it fills: 256 gives 1.4%, 64 gives 5.6%, 32 gives 11.3%, 16 gives 22.5%.
32 is the middle of that, and it is chosen for being the middle rather than for a target anyone can
defend — the rung measures whether it helps, and if it does not, that is a rejection with numbers.

**What the census contradicts.** The longest side of a labelled ship is 6.0 px at the fifth
percentile, **16.0 px at the median** and 42.0 px at the ninety-fifth. At 10 m that is a median
hull of 160 m. This repository says in several places — the README, `model.py`, `metrics.py` —
that the problem is "a hull three pixels across", and that a 60 m vessel is six pixels. The
arithmetic in those sentences is right and describes the fifth percentile; the framing generalises
it to the whole set, and the whole set is three times larger than that. Nothing already published
depends on it, and it is corrected here rather than edited away at each site, so that the gap
between the prose and the measurement is on the record.

**A reservation, written before the runs rather than after them.** The small anchors give
twenty-seven times *fewer* positives than the stock ones and a slightly worse rescue-only rate,
97% against 90%. Under both sets almost no ship reaches 0.7. That points at the RPN's foreground
IoU threshold as the binding constraint rather than the anchor sizes or the sampler — a hypothesis
this ladder does not test, because its rungs were fixed before the census existed. Rung 2 will
measure what it measures; if the small anchors do not help, the census predicted it here, in
writing, beforehand.

---

## 2026-08-20 — The Kaggle mirror does not ship LS-SSDD's own layout

**Decision.** `Layout.images` may now name a sequence of directories, not only one, and
`catalogue` reads their union, sorted by name exactly as the single-directory case already was.
`configs/train.yaml` and `notebooks/anchor_census.py` are pointed at the Kaggle mirror
(`petrarodriguez/ls-ssdd-v1-0`) as it is actually mounted, verified in a live session, rather
than at the layout LS-SSDD itself publishes.

**What differs.** LS-SSDD-v1.0-OPEN, as published, ships one `JPEGImages` directory of 9000 .jpg
and one `Annotations` directory of 9000 .xml. The mirror instead splits the images into
`JPEGImages_sub_train/JPEGImages_sub_train` (6000 .jpg) and
`JPEGImages_sub_test/JPEGImages_sub_test` (3000 .jpg) — doubly nested, not a typo in this entry
— while keeping all 9000 annotations in one directory, `Annotations_sub/Annotations_sub`. Kaggle
has also changed how it mounts a dataset: `/kaggle/input/<slug>`, the path every config in this
repository named until now, is gone; a dataset now mounts at
`/kaggle/input/datasets/<owner>/<slug>`. Both changes are unannounced, downstream of Kaggle and
the mirror's owner respectively, and neither is this repository's to fix — only to read correctly.

**Why both image directories have to be named, and not only the train one.**
`split_by_scene` holds out scenes 11 to 15 by filtering on the scene number encoded in each
tile's filename. It is a pure filter and does not raise. Point `Layout.images` at the train
directory alone — which is what the single-directory shape this project shipped until now would
be forced to do, since the mirror's train directory holds none of the held-out scenes — and the
held-out split comes back empty, silently, over a training run that runs to completion and
reports the number as if it meant something. That is the failure this change exists to close, not
a convenience for reading two directories instead of one. `catalogue` also refuses a stem that
names an image under two of the directories, for the adjacent reason: `_annotation_at` looks an
annotation up by stem alone, and a duplicate would attach one label to two images and train the
tile twice under it. Nothing in the mirror above triggers it — the two directories are disjoint —
which is exactly why it is guarded in code rather than left to be true by observation.

**Cost.** This is read off a live session's directory listing, not verified here against a
download of the mirror — the test suite's fixtures are hand-built at a few tiles, as they have
been since `dataset.py`'s test file was written, and could not check the real 9000-file layout
even if the dataset were fetched into CI. A further change on Kaggle's or the mirror owner's side
would not be caught until a run attached to it failed to find its images, and would fail loudly
rather than quietly — `catalogue`'s `FileNotFoundError` names the directory that came back empty
— which is the property this repository can actually promise here.

---

## 2026-08-23 — R0, the baseline the ladder is measured against

**Decision.** `docs/runs/r0-baseline.json` is the reference every rung of issue #11's ladder is
compared to. Its statistic is **F1 0.807**, and the noise band over its last four epochs is
**0.026**, so R1 is kept only if it reaches strictly more than **0.833**. The rule that produces
that threshold was written and committed on 2026-08-17, before any of these runs existed; the
number it yields is only knowable now.

**What the statistic is, exactly.** `best_f1` in `ladder.py` takes the **final** epoch and the
best F1 across the confidence thresholds *within* it. It is not the best epoch. Epoch 9 of this
run reached 0.821 and epoch 12 reached 0.807, and 0.807 is the number the ladder uses — a rung
that trains for twelve epochs is judged on the twelve it was given, not on the one that happened
to land well. Recorded here because the two are three thousandths apart and easy to confuse when
reading the table.

**The run, on the held-out scenes 11 to 15.** 3000 tiles, 2378 labelled ships. At epoch 12 and
threshold 0.75: 1706 ships found, 672 missed, 142 false detections — precision 0.923, recall
0.717. More than nine in ten of what it reports is a real vessel, and it misses close to three in
ten of them. It trained on 2246 tiles, which is every ship-bearing tile of the 6000 in the
training scenes plus one empty tile each, at `empty_per_ship_tile: 1.0`.

**What the epochs show, and why rung 1 exists.** F1 by epoch: 0.485, 0.782, 0.809, 0.803, 0.803,
0.806, 0.810, 0.775, 0.821, 0.795, 0.804, 0.807. The model reaches the neighbourhood of its
optimum in three epochs and bounces inside it for nine more, at a learning rate that never
changes. That is the same behaviour `docs/failures.md` recorded of the first run on 2026-08-14,
and it is what rung 1 — cosine decay — was put on the ladder to test. The bounce is also what
sets the bar: the 0.026 band is measured, not assumed, so a configuration that fails to settle
makes the next change harder to prove rather than easier.

**Which execution these numbers come from.** The interactive session of 2026-08-23, downloaded
from `/kaggle/working` in the session's own output panel — *not* from a Save Version. The runbook
tells the operator to take the metrics from the saved artefact, and that instruction exists
because Save Version re-executes the whole notebook in a fresh machine and the artefact then
comes from a second run of the same code (`docs/failures.md`, 2026-08-14). Downloading the file
from the running session avoids that divergence rather than falling into it: this file is the
execution whose console log was watched. The runbook's instruction is the right default when a
version is being saved, and is not what happened here, so the provenance is written down instead
of being left to be inferred from the file's name.

**Cost.** Nothing in the test suite holds this number — it is a measurement, not a decision the
code can be made to enforce, and a rerun on a different machine will not reproduce it to three
decimals. What *is* held is the rule that consumes it: `tests/test_ladder.py` pins the arithmetic
of the band and the strictness of the comparison, so the threshold above cannot drift without a
test failing.

---

## 2026-08-23 — R1, cosine decay: kept, on the band rather than on the gain

**Decision.** `configs/ladder/r1-cosine.yaml` is kept. Its statistic is **F1 0.8356** against a bar
of **0.8335** — R0's 0.8074 plus R0's band of 0.0261 — so it clears by **0.0021**. Every rung above
it therefore stands on cosine decay, and `configs/ladder/r2-anchors.yaml` keeps its
`extends: r1-cosine.yaml` unchanged.

**The margin is not what carries this, and should not be read as if it were.** 0.0021 is thinner
than the run-to-run variation this project has no measurement of: no configuration has ever been
run twice here, so the only noise figure available is the within-run band, which is a different
quantity. A keep resting on two thousandths would be a coin toss dressed as a verdict.

What carries it is the second number the rule reports. **R1's own band is 0.0099, against R0's
0.0261** — the oscillation the rung was put on the ladder to remove has been cut to a third. The
last four epochs make the difference plain:

| | epoch 9 | epoch 10 | epoch 11 | epoch 12 | band |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.821 | 0.795 | 0.804 | 0.807 | 0.026 |
| R1 | 0.828 | 0.826 | 0.833 | 0.836 | 0.010 |

R0 falls 0.026 between epochs 9 and 10 and never recovers its best. R1 climbs, and tightens as the
rate decays through 1.25e-03, 7.32e-04, 3.35e-04, 8.52e-05. This is the behaviour the rung claimed
it would produce, written down on 2026-08-17 before a number existed to check it against.

Two things rule out a lucky final epoch. **R1's epoch 12 (0.8356) beats R0's best epoch of the
twelve** (0.8208, at epoch 9), so the gain does not depend on which epoch the statistic reads. And
the separation opens at epoch 7 and holds for six epochs — R0 runs 0.810, 0.775, 0.821, 0.795,
0.804, 0.807 while R1 runs 0.832, 0.810, 0.827, 0.826, 0.833, 0.836.

**What the run actually does differently.** At epoch 12 and threshold 0.75, R1 finds 1959 of the
2378 held-out ships, misses 419 and reports 352 false — precision 0.848, recall 0.824. R0 at the
same threshold found 1706, missed 672 and reported 142 false — precision 0.923, recall 0.717. So
R1 is not uniformly better at a fixed threshold: it finds 253 more ships and pays 210 more false
detections for them. What improved is the balance, and with it the calibration — R1 at threshold
**0.90** reports 1695 ships with 90 false, which is R0's recall at 0.75 (1706 ships) for 52 fewer
false detections. The scores mean more than they did.

**What this costs R2.** The rule measures the band on the rung being compared against, so R2 is
judged against R1's 0.0099 rather than R0's 0.0261: **R2 is kept only above 0.8454**. R1 halved its
own cushion, which is the rule working as designed — a configuration that settles buys a tighter
test for the next change. R2 will have to earn a real 0.010 rather than clear 0.026 of noise.

**An observation that does not change the verdict.** Epoch 1 of R1 and epoch 1 of R0 ran under
identical conditions — same seed, same tiles, same rate of 5.00e-03, the scheduler not yet
stepped — and did not agree: loss 0.1813 against 0.1920, best F1 0.801 against 0.485. Seeding here
is thorough (`torch.manual_seed` before the model is built, a seeded generator per epoch) but
there is no `torch.use_deterministic_algorithms`, so cuDNN algorithm choice and atomic
accumulation remain free. The gap closed immediately: epoch 2 reads 0.784 against 0.782. R0's
epoch 1 was pathological rather than representative — 32 detections above 0.75, a model not yet
calibrated, where a small shift moves F1 enormously. Recorded because it is the only evidence this
repository has about between-run variation, and it is weak evidence pointing both ways: large
where the model is unsettled, negligible one epoch later.

**What a test holds.** The verdict itself is a measurement and cannot be enforced in code, as with
R0. What is now held is its consequence. `test_every_rung_resolves_to_the_cosine_schedule_r1_was_
kept_for` in `tests/test_config.py` asserts that all four rungs resolve to `lr_schedule: cosine`,
because a rung that lost it would measure two changes and report one — R3 would be "the stem, and
the decay given back" — and its band would widen towards R0's, loosening the bar for the rung
after it. This was unguarded until now: the sibling test compares a rung to *whatever its own
`extends` names*, so `r2-anchors.yaml` repointed at `../train.yaml` still differs from its base by
exactly one key. That revert was made and all 301 tests passed before the guard was written.
Proved by three reverts, each failing by name: r2 repointed at the baseline (3 failures), r3
repointed at the baseline (2), and `cosine` turned back to `constant` in r1 itself (5, the extra
two being the sibling test noticing that r1 then changes nothing at all).

The guard lives in `test_config.py` rather than beside the other ladder tests in
`test_training_run.py`, which is skipped wherever torch is absent — CI included. `load_config`
resolves `extends` without torch, so the check runs everywhere the repository is tested.

**Which execution these numbers come from.** The interactive session of 2026-08-23, downloaded
from `/kaggle/working` in the session's own output panel — not from a Save Version, which would
have re-executed the notebook in a fresh machine and produced a second run of the same code. Same
provenance as R0, deliberately: a bar of 0.8335 computed from one kind of execution and compared
against the other would be measuring the machine as much as the change.

---

## 2026-08-25 — Anchor sizes and pyramid levels, chosen by measurement rather than by default

**Decision.** This project ships torchvision's stock anchor sizes,
`((32,), (64,), (128,), (256,), (512,))`, over all five levels of the feature pyramid. Criterion 2
of issue #11 asks for that choice and its reasoning in the decision log; this entry is the
reasoning, and what is new in it is that the choice is now the outcome of a measurement rather
than the default nobody had tested.

**Why the stock sizes.** Because the alternative was run and lost. `configs/ladder/r2-anchors.yaml`
took the sizes down to `((4,), (8,), (16,), (32,), (64,))` — 40 m upwards at 10 m resolution
rather than 320 m upwards — on the argument that the smallest stock anchor is longer than nearly
anything in the training set. It scored F1 0.788 against R1's 0.836 and against R0's own 0.807:
worse than the configuration it changed, and worse than the untouched baseline. `docs/failures.md`
carries the numbers and the mechanism.

So "the stock sizes" is not "we did not look". It is a rejected alternative, with a session of GPU
time spent establishing it and a pre-registered prediction that called it.

**What the sizes are actually doing, which is not what they appear to be doing.** The census of
2026-08-19 counted, over the 1123 ship-bearing training tiles and their 3637 ships, a mean of 97.6
positive anchors per tile under the stock set. That number is an artefact. Ninety percent of ships
never reach the 0.7 foreground IoU threshold against any stock anchor; they are matched only
because `allow_low_quality_matches` guarantees every box its best anchor, and when a 16 px ship
sits inside a 32 px anchor the overlap is `256/1024 = 0.25` for *every* anchor containing it,
identically — so they tie at the maximum and the rescue rule forces all of them positive at once.
One tile produced 3098 that way.

The stock anchors therefore work here **through the rescue rule rather than through fitting the
targets**, and the honest statement of criterion 2 is that: the sizes that win are not the sizes
that match, they are the sizes whose failure to match is repaired most usefully.

**How big a ship actually is.** Longest side, over the same 3637: 6.0 px at the fifth percentile,
**16.0 px at the median**, 42.0 px at the ninety-fifth. At 10 m that is a median hull of 160 m.
Recorded again here because this repository said in several places that the problem is "a hull
three pixels across" — arithmetic that is right about the fifth percentile and wrong as a
description of the set, which is three times larger than that.

**Feature levels, and where the earlier argument no longer holds.** The census settled the levels
by counting which ones ever match a ship, rather than by arguing from stride arithmetic, and that
was the form criterion 2 asked for. But the count depends on the anchor set, and the entry of
2026-08-19 read it under the set the ladder then expected to keep:

| anchor set | positive anchors, by pyramid level |
| --- | --- |
| small `((4,) … (64,))` | `{0: 469, 1: 1338, 2: 1507, 3: 662, 4: 57}` |
| stock `((32,) … (512,))` | `{0: 109506, 1: 121}` |

Under the small sizes the matches spread across all five levels, and keeping five was the
conclusion. Under the stock sizes — the ones this project now ships, R2 having been rejected —
**three of the five levels never match anything at all**, and level 1 matches 121 times against
level 0's 109506. The argument for five levels was contingent on a rung that fell.

**What follows, and what deliberately does not.** Levels 2, 3 and 4 carry parameters and produce
proposals that can only ever be negatives, on this data with these sizes. Trimming them is a
reasoned candidate for a sixth rung: it would reduce the model without touching what it detects,
and the census already says what it would cost, which is nothing in positives.

It is not done here, for the reason every deferral in this ticket gives. The five rungs were fixed
on 2026-08-17, and a change introduced after the ladder's results are known is a change measured
against nothing — this ladder's rule exists precisely to refuse that. Trimming the pyramid is
recorded as reasoned and deferred, which is the state the design document anticipated for it on
2026-08-17, before the census had run.

**Cost.** Three of five pyramid levels are shipped doing no detection work, knowingly, until a
rung tests removing them. And criterion 2 is answered with a measurement whose sign is the
opposite of the criterion's premise: the ticket asks for anchors "chosen for small targets", and
what the data supports is anchors that are far too large, rescued by a matching rule. Recorded as
answered-and-contradicted rather than reported as satisfied.

---

## 2026-08-25 — Issue #11, criterion by criterion

**Decision.** The ladder is complete and issue #11 is answered. Two of its five criteria are met
as written; three are answered by measurements that did not go the way the criterion's text
assumes. This entry says which is which, because a ticket closed on "three architectural changes
shipped" would be false, and one closed on "nothing worked" would be false too.

**1. An input stage adapted to radar polarisation channels — not met as written, answered.**
The criterion, sharpened by `model.py` to "a dual-polarisation stem trained as one", has no data
on either side of the chain: LS-SSDD-v1.0 is VV throughout, and the Kattegat export is VV because
Earth Engine's 48 MiB limit forced a choice between area and polarisation. Recorded in
`docs/failures.md`, 2026-08-17, with what a real answer would cost.

A single-channel stem shipped in its place, folded down from the pretrained RGB kernels, and R3
measured it: F1 0.83556 against 0.83557. The three copies of one amplitude channel were not
costing anything, to five decimal places. The criterion is unmet on data, and the substitute is
measured rather than asserted.

**2. Feature levels and anchor sizing chosen for small targets, with the reasoning here — met,
with its premise contradicted.** The entries of 2026-08-19 and 2026-08-25 above carry the counts
and the argument. The reasoning takes the form the criterion asked for — per-level positive-anchor
counts over 3637 ships rather than an inference from stride arithmetic — and its conclusion is the
opposite of the criterion's premise: anchors sized *for* small targets lost by 0.048, and what the
data supports is anchors far too large, rescued by `allow_low_quality_matches`. Also recorded
there: under the shipped sizes three of five pyramid levels match nothing, and trimming them is
reasoned and deferred to a rung this ladder does not have.

**3. Foreground/background imbalance addressed at the loss — addressed, rejected inside the
noise.** R4 took `rpn_batch_size_per_image` from 256 to 32 and lost 0.0087 against a band of
0.0099 — indistinguishable from the configuration it changed. `docs/failures.md`, 2026-08-25, also
records that the arithmetic justifying 32 was measured under R2's anchors, which the ladder
rejected, so what R4 actually measured is a smaller batch under the stock anchors. The value was
not rechosen after the fact, and the entry says why.

**4. Each change measured against the previous configuration on the same held-out split — met.**
This is the criterion the design document called one of the two that decide whether the ticket is
worth anything, and it is the one this work spent most of its effort on. Five runs of twelve
epochs, one line different each, every one scored over scenes 11 to 15 entire — 3000 sub-images,
2378 ships — with the keep/reject rule and its noise band written and committed on 2026-08-17,
before the first run existed. `darkvessel compare` applies it mechanically;
`tests/test_ladder.py` pins the arithmetic and the strictness of the `>`; and
`test_no_rung_of_the_shipped_ladder_stands_on_one_the_rule_rejected` holds the greedy chain, which
it caught in the wild the moment R3's journal landed.

**5. Changes that did not help recorded in the failure log — met.** Three entries, dated
2026-08-23, 24 and 25, one per rejected rung, each with its numbers, its mechanism and its distinction from
the others: a clear harm, a draw to five decimals, and a draw inside the noise. The entry for R2
also records that the anchor census predicted its failure in writing on 2026-08-19, before any
rung had run.

**What the ticket produced.** One kept change out of five, and it is the one that was not among
the ticket's three adaptations: cosine decay of the learning rate, +0.028 over the baseline and a
noise band cut from 0.026 to 0.010. That is a smaller architectural result than the issue text
anticipates and a larger methodological one — the ladder now measures changes at a resolution the
2026-08-14 run could not have supported, and the three adaptations are refuted at that resolution
rather than left unproven.

**Cost, and the thing still standing.** Almost no ship reaches an IoU of 0.7 against any anchor in
either set, which points at the RPN's foreground IoU threshold rather than at anchor geometry or
sampler batch size. Two rungs failed in the region that hypothesis describes and neither tested
it, because the five were fixed before the census that produced it. Issue #11 closes without
having tested the most likely explanation of its own results, and that is stated here rather than
left for a reader to notice.

---

## 2026-08-25 — The chain loads the rung the ladder kept, at the operating point the swap chose

**Decision.** `configs/kattegat-lane.yaml` loads `models/r1-epoch-012.pt` — epoch 12 of R1, the
one rung of issue #11's ladder that was kept — at `score_threshold: 0.90`. It loaded the weights
of 2026-08-14 at 0.75 until now.

**Why this is an entry at all.** The ladder closed on 2026-08-25 having established that R1 scores
F1 0.836 against the baseline's 0.807, and the chain went on loading the baseline. Nothing
reported it: five rungs' worth of tests, a rule applied mechanically by `darkvessel compare`, a
test that holds the greedy chain of rungs — and not one of them looks at which checkpoint the
pipeline actually opens. The ladder proves which weights are best; the config is free to name any
others, and did, for two days. That is the gap this entry closes, and the reason the closing is a
test rather than a corrected line.

**Why 0.90, when the old config said 0.75.** Because the operating point is the decision and the
threshold is only its coordinate, and the coordinate does not survive a change of weights. Cosine
decay moves the calibration of the scores — the same property the 2026-08-14 run's oscillation was
made of, where precision at a fixed 0.50 went 0.55, 0.74, 0.75, 0.41 across adjacent epochs while
the loss sat still. So the two detectors read the same number differently:

| | threshold | precision | recall | found | false |
| --- | --- | --- | --- | --- | --- |
| Baseline, 2026-08-14 | 0.75 | 0.941 | 0.706 | 1680 | 106 |
| R1 | 0.75 | 0.848 | 0.824 | 1959 | 352 |
| **R1** | **0.90** | **0.950** | **0.713** | **1695** | **90** |
| Baseline, 2026-08-14 | 0.90 | 0.986 | 0.535 | 1272 | 18 |

Carrying 0.75 across would have moved this chain from one false alarm in seventeen to one in
seven inside a commit about a checkpoint path — a change of what the project is willing to accuse
a vessel of, made silently and by inheritance. Held at the precision the swap of 2026-08-16 was
decided on, R1 dominates the configuration it replaces on all three figures at once: 15 more ships
found, 16 fewer false alarms, 0.009 more precision. Nothing was traded, which is why this move
needed no argument about what a miss is worth against a false alarm. The rung that does trade —
R1 at 0.75, 279 more ships for 246 more false alarms — is a real option and a different decision,
and it is not this one.

**What is verified, and what is not.** Verified: the held-out numbers above, read out of
`docs/runs/r1-cosine.json`, and two tests in `tests/test_config.py` —
`test_the_chain_runs_the_weights_of_the_rung_the_ladder_kept` reads the ladder's verdict rather
than the string "R1", so the next rung to be kept fails it until the chain is repointed, and
`test_the_chains_score_threshold_holds_the_precision_the_swap_was_decided_on` refuses a threshold
the kept rung never scored or that buys less than 0.94 precision. Both were confirmed by reverting
this config and watching them fail.

**Not verified: anything this scene says.** R1's weights are on Kaggle, not on this machine —
330 MB, and `*.pt` is ignored — so the Kattegat run has not been executed under them. Everything
the README reports about that scene was measured with the baseline at 0.75 and stands unrepeated:
6 detections for 6 hulls with none on open water, the azimuth correction's 5 matched against 1
dark, and the decibel window's sweep, of which "every width from 25 to 45 recovers all six hulls"
is the part that was measured against a detector. The window's *floor* is unaffected — it is fixed
by LS-SSDD's sea sitting at 0.2000, which is a property of the amplitude domain both detectors
were fitted in, not of either one's weights.

**Cost.** Between this commit and that download, the shipped config names a file that exists on no
machine here, and `darkvessel run --config configs/kattegat-lane.yaml` fails until it does. That
is deliberate. The alternative — a config that describes R1 in its comments and loads the baseline
— is precisely the state this entry exists to end, and it is the state that survived two days
because it looked fine.

**What closes it.** Bring `/kaggle/working/checkpoints-r1/epoch-012.pt` down to
`models/r1-epoch-012.pt`, run the chain, and record the scene-level table again beside the one
from 2026-08-16. Until then the README says which of its numbers are the baseline's.

---

## 2026-08-26 — R1 run twice, and what two executions of one config agree on

**Decision.** The chain loads the weights of a **second execution** of
`configs/ladder/r1-cosine.yaml`, journalled in `docs/runs/r1-cosine-rerun.json`, and the config
names that journal in a new `run.trained.metrics` key. The ladder keeps judging the first
execution, `docs/runs/r1-cosine.json`, unchanged.

**Why there is a second execution at all.** The run of 2026-08-23 was interactive, and its numbers
were read out of the session's own output panel rather than a saved version — recorded above,
deliberately, so that R0 and R1 were compared under the same kind of execution. What that entry
did not say is that a checkpoint read out of a panel is a checkpoint that exists nowhere else. By
2026-08-26 the notebook's persistent `/kaggle/working` had been emptied, no version had ever been
saved, and `epoch-012.pt` was gone. The weights the ladder's whole argument rests on survived for
three days.

**The operational fact that cost it, now that it is known.** A Kaggle **Quick Save** does not
publish `/kaggle/working`. It renders the notebook to HTML and saves that; the version's output,
pulled with `kaggle kernels output`, is a 799-byte conversion log. Only a committed run —
*Save & Run All* — turns the working directory into an output dataset. Persistence is not a
substitute: it is a property of a session's workspace, and it was silently empty when it mattered.
Anything a run produces that is wanted afterwards has to leave through a committed run or be
downloaded before the session ends.

**What the two executions agree on.** Same config, same seed 20260814, same GPU class, twelve
epochs each. The run block — build, schedule, reporting, split sizes — is byte-identical between
the two journals.

| | first (2026-08-23) | second (2026-08-26) |
| --- | --- | --- |
| training loss, epoch 1 | 0.1813 | 0.1816 |
| training loss, epoch 12 | 0.1174 | 0.1167 |
| best F1, final epoch | 0.83557 | 0.83831 |
| noise band, last four epochs | 0.00986 | 0.01550 |
| precision at 0.90 | 0.950 | 0.946 |
| recall at 0.90 | 0.713 | 0.726 |

Training losses agree to within 0.0016 at every epoch, and the learning rates are identical to
three significant figures throughout — the cosine schedule is arithmetic and does not drift. The
statistic the ladder is decided on differs by **0.0027, which is smaller than either run's own
noise band**, so under the rule this project committed to on 2026-08-17 the two executions are
indistinguishable: a change of that size would have been rejected as noise.

**The verdicts survive.** Substituting the second execution for R1 and re-running `judge` returns
the same verdict on all five rungs — R0 and R1 kept, R2, R3 and R4 rejected — with R2 at −0.051
against −0.048, R3 at −0.003 against −0.000, R4 at −0.011 against −0.009. The conclusions of issue
#11 are therefore a property of the configurations rather than of the session that ran them, which
is the first time this project has been able to say so. Before the seeding fix of 2026-08-15 it
could not: the same config run twice produced two detectors that disagreed at every epoch
(`docs/failures.md`, 2026-08-14).

**Why the chain quotes the second and the ladder keeps the first.** Because a number must describe
the thing it is attached to. The ladder's table describes an experiment that was run and judged;
rewriting it with numbers from a later execution would move published verdicts under cover of
recovering a file. The chain's config describes weights someone may be sent out on the strength
of, and those weights come from the second run. So the two live side by side, each pointing at its
own journal, and `test_the_chains_score_threshold_holds_the_precision_the_swap_was_decided_on`
reads the one the config names rather than whichever is nearby.

The build blocks are held equal by a second test: an execution of a *different* rung would be a
different detector wearing R1's name, and anchors, tile size and stem leave no trace in a state
dict.

**What the chain does with them, on the scene.** `darkvessel run --config
configs/kattegat-lane.yaml`, fourteen seconds on an M1 laptop: **six detections, five matched, one
dark**, against twelve declared positions at a 200 m tolerance, every match on a position
interpolated to the acquisition. The same six vessels the 2026-08-14 detector found, by MMSI, and
every score higher — 0.850 → 0.976 on the 274 m vessel, 0.862 → 0.927 on the 24 m one.

That last pair is the result worth keeping. **At the 0.90 this chain now runs, the old weights
would have returned four of the six hulls**, dropping the largest vessel in the frame and the
smallest, both scored under the bar. The promotion is not a tenth of a point of F1 on a held-out
split; on this scene it is two ships.

**Cost, and what is still not repeated.** The match distances moved a few metres — the 274 m
vessel is now matched at 186 m against a tolerance of 200 m, where the old detector put it at
172 m. Nothing crossed the tolerance, and one match now sits 14 m nearer to being called dark than
it did. The decibel window was swept under the old weights at 0.75 and has not been re-swept; it
recovers all six hulls under these, which is what it was chosen to do, but the sweep itself
belongs to the other detector.

---

## 2026-08-26 — The embedding stage is optional, and the chain has to be unchanged by it

**Decision.** `pipeline.run` takes an `embedder` alongside the detector, defaulting to None. With
one, each detection's crop is described and the vector travels in the layer, one column per
dimension. With none, nothing is cut, no framework is imported, and what comes out is what came
out before this level was written.

**Why it is a parameter and not a stage.** The chain answers one question — which of these
detections declared themselves — and the seam exists so that the thing which answers it can be
substituted. A representation of what was found is a second question asked of the same
detections, and it is not always asked: `configs/pipeline.yaml` runs on a synthetic scene with a
threshold detector and has no encoder to load, and every config written before today has none
either. Making it a parameter with a default is what lets both be true at once — the optional
stage is injected exactly the way the detector is, and absence is spelled `None` rather than
being a branch inside the pipeline.

**Why the vector goes in the layer.** Sixteen columns is a wide attribute table and a cheap one.
The alternative is a file of vectors beside the GeoPackage, joined by a row order nobody states —
and a join that is a convention rather than a key is a join that silently stops corresponding to
anything the first time a stage sorts. `embedder.attach` refuses a length mismatch for the same
reason: attached by position, a mismatch would put every vector on the wrong vessel and still
write a layer that opens in QGIS.

**Verified.** `test_the_chain_runs_end_to_end_with_the_embedding_stage_disabled` and
`test_the_embedding_stage_adds_columns_and_changes_nothing_else` in `tests/test_pipeline.py`: the
second asserts the frame without the stage is exactly the frame with it, restricted to the
columns it had. `test_a_detection_is_described_by_the_pixels_around_it_and_not_by_its_neighbour`
is the one that would catch a crop order drifting from a row order — two targets, one of them
beside a bright patch no detection stands on, so a swap changes which row is the crowded one.

---

## 2026-08-26 — The archive is fifty acquisitions of one rectangle, cut at its own operating point

**Decision.** The representation is fitted on 348 crops from 49 acquisitions of the Kattegat box —
fifty were fetched and one held no detection at all —
between 2026-06-01 and 2026-08-10, ascending and descending both, cut from detections the trained
detector returns at a score of 0.05 rather than the 0.90 the chain publishes at.

**Why many acquisitions.** One acquisition of this rectangle holds six vessels. Six objects is not
an archive and a representation fitted on it has learned six objects; the question this level
exists to answer — which detections resemble which — is a question across the record rather than
inside one scene. Ten weeks of the same water is what makes it one, and it costs a gigabyte of
GeoTIFF and seven minutes of downloading, both of which are cheap against what they buy.

**Why a lower threshold.** 0.90 was chosen on 2026-08-25 for precision, because every unmatched
detection the chain publishes is a claim someone may be sent out on. Nothing here is published.
This is what a representation is fitted on, and one fitted only on the objects the detector was
already certain about has never been shown the ones it was not — which are exactly the objects
this level exists to make separable after the fact. It is the same detector at a different
operating point, both stated in one config file, each where it belongs; `_detector_from` takes the
override rather than a second detector being built.

**Consequence, accepted.** Two thirds of the crops have another detection within 200 m of them in
their own acquisition, because a detector at 0.05 cuts a large hull more than once. That is not
noise to be filtered — it is the second cut of a real ship — and what it required was a change to
how every check at this level counts, recorded below.

---

## 2026-08-26 — Speckle is the augmentation radar allows, and the looks are measured per scene

**Decision.** A contrastive view is one of the eight symmetries of the square, a translation of up
to eight pixels, and a multiplicative perturbation drawn from a Gamma distribution of 4.1 looks —
the median of what `views.looks_of` measures across the archive's own scenes. Nothing else.

**Why.** There are no labels at this level, so the augmentations *are* the supervision: what two
views of one crop have in common is the whole of what the representation is told to keep. The
spec's rule for the detector applies here with more force — colour and contrast jitter have no
physical meaning on a backscatter coefficient, and a network told to ignore a shift in decibels is
told to ignore the one measurement the image carries. Speckle is the exception the physics itself
provides: a multi-looked intensity image carries a fluctuation that is Gamma distributed with
shape equal to the number of looks, so a second look at the same sea is that sea times a draw from
that distribution. `dataset.py` said of it in August that "it needs a speckle model to be argued
for, and that belongs with the rest of the work on what this data actually is". This is that work.

**Why measured rather than quoted.** The nominal figure for an IW GRDH product is 4.4. Across
fifty acquisitions of one rectangle the measured figure runs from 0.01 to 5.14, and the spread is
the sea rather than the processor: a calm morning backscatters at -37 dB, close enough to the
noise floor that its relative variation in decibels is five times a windy day's. The estimator
measures the variability of this water, which is what an augmentation should be scaled to. The
median ships, because a view perturbed at the calmest scene's figure would be perturbed harder
than any real second look at this sea, and the mean would be dragged down by the same handful of
scenes.

**Verified.** `test_the_number_of_looks_is_recovered_from_a_sea_that_has_that_many` builds a
synthetic sea at 4.4 looks and requires the estimator to return it within five per cent — which is
what forced the sigma clip: cutting the brightest tenth by percentile, the obvious way to exclude
ships, removes most of what speckle is and reports 6.6.

---

## 2026-08-26 — Two cuts of one hull are one object, and every check at this level says so

**Decision.** `Archive.co_located` marks two crops as the same object when they come from one
acquisition and stand within the fusion's own match tolerance of each other, and both checks at
this level — the twin recall recorded every epoch, and the nearest-neighbour share reported by
`darkvessel retrieve` — count a hit on any of an object's cuts as a hit. The chance level moves
with the leniency: `chance_of` computes it from the same equivalence rather than assuming `1 / n`.

**Why.** A detector run at 0.05 cuts a large ship more than once. In this archive two thirds of
the crops have a neighbour within 200 m in their own acquisition, the median distance between
such a pair is 31 m, and there are 1.78 cuts per object. A ranking that called the second cut of a
vessel a wrong answer when the first was asked for would be measuring how duplicated the archive
is, and it would get worse as the detector got better at finding big ships. The same encoder
scores 0.316 under the strict rule and 0.483 under this one, and the 0.167 between them is
entirely duplication.

**Why the fusion's tolerance and not a new number.** Because this project already has a distance
at which two positions are one vessel, it is stated in the config, it is reported beside every
result the chain publishes, and inventing a second one here would mean two answers to one
question with nothing to keep them together. It is passed into the training run rather than
defaulted, so a config that changes it changes what the run reports and `Journal.describe` refuses
to fold the two into one file.

**What it does not excuse.** Two cuts of one hull are the *easiest* pair in the archive, so a
representation could score well on this alone. That is why `same_object` reports a third figure
beside it — how often a neighbour is a different object in the same acquisition, which is where a
representation that had learned the weather rather than the ship would show up. It comes out at
4%, against 66% same-object and 0.2% at chance — and the chance level for that figure is the one
that *excludes* the query, because retrieval takes it out before ranking. `chance_of` carries both
readings for that reason: the twin recall ranks against the crop's own vector and this does not,
and using one baseline for both would halve or double an apparent margin.

---

## 2026-08-26 — What the embedding level claims, and what it does not

**Decision.** The level ships with three numbers and a figure, all written by
`darkvessel retrieve` into `docs/runs/retrieval-kattegat.json` and
`docs/figures/retrieval-kattegat.svg`. It claims that retrieval returns visually similar objects.
It claims nothing about separating vessels from fixed structures, which is issue #14.

**What is measured.** Over 348 crops from 49 acquisitions, with a 16-dimensional representation
fitted in eleven minutes of laptop CPU:

| | measured | at chance |
| --- | --- | --- |
| A second view of a crop retrieves its object first | 0.483 | 0.005 |
| The nearest neighbour is another cut of the query's object | 66% | 0.2% |
| The nearest neighbour is a different object, same acquisition | 4% | — |
| The nearest *different* object differs in apparent size by | 20.0 px | 61.0 px |

**What each of them is worth.** The first needs no labels of any kind and is the one that fails
loudest: a representation that collapses onto a point returns ranked neighbours with similarities
near one for every query, and scores at chance here. The second is the strongest agreement a
representation of an object can show, because two cuts of one hull are the most similar pair the
archive contains. The third is the diagnostic, not a result — see the entry above. The fourth is the
only one that speaks to resemblance *between* objects, and it is ranked over everything the query
is not for that reason: measured over all neighbours it reads 2.0 px, which is the duplication in
the entry above restating itself. It is reported last because apparent size is measured from the
same pixels the encoder saw and is therefore not an independent label; what it rules out is a
representation whose neighbours are no closer in size than a crop drawn at random.

**What the figure adds.** Six queries spread over the archive's range of target size, each with
its four nearest neighbours, drawn through the same decibel window so that two cells side by side
are two crops in one unit. Two numbers can be satisfied by a representation that has learned
something real and useless; a reader looking at the sheet can see in a second that the rows are
coherent — point scatterers with point scatterers, elongated hulls with elongated hulls,
saturated crosses with saturated crosses. The queries are chosen by spreading them over that
range rather than by hand, because choosing six by hand is exactly where a flattering figure would
come from.

**An unannotated class, found.** The first row of the sheet is detections standing on the boundary
of a nodata hole, and its neighbours are other detections on other holes in other acquisitions.
Nothing labelled them, nothing was told they exist, and they sit together. That is the claim this
level was built to support, made on the least interesting class it could have been made on.

**What is not claimed.** That the representation transfers beyond this rectangle: one study area
does not support a claim about transfer and none is made. That the schedule was long enough — the
twin recall was still rising when it ended, 0.124 at epoch 1, 0.29 at 80, 0.35 at 200, 0.48 at
400, and a longer run is the obvious next experiment rather than a finished one. And that
clustering works, which has not been tried, because the study area moved off the Anholt wind farm
in August and the archive contains no fixed structures to separate.

---

## 2026-08-26 — The archive draws on two rectangles, and the second one is the farm the study area left

**Supersedes** *The archive is fifty acquisitions of one rectangle, cut at its own operating
point*, in the part that says one rectangle. Everything else in that entry stands, and the run it
describes is kept: `docs/runs/embedding-kattegat.json` and `docs/runs/retrieval-kattegat.json` are
the one-box archive, and they are the numbers issue #13 was closed on.

**Decision.** `configs/embeddings.yaml` names the boxes the archive draws on rather than borrowing
the run's study area, and it names two: the Kattegat lane the chain runs over, and the Anholt box
(11.15–11.40 E, 56.58–56.71 N) the study area moved off on 2026-08-14. A scene is named by its box
as well as its acquisition, and the scenes live one subdirectory per box.

**Why.** Issue #14 asks that fixed structures cluster separately in the embedding space and be
verified against known offshore wind farm locations. The archive it inherited could not answer
that at any quality of representation, because it contained no fixed structures: the study area
was moved onto the shipping lane precisely because Anholt had turbines and no ships, and the
archive was built from the lane alone. A clustering fitted on data holding no instance of the
class it exists to find is a figure, not a finding — so the missing half was fetched before the
method was written, rather than after it had produced something.

The two boxes are complements rather than a bigger sample of the same thing. The lane holds five
or six commercial hulls at an arbitrary instant and no fixed structure; Anholt holds a documented
111-turbine lattice and, across thirty acquisitions in 2026, never a vessel longer than 15 m.
Between them the archive holds both halves of the problem the representation exists to separate,
and neither box alone does.

**Why the box is a name and not a bounding box in the provenance.** Because a name reaches the
checks. One Sentinel-1 product can cover both rectangles, and written flat the second clip would
share a file name with the first — one of them silently skipped as already fetched. Named, they
are two scenes, which is what they are: two pieces of water, two sea states, two noise floors.
The `elsewhere` diagnostic in `same_object` asks whether a neighbour comes from the query's own
acquisition, and folding two clips into one acquisition would answer it wrongly in the direction
that flatters.

**Cost.** Another gigabyte of GeoTIFF, and the encoder refitted from scratch: the run block names
the crops and the scenes, so `Journal.describe` refuses to fold the two archives into one file —
which is the behaviour, not an obstacle to it.

---

## 2026-08-27 — What the two-box archive says, and what the one-box run still stands for

**Supersedes** *What the embedding level claims, and what it does not*, in its numbers. The run
that entry describes is kept whole — `docs/runs/embedding-kattegat.json`,
`docs/runs/retrieval-kattegat.json` and `docs/figures/retrieval-kattegat.svg` are the one-box
archive, and they are what issue #13 was closed on. What follows is a different archive, and
therefore a different measurement rather than a correction of that one.

**What the archive is.** 4676 crops from 96 acquisitions: 348 from the Kattegat lane and 4328 from
the Anholt box. The imbalance is the point rather than a flaw — 111 fixed scatterers stand in
every Anholt frame while five or six ships pass through the lane — and it is what makes the
archive hold both halves of the problem.

**The class is present, checked before a method was written to find it.**
`notebooks/recurrence.py` asks nothing of the embedding: it groups detections whose ground
positions fall within 100 m and counts the acquisitions each standing position appears in.

| Positions seen in… | kattegat-lane | anholt |
| --- | --- | --- |
| 2+ acquisitions | 21 | 91 |
| 5+ | 2 | 69 |
| 10+ | 1 | 67 |
| 20+ | 0 | 65 |
| most persistent | 11 acquisitions | 46 of 47 |
| crops at a position seen 5+ times | 23 of 348 | 4232 of 4328 |

*(The last row read `18 of 348` and `2612 of 4328` until 2026-08-27. The notebook that produced
it summed acquisition counts under a label that said crops — see docs/failures.md. The corrected
figures are above; nothing else in this entry depended on them.)*

The lane is a real control and not a rhetorical one: same detector, same threshold, same ten weeks,
same crop geometry, different water. Sixty-five standing positions against a documented 111
turbines is a partial recovery, and verifying which is #14's second criterion rather than this
check's. The lane's one position seen in 11 acquisitions is not a ship, is not explained here, and
is written down rather than tidied away.

**What the representation measures now.**

| | measured | at chance |
| --- | --- | --- |
| A second view of a crop retrieves its object first | 0.066 | 0.0004 |
| The nearest neighbour is another cut of the query's object | 71% | 0.04% |
| The nearest neighbour is a different object, same acquisition | 11% | — |
| The nearest *different* object differs in apparent size by | 6.0 px | 16.0 px |

**Why the first number is not the encoder getting worse.** It fell from 0.483 to 0.066, and the
archive changed underneath it: retrieving *this* turbine rather than one of its sixty-four
identical siblings, from a view shaken by speckle, is a question the one-box archive never asked.
So the two encoders were put to the identical task — the same 348 lane crops at the same indices,
the same augmented twins, ranked against those same 348 candidates. The one-box encoder scores
0.489 and this one 0.422. Of the 0.067 between them, none is a mystery: 93% of what this encoder
sees is turbines, so ships get a smaller share of a fixed capacity. That is the trade #14 needs
made, and it is stated rather than absorbed.

**What is still open.** The recall was rising when the schedule ended, so a longer fit is the
obvious next experiment. Whether 65 standing positions are 65 turbines is unverified. And the
window between decibels and amplitude is still the one fitted to a single Kattegat scene, applied
across two boxes whose seas run from -37 dB to -11 dB — the `elsewhere` figure says the
representation has not keyed on it at 11%, which is a bound rather than an all-clear.


---

## 2026-08-27 — Fixed structures are excluded on where they stand, not on what they look like

**Supersedes** *What the embedding level claims, and what it does not*, in the sentence that says
this project claims nothing about separating vessels from fixed structures. It does now. It also
answers the open question left by *What the two-box archive says* — whether the 65 standing
positions are turbines — and the answer is 64 turbines and a transformer platform.

**Decision.** The chain will not report a detection as a dark vessel when it stands at a position
in `data/reference/fixed-structures.csv`. That register holds 65 positions, every one of them a
place the archive carried a detection in 20 or more distinct acquisitions, and every one verified
against coordinates published by somebody else. The exclusion is applied after the AIS matching,
the excluded rows stay in the layer carrying `status = structure` and `structure_distance_m`, and
every run prints what it excluded whether or not it excluded anything.

**The clustering over the embedding space is not what does it.** Issue #14's premise was that
turbines cluster apart from vessels and can therefore be excluded wholesale, without labels. They
do cluster apart; that is measured below and it is a real result. Excluding on it is not.

### What recurrence says

A position that carries a detection acquisition after acquisition is not a ship. `standing()`
groups detections within 100 m of one another and counts the distinct acquisitions each group
appears in. Over the two-box archive, 318 distinct positions, of which:

| positions seen in… | kattegat-lane | anholt |
| --- | --- | --- |
| 2+ acquisitions | 21 | 91 |
| 5+ | 2 | 69 |
| 10+ | 1 | 67 |
| 20+ | **0** | **65** |

**Why the floor is 20.** It is the lowest floor at which every entry in the register stands on a
structure somebody else published — and that is the property that matters, because a register
entry nobody published is a coordinate at which this chain stops reporting dark vessels on the
strength of its own archive alone.

| floor | registered | published positions found | registered but unpublished |
| --- | --- | --- | --- |
| 5 | 71 | 66 of 66 | 3 |
| 10 | 68 | 66 of 66 | 2 |
| **20** | **65** | **65 of 66** | **0** |
| 30 | 63 | 63 of 66 | 0 |

At a floor of 10 the register contains the object in the Kattegat shipping lane that stands in 11
acquisitions and that nothing published explains. Silently excluding an unexplained recurring
object in a shipping lane is precisely the failure this chain exists not to commit, and it is
worth more than the one turbine that 20 gives up — which sits **73 m from the western edge of the
box**, is cut by the clip rather than missed by the method, and goes on being reported as a dark
candidate. An over-report, which is the safe direction. `test_the_shipped_register_holds_no
_position_the_published_lists_cannot_explain` fails if a future archive puts an unexplained entry
back in.

**The three distances, because swapping two of them would still verify and still run.** 100 m
decides what one standing object is: a registered structure's own detections sit 13 m from its
centre at the median and 108 m at the worst, and the masts stand 583 m apart at the closest, so it
is six times the wobble and a sixth of the spacing. 100 m again is the radius a register entry
explains at run time — the same number because it is the same question. 200 m is how far a
registered structure may sit from a published one and still be called the same structure, and it
is load-bearing for nothing: the matches it accepts are 5.1 m apart at the median and 15.8 m at
the worst, and the one position it rejects is 602 m out.

### What it was verified against, and what that reference is worth

**OpenStreetMap, through the Overpass API**, fetched by `darkvessel known` into
`data/reference/*-structures.csv` and kept in the repository so nothing else needs a network. It
is not the authoritative source: the authority for Danish turbines is Energistyrelsen's
Stamdataregister, which as of today is published through a map viewer rather than as a file this
could fetch. Two limits, both recorded per row in the file rather than left for a reader to find:
**108 of the 112 structures carry OSM's own `note=position only approximate`**, and OSM is a
volunteer record with no completeness guarantee.

What survives those limits is the agreement itself. An approximate volunteer list and an
independent ten-week radar archive place the same 65 objects **5.1 m apart at the median** — half
a pixel. A list that were badly wrong could not do that, and neither could a method that were.
The count agrees too, at the level the archive can see: OSM records 111 turbines at Anholt, which
is the documented size of the farm, and 65 of them plus the farm's transformer platform fall
inside the archive's rectangle.

**The platform is in the reference on purpose.** A reference containing only turbines would have
reported the single most persistent non-mast object in the archive — 37 acquisitions, 1759 m from
the nearest turbine — as this method's one false alarm. It is `Transformerplatform Anholt
Havmøllepark`, and it is a fixed structure by every argument that makes a mast one.

### What the clustering says, and why it is reported rather than acted on

Eight spherical k-means clusters over the 16-dimensional embeddings, seed 20260827. The embedding
**does** carry the distinction: ranked by similarity to the centre of the crops recurrence is sure
about, it orders standing crops ahead of the rest at **0.768** against 0.5 at chance, with no
threshold involved. Seven of the eight clusters are 94–97% standing crops and the eighth is 51%.
Those are identifiable clusters, and that is issue #14's first criterion met.

The second criterion is where it fails. Calling a cluster fixed when 80% or more of it stands
still, and excluding on that, would take out **62 of the 348 Kattegat lane crops** — 18% of the
box that contains no fixed structure at all, published or found. The register takes out **zero**
from that box. Nor is the rule the problem: labelling every cluster by its own majority against
the published coordinates — an oracle no unlabelled method could have — still leaves 71 to 115
lane crops inside structure-majority clusters at every k from 12 to 32, and at k ≤ 8 the oracle
calls *everything* a structure, because 92.5% of this archive is structures. A ranking can be good
while every cut through it is bad, and that is what a 0.768 separation with a 93/7 class balance
buys.

**So the honest reading of the premise is that it is half right.** The structures are separable in
the embedding space, measurably. They are not separable *well enough to delete a detection on*,
and deleting detections is what this was for. The number that decides it is 62 dark candidates
that would have stopped being reported in a shipping lane, and no representation quality argument
outweighs that.

### What it excludes, quantified

Over the archive, at the chain's own publishing threshold of 0.90 — the detections a run would
actually have reported:

| | detections | at a registered structure | remaining |
| --- | --- | --- | --- |
| archive, at its own 0.05 | 4676 | 4187 (89.5%) | 489 |
| **published at 0.90** | **972** | **782 (80.5%)** | **190** |

On one scene, end to end: `darkvessel run --config configs/anholt-structures.yaml` returns 47
detections over the farm and the register explains all 47. That config exists because
`configs/embeddings.yaml` runs over the shipping lane, where the register excludes nothing — a
stage demonstrated only where it has no effect has not been demonstrated. It has no AIS behind it,
so its detections are `unsearched` rather than `dark`, and the printed line says `unsearched`
rather than borrowing the stronger word.

**Cost.** A fourth value in the `status` column, a column on every layer this chain writes
including the ones that exclude nothing, and a reference file per archive box that has to be
refetched when a farm is built. The exclusion is a file rather than a rule inferred at run time,
which means one acquisition can never produce one: only the archive can, which is correct — a
single scene cannot tell a mast from a ship that happens to be there — and it means a new study
area needs an archive before it needs this.

## 2026-08-27 — The contextual layers are sampled by a command of their own, and a missing value is missing

Issue #15 asks for four variables at every detection — distance to shore, water depth, EEZ
membership, fishing effort — sampled server-side and attached to the written output. Three
decisions inside that, and the third is the only one that could have gone wrong quietly.

### Where the sampling happens in the chain

Not inside `pipeline.run`. Every other optional stage of this project is a parameter of that
function — the detector, the embedder, the register — and this one is not, because it is the only
one that needs a credentialed network connection at the moment the chain executes. The whole
argument for the injected detector is that the pipeline runs on a laptop with nothing behind it;
`darkvessel synthesise && darkvessel run` is the first thing the README asks a reader to do, and a
`run` that sometimes reaches for Earth Engine is a `run` that has to be explained before it is
demonstrated.

So `darkvessel context` is a command beside `export`, `ais`, `scenes` and `known` — the four that
already need a network — and it reads the layer the run wrote, samples, and writes it back. Not a
second file beside it: a detection and its context are one row, and two files joined on a row
order nobody stated is exactly the sidecar `embed/embedder.py` refuses.

Writing back over its own input is what made `write_detections` atomic. It was an unlink
followed by a write, which is harmless while the only caller is `run` — a lost output is
regenerated by running the chain again. Here the file being replaced is the file that was read, so
a write that failed halfway through would leave nothing to re-read and nothing to fall back on,
and the way back would be a scene, a 330 MB checkpoint and the whole pipeline. It now goes through
the same `checkpoints.atomically` the weights do.

What `run` still owes the columns is that they exist. `pipeline.run` writes all four empty on
every layer, including the synthetic demo, for the reason `without_a_register` writes an empty
`structure_distance_m`: a layer whose attribute table depends on which stages were switched on
cannot be stacked with the one beside it.

### The sources are config keys, not constants

`data/gee_export.py` hard-codes `COPERNICUS/S1_GRD`, and that is defensible — the project is about
Sentinel-1 and a different collection would be a different project. These four are not like that.
The bathymetry is whatever bathymetry is available, the EEZ boundaries are not in the public
catalogue at all, and no fishing-effort product covers the years these scenes were acquired. Those
are facts about somebody else's catalogue, they move, and a hard-coded identifier is a claim this
repository cannot keep true. They live in `configs/kattegat-lane.yaml`, where they can be
corrected without touching code, and `context_request_from` checks what can be checked without
credentials — a scale, a search radius, a window that runs forwards — for the same reason
`export_request_from` exists.

Two of them are worth naming here because they are compromises rather than choices:

- **Distance to shore** is not a published raster. It is `FeatureCollection.distance` over LSIB's
  simplified land polygons — Earth Engine computes metres to the nearest feature on its own side.
  Country-scale polygons are coarse for a fjord and adequate for open water 10 km off Skagen.
  Beyond the search radius the image is masked, so *further than this was not measured* arrives as
  a missing value rather than as the radius.
- **Fishing effort** is Global Fishing Watch's daily hours, and that collection ends years before
  these acquisitions. Summed over the window the config names, what it answers is *where fishing
  effort has been recorded*, not *was anyone fishing here that morning* — no public product can
  answer the second for 2026. A detection in a square that has never carried fishing hours is
  still a different object from one in a square that always has, and that is the question level 4
  was asking.

**The EEZ is null in the shipped config**, and that is the honest state rather than an oversight.
Marine Regions publishes the world's EEZ boundaries under CC-BY; Earth Engine's public catalogue
does not carry them, so they have to be ingested once as a table asset and named. Until that is
done every detection reads `unavailable` in the `eez` column — which is precisely what that word
exists for, and is a different statement from `high seas`.

### A value nobody could sample is missing, never zero

This is the criterion the ticket puts last and the only one that fails silently. Every one of
these four variables has a plausible zero: no fishing effort recorded in a square is a real and
useful finding, zero metres from shore is a detection aground, and a depth of zero is the
waterline. An unanswered layer that filled in a zero would be indistinguishable from any of them,
and the analysis level 4 exists to do — where does undeclared traffic concentrate, and under what
conditions — would be reading gaps in somebody's coverage as concentrations of shallow water.

So numbers come back NaN, which a GeoPackage carries as NULL and QGIS shows as empty, and the EEZ
carries two distinct words: `high seas` for a position outside every zone, which is an answer, and
`unavailable` for a layer that gave none. `test_a_value_of_zero_is_kept_and_a_missing_one_is_not
_turned_into_zero` holds it, and it is asserted again after a round trip through the GeoPackage,
because the criterion is about the written output and NaN reaching a driver is where it would be
lost.

The same distinction is what the EEZ join turns on, in the one place a test cannot reach. An
Earth Engine `saveFirst` join drops every primary feature that matched nothing — so without
`outer=True`, a detection on the high seas would not come back at all, and a point missing from
the answer is indistinguishable downstream from a point the layer could not answer for. The one
case the variable exists to identify would have read as `unavailable`.

### What the catalogue said when it was finally asked — 2026-08-27

The section above was written before `darkvessel context` had been run against Earth Engine, and
said so. It has now been run over `configs/kattegat-lane.yaml`. Two of the three sources were
right and one was wrong, which is roughly the hit rate that justifies having put them in a config
file rather than in the code.

**`NOAA/NGDC/ETOPO1` carries `bedrock` and `ice_surface`, as named.** **`USDOS/LSIB_SIMPLE/2017`
holds 312 features** and `FeatureCollection.distance` over it returns metres, as assumed.

**`GFW/GFF/V1/fishing_hours` has no total band, and the config named one.** Earth Engine's refusal
is the correction, quoted rather than paraphrased: `Band pattern 'WLD' did not match any bands.
Available bands: [drifting_longlines, fixed_gear, other_fishing, purse_seines, squid_jigger,
trawlers]`. The collection is two-dimensional — one image per flag state per day, one band per
gear type — so the variable is two sums: `.sum()` over the images the window selects, then
`Reducer.sum()` over the six gears. The window itself was a guess that survived: the collection's
images run **2012-01-01 to 2016-12-31**, so 2016 is the last full year in it, and **15 004 images**
fall inside that window. That is why this one layer takes about a hundred seconds to sample where
the other two take one.

**One band is unmasked and the other two are not**, which looks like the thing this level exists
to refuse and is the opposite of it. A mask means something different in each product. An
unmeasured depth is water nobody surveyed and an unmeasured distance is beyond the search radius,
so both stay missing. GFW's grid covers the ocean, and a masked cell in it is a cell where no
fishing hours were recorded — an answer, and the answer this variable will most often have. Left
masked it would arrive as NaN and be indistinguishable from a run with no effort source at all,
which is the confusion the rest of this section is about. So `unmask(0)` on that band and no
other, and a source left null is still the only way `fishing_hours` comes back empty.

### What it returned

Six detections on the Kattegat lane scene, all six answered by all three available layers:

| status | MMSI | length | distance to shore | depth | fishing hours 2016 |
| --- | --- | --- | --- | --- | --- |
| matched | 636026410 | 274 m | 21.3 km | −35 m | 22.7 |
| **dark** | — | — | 27.0 km | −35 m | 57.9 |
| matched | 255805577 | 140 m | 29.3 km | −42 m | 40.8 |
| matched | 219025245 | 24 m | 31.2 km | −42 m | 58.3 |
| matched | 538002621 | 228 m | 27.4 km | −33 m | 39.2 |
| matched | 667002360 | 244 m | 28.2 km | −49 m | 41.9 |

The numbers are the right shape for this water, which is the only claim being made about them. The
northern Kattegat is 30 to 50 m deep and these are 33 to 49. Skagen is the nearest land, roughly
half a degree west of the box, and these are 21 to 31 km. The EEZ column reads `unavailable` six
times, as the config says it must until the boundaries are ingested.

Two of the depths repeat, and the repetition is the resolution showing through: ETOPO1's cell is
about 1.85 km, the six detections span roughly 10 km, and a bathymetry that gave six distinct
values here would be telling us something it does not know. That is the limitation the config
comment claims, visible in the output rather than only asserted.

**The dark detection sits in the second-highest fishing-effort cell of the six.** That is written
down because it is the kind of number this level exists to produce and the kind that is
meaningless at n = 1. Six detections on one scene support no distribution, and the difference
between 57.9 and 39.2 hours over a year in adjacent 0.01° cells is noise until it is asked of a
few hundred detections. It is a hypothesis the columns now make it possible to test, not a result.

### What is not claimed

**Sampling is not analysis.** The ticket asks for the variables at each detection; it does not ask
where dark vessels concentrate, and this does not answer that. What this level produces is the
column an eventual answer would be computed from, and the table above is one scene's worth of it.

**The EEZ has still not been sampled.** The code path is exercised against a fake and the shipped
config names no asset, so every row reads `unavailable`. Earth Engine's public catalogue carries
no EEZ layer; Marine Regions publishes one under CC-BY and it has to be ingested once. Until then
this criterion is met in code and not in data, and the column says so rather than being absent.

**Sampling is not analysis.** The ticket asks for the variables at each detection; it does not ask
where dark vessels concentrate, and this does not answer that. One scene carries six detections,
of which one is dark, and no distribution is worth reading off that. What this level produces is
the column an eventual answer would be computed from.

**One round trip per scene, not one per detection.** All the detections of a run go across as one
feature collection and come back as one table, and the order they come back in is not trusted: an
index travels with each point and the answers are put back in the order they were asked. A sampler
that returned one row fewer would otherwise put every value on the wrong vessel and still write a
layer that opens.

## 2026-08-29 — The analysis reports a rate, and every interval is resampled over acquisitions

Level 4's last stage, and the first one whose output is prose rather than a column. Four decisions
were made in it, each of which would have produced a plausible README paragraph if made the other
way. Every one is held by a test in `tests/test_concentration.py`, on the standing bar that a
decision nothing would catch being reverted is a defect rather than a finished piece of work.

### The reported quantity is a share, not a count

A histogram of where the 40 dark detections were found is mostly a picture of where detections are
found at all. The shipping lane carries a quarter of everything the archive saw inside a stripe
770 m across, and it would carry a large share of the dark candidates under any hypothesis
including the null one. Dividing by the detections standing in the same water turns "there are
more of them here" into "a larger share of what is here is undeclared", which is the question #16
asks and the only one a detection archive can answer.

It changes the sign of the headline. By count, the lane is the third-busiest band for dark
detections and unremarkable. By share it is 2.1% against an archive-wide 21.2% and it is the only
band on the page that separates from anything.

### The interval is a bootstrap over scenes, not a Wilson interval over rows

**The decision the whole analysis turns on.** 189 detections came from 49 acquisitions, and two
detections of one acquisition share a sea state, a pass direction, a morning, and frequently a
hull that stood in the box again a week later. Treating them as 189 independent Bernoulli trials
is the default every textbook interval assumes and it is wrong here in the direction that
manufactures findings: the archive-wide interval is [15.9%, 27.5%] under independence and
[13.6%, 29.4%] once whole acquisitions are resampled, about a third wider, and a third is enough
to decide most of the band-to-band comparisons.

So `interval_over` draws scenes with replacement and each drawn scene brings all of its
detections. Both numbers are printed, the row-wise one beside the scene-wise one, so that the cost
of the easier assumption is visible rather than argued about.

`test_the_interval_widens_when_the_scenes_disagree` is the guard, and it is constructed to fail on
the revert rather than to pass on the current code: twelve acquisitions, six entirely dark and six
entirely declared, eight detections each. Resampled by row the interval is about ±0.10; resampled
by scene it is half the unit interval. The test asserts more than a factor of two between them.
`test_the_scenes_are_resampled_whole` holds the other half of it — with every scene internally
unanimous, only rates on the twelfths are reachable, so bounds landing on that grid is the
evidence that acquisitions and not rows were drawn.

Wilson rather than the normal approximation for the row-wise figure, because the bands here run to
one dark detection in 47 and a symmetric interval prints a negative probability there.

### Bands are quartiles of the population, and the population is every detection

Cut on the dark subset, the bands follow wherever the dark detections happen to sit and the rate
per band tends to flat by construction — the analysis would then have been incapable of finding
anything, and would have reported that as a null result. Cut on all 189, each band asks a fixed
question of one slice of water.

Quantiles rather than equal widths, so no interval is wide merely because its band was empty. It
also turned out to be where the finding lives: the bands hold equal counts, so their *widths* are
the measurement, and the lane is visible as a quartile 770 m across beside one 5.44 km across.

Duplicate edges collapse. ETOPO1's cell is 1.85 km and a box this size can return a single depth;
four bands of one value would be three empty bands implying a gradient nobody measured.

### A variable nobody could sample is unavailable, never a null result

Carried straight through from the sampling stage. The EEZ column reads `unavailable` on all 189
rows because Earth Engine's public catalogue has no such layer, and an analysis that binned it
would report that dark candidates are spread evenly across EEZs — a sentence that is not false so
much as about nothing. `Category.available` is false when every value is `unavailable`, the
report says so, and #16's EEZ criterion is recorded as unmet rather than satisfied in form.

The same rule applies per-row: a detection the catalogue could not answer for is excluded from a
band and counted as unsampled, never folded in as a zero. Zero metres from shore is a detection
aground and zero fishing hours is a real fact about water, so a filled-in zero would be
indistinguishable from a finding.

### No model is fitted, and the comparison is interval overlap

There is no regression of dark rate on depth and no p-value on this page. 189 detections over ten
weeks of one rectangle support four bands, a rate in each, and the question of whether two
intervals overlap. That bar is stricter than a two-sample test at 5%, which is the direction to
err in for a page that will be read as a result — and it is why the visible slope in the depth
estimates, 27.3% deepest against 9.3% shallowest, is reported as nothing found.

### What the run produced, and the confound it cleared

189 detections over 49 of the 50 acquisitions, 40 dark, 21.2% [13.6%, 29.4%]. One band separates
from all three others and it is the declared lane. Depth and recorded fishing effort separate
nothing.

The sea-state columns the archive-wide run put on every row were there against the possibility
that a dark rate is a fact about the wind. Over 49 acquisitions the detection count is flat in the
sea level (Spearman −0.02) and the dark rate weakly negative (−0.20), which 49 points produce by
chance. Recorded rather than corrected for, and it does not appear to be operating.

Two limits of the study area are visible in the answer and are stated in the README rather than
worked around, because neither is fixable by a different analysis of the same box: distance to
shore correlates with longitude at 0.51 and depth at −0.84, so both variables are close to spatial
coordinates of a 17 km rectangle, and a finding about distance from land cannot be separated from
a finding about where the lane runs.

### Three ways the report nearly said "nothing found" about something it never measured

Found reviewing the first cut of this module, and worth recording because all three were correct
arithmetic wrapped in a wrong sentence — the only kind of defect a stage whose output is prose
can have, and invisible to every test that checks a number.

**A band whose detections all came from one acquisition has no interval,** because there is one
morning to resample. The summary line printed `nan%` bounds and then concluded "every interval
overlaps every other; no concentration established", which is the module's own cardinal error
stated by its own summary. It now says no interval could be estimated, and `Band.estimated` and
`Profile.comparable` are what the sentence is chosen on.

**A variable whose column is not on the layer was filtered out of the report entirely.** A layer
that never went through `darkvessel context` carries none of the four, and the answer to "where
do they concentrate against fishing effort" was to not mention fishing effort — which reads as
though the question had been asked. Every measure is now profiled, and an absent column is
unavailable for the same reason a column of nulls is.

**A missing sea-state correlation was always blamed on the acquisition count.** Spearman returns
nothing for two different reasons — fewer than three scenes to rank, or scenes whose sea never
moved, which includes a layer never sampled for it — and the message named only the first. A
reader told there were too few acquisitions would go and fetch more.

Each is held by a test in `TestWhatCouldNotBeMeasured`, and each guard was watched failing on the
revert before it was kept.

## 2026-08-29 — The interval has an interval, and the page states what resolution to read it at

Found by re-reading the previous entry against a measurement, which is the order it should have
happened in. The config's comment beside `draws` claimed the bootstrap percentiles were "stable to
well under the digit the README prints". Measured over twelve seeds on the archive layer, the
bounds move **0.47 and 0.52 points** at the shipped 4000 draws, and the README prints one decimal.
The claim was false in the digit it was about.

**The first fix considered was the wrong one.** Raising the draw count looks like the answer and
is not: the Monte Carlo error of a percentile falls as one over the square root of the draws and
never reaches zero. Measured on the same layer — 0.47 points at 4000, 0.22 at 20 000, 0.12 at
50 000 — and even 50 000 moves nine of the thirteen printed intervals between seeds. Buying the
printed decimal would take a draw count nobody would run, to make real a precision the method does
not have.

**Reproducibility was never what the draw count bought.** The seed is in the config, so
`darkvessel analyse --config configs/kattegat-lane.yaml` returns those figures exactly, and it did
before this entry. What was missing was a statement of how much of the printed precision means
anything.

So the spread is measured rather than asserted. `monte_carlo_spread` re-runs the archive-wide
interval over twelve consecutive seeds and reports the range of each bound; it lands in
`docs/runs/analysis-archive.json` and on the terminal beside the interval, and the README quotes
it from there and tells the reader to read every bound on the page at whole-percent resolution.
The same rule the ladder follows, that a number on the page comes out of a committed run rather
than out of somebody's terminal.

The published figures are unchanged. Rewriting thirteen intervals across a merged pull request and
a closed issue to gain four tenths of a point of Monte Carlo precision, on bounds eight points
wide, would have been churn dressed as rigour — and it would have left the same defect in place,
because the new numbers would have carried an unstated error too.

Guarded by `TestTheMonteCarloErrorOfTheBounds`, including a test that reads the shipped config and
fails if the old claim comes back. Its fixture makes the dark flag a property of the acquisition
rather than of the row, because a fixture without that clumping understates the spread by half and
would pass on a bootstrap that had stopped clustering at all.

---

## 2026-08-29 — The RPN's foreground IoU threshold is a build parameter, and the sixth rung is set from a sweep rather than from a worked example

**Decision.** `rpn_fg_iou_thresh` and `rpn_bg_iou_thresh` are build parameters of
`detector_model`, recorded in the block every checkpoint carries and checked when one is loaded.
The value the sixth rung of issue #24 runs at is **not fixed in this entry**. It is fixed from a
threshold sweep, on a CPU session costing no GPU quota, in an entry of its own written before the
rung trains — the order rung 4 was set in on 2026-08-19, and for the same reason: a number chosen
after the run it justifies is a narration of that run.

**Why this parameter and not another.** The census of 2026-08-19 found that ninety percent of the
3637 training ships never reach torchvision's 0.7 foreground threshold against any anchor, in
either set tried. They are positive only because `allow_low_quality_matches` guarantees every box
its best anchor. Two RPN rungs then ran inside the region that describes — R2 moved the anchor
geometry and lost 0.048, R4 moved the sampler batch and lost 0.0087 inside a band of 0.0099 — and
neither moved the threshold the region is *defined* by, because the five rungs were fixed on
2026-08-17, before the census existed. Issue #11 closed saying so. This is that rung's parameter.

**Two keys, and why the second one is not a second decision.** `Matcher` refuses a background
threshold above the foreground one. Torchvision's background default is 0.3, so a foreground
threshold below 0.3 cannot be reached by moving one key, whatever the ladder's one-line rule
prefers. `rpn_bg_iou_thresh` is exposed for that constraint alone and for no argument of its own:
it separates an ignored anchor from a negative one, and nothing measured here counts either.
`detector_model` refuses the inversion itself, naming both config keys, rather than letting
torchvision's `torch._assert` — which names neither key nor the file they came from — surface on
a machine rented by the hour.

**What the check on a checkpoint is and is not.** Unlike `tile_px`, `anchor_sizes` and `stem`,
this threshold changes nothing about a loaded model's behaviour: `RegionProposalNetwork` consults
its matcher only while training, so a checkpoint fitted at 0.7 and one fitted at 0.1 detect
identically. The refusal in `_check_built` is therefore about **provenance, not behaviour** — it
is what stops a run config naming one training regime while loading the checkpoint of another,
which no precision or recall downstream of it could ever contradict. Silence in a build block
means torchvision's 0.7 and 0.3, because every checkpoint written before this date was fitted
under them; the same allowance `stem` has, for the same reason.

**The sweep, and a prediction about it written before it runs.**
`notebooks/anchor_census.py` now reports, over the same ship-bearing training tiles, every ship's
best overlap with any anchor and — at nine candidate thresholds from 0.7 down to 0.05 — the
positives per tile, the rescue-only share and the realised positive fraction. One pass over the
boxes rather than nine: `box_iou` is the expensive line and it does not depend on the threshold.

The prediction, and it contradicts this log rather than extending it. The entries of 2026-08-19
and 2026-08-25 both explain the rescue rule through a worked example — "a 16 px ship sits inside a
32 px anchor, the overlap is `256/1024 = 0.25` for every anchor containing it" — and 0.25 has
since read as the threshold at which the median ship would stop being rescued. **That number
squares a length.** Every level-0 anchor has an area of 1024 px² whatever its aspect ratio, and a
contained ship's IoU is its own *area* over that, not its longest side squared. A hull is longer
than it is wide, which this project's own aspect-ratio choice says in `model.py`. So the median
ship's best IoU is predicted to come in **well below 0.25** — nearer 0.10 than 0.25 — and 0.25 is
predicted to leave most ships rescued rather than to be the point at which they stop being.

Recorded before the census runs, in the form the census's own prediction of 2026-08-19 was
recorded in and was then found wrong in. If the sweep contradicts this, that is the entry that
gets written.

**Cost.** A parameter that changes nothing at inference now blocks a checkpoint from loading under
a run config that misdescribes it — a refusal for a difference no measurement could reveal, which
is a deliberate trade of convenience for provenance and is stated here rather than discovered by
whoever meets it. And the ladder's one-line rule bends: the sixth rung moves two keys, presented
as one decision that torchvision splits across two parameters, which is a weaker claim than the
five rungs before it made and is written down as one.

Held by `tests/test_rpn_thresholds.py` — the threshold reaching the matcher torchvision actually
labels anchors with, the inversion refused by name, the checkpoint refused by a run naming another
regime, and silence read as torchvision's own — and by the sweep's arithmetic in
`tests/test_anchor_census.py`. Each was proved by making the revert and watching the test fail.

---

## 2026-08-30 — The threshold sweep, and rung 5 set at 0.3 because the rule says one line

**Decision.** `configs/ladder/r5-fg-iou.yaml` runs `rpn_fg_iou_thresh: 0.3`, moving one key and
leaving `rpn_bg_iou_thresh` at torchvision's 0.3. Fixed here, from the sweep below, **before the
rung trains** — the order rung 4's `32` was fixed in on 2026-08-19.

**The sweep, run on a Kaggle CPU session on 2026-08-30.** Same 1123 ship-bearing training tiles,
same 3637 ships, no GPU quota. Its first two blocks reproduce the census of 2026-08-19 exactly —
stock anchors: mean 97.6 positives per tile, max 3098, 3257 rescue-only, `{0: 109506, 1: 121}`,
realised fraction 0.168 — which is the point of re-running it rather than trusting the entry.

Best IoU any anchor offers a ship, over the 3637:

| p5 | p25 | **p50** | p75 | p95 |
| --- | --- | --- | --- | --- |
| 0.028 | 0.079 | **0.207** | 0.458 | 0.772 |

| fg IoU | positives per tile | rescue-only | share | realised fraction |
| --- | --- | --- | --- | --- |
| **0.70** (shipped) | mean 97.6, max 3098 | 3257 / 3637 | 0.896 | 0.168 |
| 0.50 | mean 111.1, max 3098 | 2807 / 3637 | 0.772 | 0.210 |
| 0.40 | mean 141.0, max 3098 | 2589 / 3637 | 0.712 | 0.282 |
| **0.30** (rung 5) | mean 210.0, max 3114 | 2238 / 3637 | 0.615 | 0.375 |
| 0.25 | mean 270.3, max 3147 | 2037 / 3637 | 0.560 | 0.415 |
| 0.20 | mean 361.4, max 3898 | 1772 / 3637 | 0.487 | 0.449 |
| 0.15 | mean 506.5, max 5978 | 1457 / 3637 | 0.401 | 0.477 |
| 0.10 | mean 743.4, max 9455 | 1120 / 3637 | 0.308 | 0.493 |
| 0.05 | mean 1216.3, max 15525 | 616 / 3637 | 0.169 | 0.499 |

**The prediction of 2026-08-29 was half right, and the half that was wrong is the number.** It
said the median ship's best overlap would come in "well below 0.25 — nearer 0.10 than 0.25", on
the ground that `256/1024 = 0.25` squares a length where the overlap is an *area* over an anchor's
1024 px².

The mechanism holds. At 0.25 the rescue-only share is still **0.560**, so the median ship does not
reach 0.25 and the figure two earlier entries quote is not the point at which ships stop being
rescued. Below 0.25, as predicted.

The magnitude is wrong by about a factor of two: **0.207**, which is nearer 0.25 than 0.10. The
reason is stated rather than shrugged at. The prediction reasoned from a hull's aspect ratio, about
2.5 to 1, and a hull's aspect ratio is not its *label's*. LS-SSDD's boxes are axis-aligned and a
ship lies in any direction — which is the argument `model.py` already makes for keeping the stock
`ASPECT_RATIOS` — so a vessel at forty-five degrees has a nearly square bounding box. A median
best IoU of 0.207 implies a median box of about `0.207 x 1024 = 212` px², roughly 16 by 13 against
a longest side of 16.0. The boxes are almost square, and the prediction applied the geometry of the
ship to the geometry of the rectangle drawn around it.

**Why 0.3 and not 0.2, which is where the median ship actually crosses.** Because
`rpn_bg_iou_thresh` is 0.3 and `Matcher` refuses a background threshold above the foreground one,
so 0.3 is the last value reachable by moving one key — and one line different from R1 is the rule
issue #24 restates and the five rungs before it kept. A sixth rung that quietly moved two keys
would be the first on this ladder to measure two things and report one.

It is not a weak change for being the permitted one. Against the shipped 0.7, dropping to 0.3:

* **1019 more ships gain a genuine match** — rescue-only falls from 3257 to 2238 of 3637, from 90%
  to 62%.
* **the positives the sampler actually draws roughly double**, from `0.168 x 256 = 43` per image to
  `0.375 x 256 = 96`.

Both are larger moves in the RPN's positive set than either rung the ladder already spent a session
on. R2 took the realised fraction *down* to 0.014 and lost 0.048; R4 raised the fraction by
shrinking the batch, which reduced the absolute positives from about 43 to at most 16, and lost
0.0087. Rung 5 is the first that raises the count and improves the quality at once, and the two
had never moved together before.

**The rung's own prediction, written before it trains.** It will **not be a draw**: the statistic
moves by more than R1's band of 0.0099 in absolute value, unlike R3 (0.00001) and R4 (0.0087).
Direction predicted positive. The mechanism is the paragraph above — the RPN currently learns
confidence from anchors forced positive by a tie at the maximum, where a 32 px anchor sits around
a 16 px ship, and at 0.3 it learns from anchors that genuinely overlap.

The risk that would make it wrong, stated now rather than after: an anchor at 0.3 IoU is a poor
localisation target, and the box regression is trained to move it onto the ship from that overlap.
The statistic is decided at a score threshold of 0.75, where precision is what R1 is carrying
(0.848), so a rung that finds more ships and localises them worse can lose on the number it wins
on the mechanism.

**What is deferred, and why it is a rung of its own.** Everything below 0.3 — and 0.2 in
particular, where the rescue-only share crosses half at 0.487 and the median ship finally has a
genuine match. It needs `rpn_bg_iou_thresh` to move with it, which is a second line, and this
ladder's rule has one. Recorded as reasoned and deferred, the same state the pyramid-level trim of
2026-08-25 is in, rather than folded into this rung as a second variable.

**Cost.** The ladder's sixth rung was chosen *after* the first five had run, which is the thing the
rule exists to make suspect. Two things make it admissible and neither is that it seemed like a
good idea: its hypothesis is in this log at 2026-08-19, at the foot of the census entry, three days
before the first rung trained; and its value comes from a measurement rather than from the five
outcomes. `configs/ladder.yaml` says so at the rung itself.

Held by `tests/test_config.py`, which asserts this rung differs from R1 by exactly
`model.rpn_fg_iou_thresh` and resolves to the cosine schedule R1 was kept for, and by
`tests/test_anchor_census.py`, which pins the sweep's arithmetic including the float32 boundary
`docs/failures.md` records for 2026-08-30.

**Outcome, added after the run.** R5 scored F1 0.82282 and was rejected — a loss of 0.0128 against
R1's band of 0.0099. The prediction above went one for two: not a draw, correctly, and the
direction wrong. More usefully, the hypothesis this whole rung was built on does not hold, and the
mechanism runs the other way: `allow_low_quality_matches` was selecting a *better* positive set
than a lowered threshold does, because a tied maximum is a centring criterion and an IoU floor is
not. `docs/failures.md`, 2026-08-30, with the numbers.

## 2026-08-30 — The map is two files, and the basemap is the one with nothing in front of it

Issue #8 asks for a static page showing the detections over a basemap, matched in one colour and
dark candidates in another, and it says why it is built now rather than at the end: once the map
exists, every later improvement to the chain becomes visible instead of living in a metrics
table. `darkvessel map` writes it, from a layer the chain has already produced. No network, no
torch, no credentials, no scene — the same standing as `darkvessel analyse`.

**Two files, written together.** `docs/map/detections.geojson` is the export the ticket asks for,
and QGIS opens it directly. `docs/map/index.html` is the page. The page does *not* fetch the
GeoJSON beside it: opened from a disk — which is how anybody checks a page before publishing it —
the browser's own origin rules refuse that read and the map draws an empty sea over a working
basemap, which looks exactly like a run that found nothing. So the collection is inlined into the
page as well as written beside it, and `test_the_page_and_the_geojson_beside_it_hold_the_same
_detections` is the only thing stopping the two from drifting apart.

**The basemap changed after it was looked at.** The first version drew CARTO's Positron, which is
the quieter backdrop and the better one for a scatter of points over water. Opened in a browser,
every tile came back as a grey square with `API KEY REQUIRED` printed across it. The page loaded,
placed all 189 detections correctly, and was worthless. That is precisely the failure the ticket's
"no backend, no scheduled job, no hosted service" is written against, and it arrived through a
third party rather than through a service of ours — "nothing of mine can go down" is not the same
claim as "nothing here can go dark". The tiles are now OpenStreetMap's own, which need no account
and no key, and `test_the_basemap_needs_no_account_and_no_key` holds it. Leaflet stays on a CDN
and is pinned by version and by subresource hash: a script that comes back altered does not run.

**`unsearched` is a third colour, not folded into either.** `fusion/match.py` keeps that status
apart because a run with no declarations that called its detections dark would publish a sea full
of undeclared vessels. This page is the one output of the chain read by people with no way to open
the layer and check, so it is the last place to lose the distinction. The archive layer carries
none today; the class exists on the page anyway, because the run that produces one will not come
with a reminder to add it.

**The MMSI does not leave the GeoPackage.** A matched detection is a vessel that declared itself
and did everything right, and naming it on a public page adds nothing to the demonstration — the
finding is the detections nobody declared, and those carry no identifier by definition. The column
stays in `outputs/`, where an analyst who needs it has it.

**The export is reprojected, and the measurements are rounded.** GeoJSON is WGS84 by
specification and has no way to say otherwise, so the layer's EPSG:25832 is converted rather than
relabelled; written as it stands, the northern Kattegat lands off the coast of Ghana and the
terminal still reports 189 detections written. A NaN, meanwhile, is not JSON — `json.dumps` emits
the bare literal and no browser will parse the file — so a missing measurement is `null`, and one
dark detection, which has no match distance by definition, was enough to empty the map. Distances
are published to a tenth of a metre and scores to four places: 136.51302541327746 m is a claim
about femtometres made of a detection placed to the nearest 10 m pixel, and a reader of the file
cannot tell a digit that means something from one that fell out of a float.

**The page is committed; the layer it came from is not.** `/outputs/` is ignored, and a page
nobody can reach is a page that does not exist. So `docs/map/` is in the repository and the
GeoPackage behind it is not, which means the two can disagree — regenerate the page and it is
consistent again, and `darkvessel map` prints the same three numbers the page carries so that a
run and its caption cannot quietly differ.

**Everything the ticket asks to be shown is in the HTML as text, not only in a popup.** The
acquisition date, the scene identifier and the match tolerance are on every row of a table
rendered into the file. A reader who never clicks a marker, or arrives with scripting off, or
turns up on a morning a tile server is down, still has the four facts. What Leaflet adds is where
the detections are, which is the one thing the table cannot say.

**Two things the review changed.** The page's script had `['unsearched', 'matched', 'dark']`
written into it as literals, beside a module that imports those three names from
`fusion/match.py` precisely so the vocabulary has one owner. Renaming a status there would have
put every marker into a layer group that does not exist, drawn an empty map, and failed no test
in this repository; the order and the fallback are now handed to the page as data, and
`test_the_statuses_reach_the_script_as_data_rather_than_as_literals` holds it. And Leaflet's
absence is now a sentence where the map would have been rather than a throw at the first
`L.map(...)` — which took the table's own click handlers down with it, so the page lost the part
it could still do because of the part it could not.

**Leaflet stays on a CDN, and that is the one open risk on this page.** It is pinned by version
and by subresource hash, so it cannot be substituted, but it can be absent, and the page is then
a table. Vendoring the two files beside `index.html` would leave the tiles as the only external
dependency; it would also put 160 KB of somebody else's JavaScript in this repository, where it
would never be updated again. Recorded as a choice rather than an oversight.
