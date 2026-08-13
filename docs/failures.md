# Failure log

What was tried and did not work, and what it cost. Kept deliberately: the dead ends are part of
the method, and a pipeline whose limits are known is more useful than one whose limits are not.

Each entry: what was attempted, what happened, why, and what was done instead.

---

## 2026-08-13 — `.gitignore` silently deleted a source package

**What happened.** `.gitignore` opened with an unanchored `data/`, meant for downloaded imagery
at the repository root. A gitignore pattern containing no slash matches at *every* level, so it
also matched `src/darkvessel/data/` — the package holding scene loading, AIS ingestion, tiling
and the synthetic inputs. Six source files were never tracked. Nothing complained: the working
copy was complete, the tests passed, and the scaffold commit that introduced the rule looked
clean.

**What it would have cost.** A clone of this repository could not have imported `darkvessel.cli`
at all. Confirmed by checking the staged tree out into an empty directory and running the suite
there — `ModuleNotFoundError: No module named 'darkvessel.data'`.

**Why it survived.** Every check ran against the working directory, which had the files. No
check ran against what was actually committed.

**What was done instead.** The rules are anchored to the repository root — `/data/`, `/outputs/`,
`/checkpoints/` — so they cannot reach into `src/`. The lesson generalises past this bug: a test
suite that only ever runs in the working directory cannot see what the repository is missing.

---

## 2026-08-13 — The chain read the holes in a product as the brightest ships in it

**What happened.** The first real Sentinel-1 scene run end to end returned 126 detections. Twelve
were not vessels. Earth Engine writes masked pixels as a fill value and declares that value as
nodata; this export took 0 for the fill, and 6.2% of the scene was fill. `Scene.from_geotiff` read
the band plainly, so 0 arrived as data — and on a scene in dB, where the sea sits near -14 dB, 0
is brighter than anything afloat. The threshold detector returned three "targets" of 72100, 38955
and 36428 pixels.

**What it would have cost.** Not a crash and not a warning. A plausible count, detections carrying
scores and coordinates like any other, and the largest of them looking in QGIS like an unusually
large vessel rather than a hole. Had it survived to the AIS fusion stage, three enormous
undeclared "vessels" would have been the headline result of the first real run.

**Why it survived until now.** Every scene the chain had ever seen was written by this repository,
and a synthetic scene has no holes. The synthetic fixture is what made the chain testable, and it
is exactly why this class of fault could only appear on the first real product: a fixture cannot
contain a defect nobody thought to put in it.

**What was done instead.** Declared nodata becomes NaN before the image reaches a detector, so a
hole cannot exceed any threshold — see docs/decisions.md for why NaN rather than a masked array.

**What it also confirmed, at no cost.** The blobs were far wider than the 64 px tile overlap and
came back duplicated across tiles, precisely as the ownership scheme's documented precondition
says they must. The scheme held; the input broke the condition it requires. That precondition had
until then been an argument on paper.

**The second thing the same run corrected.** Two numbers in the export had been assumed rather
than measured: Earth Engine returns S1 GRD bands as float64 rather than float32, so the size guard
was estimating every request at half its true size — waving through exactly what it exists to
stop — and the 32 MB cap quoted for a direct download was contradicted by a 33 MB response that
arrived without complaint. Both now come from the measurement rather than from memory.

---

## 2026-08-13 — The first real AIS slice was empty, and the chain reported 115 dark vessels

**What happened.** The ingestion was finished, pointed at the archive for the scene this project
had already exported — 2026-07-02, acquired 17:00:36 — and read all 18 553 230 position reports
of that day in 27 seconds. It kept none. Not one vessel had declared itself inside the study area
in the half hour around the acquisition. `darkvessel run` then did exactly what it is designed to
do: an empty AIS slice is a search that ran and returned nothing, so every detection is honestly
dark, and the run printed `0 matched, 115 dark at a tolerance of 200 m`.

**What it would have cost.** A GeoPackage of 115 dark vessels over the Kattegat, opening in QGIS
looking precisely like the finding this whole project is built to produce. Every part of it is
correct. `unsearched` did not fire, because AIS *was* supplied. The tolerance was in every row,
as designed. Nothing in the layer said that no ship had declared itself there in the first place.

**Why it survived the design.** The distinction the chain already had — no AIS supplied against
an empty AIS slice — was drawn while both sides were hypothetical. The reasoning behind it still
holds: an empty slice really is a search that found nothing. What the reasoning missed is that
the same words describe a quiet sea and a fleet of undeclared ships, and only the count separates
them. A correct claim that reads as its opposite is not a claim anybody can use.

**What was done instead.** `declarations_searched` travels with every classified detection, and
the run adds a line when it is zero. The argument is the same one that puts the tolerance in the
row: *dark* is a claim about a search, and a reader with only the layer cannot check a search
whose size is not in it.

**The second thing the same run found, and the larger one.** The study area has no ships in it.
Anholt was chosen for its wind farm — bright point scatterers that guarantee a false-positive
problem — and that put the box in open water off the main Kattegat lane. A handful of vessels
cross it a day, at no particular hour, so most acquisitions catch nothing that declared itself.

That was checked exhaustively rather than assumed, because the obvious response is to go and find
a luckier scene. All 30 Sentinel-1 acquisitions over the box between 21 June and 28 July 2026
were run against the daily archives: 19 had no declared vessel inside the frame at the instant
they were taken, and across the remaining 11 the largest vessel ever present was 15 m — every one
a sailing boat or a pleasure craft, none of them a target a SAR scene at 10 m pixels can show.
There is no luckier scene. An area chosen to make the detector's problem visible turns out to
make the fusion's problem invisible, and no choice of acquisition fixes it: the fusion level
needs a box on the shipping lane.

---

## 2026-08-13 — The AIS outlier rule removed the evidence along with the noise

**What happened.** The first version judged each report against its neighbours in time: a report
unreachable from both the one before and the one after it is spurious. It reads as obviously
right, and the test written for it — three reports with a 120 km jump in the middle — came back
with nothing left at all.

**Why.** The two good reports are neighbours of the bad one, not of each other. Each is therefore
as unreachable from its only relevant neighbour as the bad report is from it, and the rule cannot
tell which of a disagreeing pair is the wrong one. All three go. The alternative that suggests
itself — walk the track forward, drop whatever the last kept report cannot reach — has the
mirror-image fault: it anchors the whole track on its first position, so a bad report at the start
takes the vessel's entire slice with it.

**What it would have cost.** Vessels silently deleted from the search, in the module whose whole
purpose is to stop declarations from disappearing on their way to the matching. A vessel removed
here is a detection published as a dark vessel.

**What was done instead.** Each report is compared against the median position of its own track in
the window. A median moves for no single report, which is the property the rule needed and neither
of the obvious versions had.
