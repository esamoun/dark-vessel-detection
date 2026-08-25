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

---

## 2026-08-14 — The export size guard was a number nobody had measured, and it let the request through

**What happened.** The study area moved onto the shipping lane and grew by a third. The guard in
`gee_export.py` estimated the request at 48 MB against a 64 MB ceiling and let it go. Earth Engine
took it, and answered: `Total request size (57353670 bytes) must be less than or equal to
50331648 bytes`.

**Why it survived.** The ceiling was reasoned about rather than measured, and the reasoning was
good: the real cap is a server-side detail that would go stale in a comment, so the guard was
calibrated against a request that had worked and set well above it. Every part of that is
defensible and none of it is a measurement of the limit. The guard's whole purpose is to refuse
locally what Earth Engine would refuse after the wait, and the first request that tested it was
the first request to come near the ceiling — which is to say the guard had never been exercised
by anything except tests it agreed with.

**And the estimate was low twice over, in the same direction.** 57 353 670 against an estimate of
48 000 000 is not a rounding difference. Two independent faults, each about an eighth:

- *Nine bytes per sample, not eight.* Earth Engine counts a byte of validity mask alongside each
  float64. The refusal proves it: the scene is 1845 x 1727 px, and 1845 x 1727 x 2 x 18 is
  57 353 670 exactly.
- *Two corners of a rectangle that is not one.* A lat/lon rectangle's edges bow in a projected
  CRS, so its bounding box is wider than the box between two opposite corners. That understated
  the first study area by 6.5% — invisible, because that request was nowhere near the ceiling.

**What it would have cost.** Nothing irreversible: a wait and an error message. The reason it is
recorded is what it says about the class of guard it belongs to. A refusal that has never fired
against the real thing is a claim, not a check, and this one had a test suite agreeing with it.

**What was done instead.** The ceiling is Earth Engine's, quoted from the refusal. The sample size
is nine bytes, derived from the same message. The estimate transforms four corners. A test pins
the guard against all three real observations — the area that came back, the area that was
refused, and the area in one polarisation that came back — so the next change to any of those
three numbers has to stay consistent with what actually happened.

**What was rejected.** A safety margin. It would have worked, and it would have left two wrong
models in the code with a constant on top hiding both of them.

---

## 2026-08-14 — Four declared ships came back as fourteen dark vessels, and every one of them was where the radar put it

**What happened.** The first run over the new study area: 16 detections, `2 matched, 14 dark at a
tolerance of 200 m, against 12 declared positions`. Six bright objects in the scene, six large
vessels declared inside the frame. Two matched. The other four came back dark.

They are not dark. The 14 dark detections belong to four vessels, and every one of them sits
between 341 m and 632 m from a declared vessel of 140 m or more. The offsets are not scattered.
The first two rows below are the two matched detections rather than dark ones, and they are here
because they are the contrast that makes the rest readable:

| MMSI | Length | Speed | Course | Offset | Bearing of the offset |
| --- | --- | --- | --- | --- | --- |
| 538002621 | 228 m | 0.0 kn | — | 41 m | 248° |
| 219025245 | 24 m | 2.6 kn | 285° | 116 m | 001° |
| 255805577 | 140 m | 13.4 kn | 317° | 475 m | 000° |
| 636026410 | 274 m | 12.8 kn | 137° | 480 m | 176° |
| 667002360 | 244 m | 11.6 kn | 316° | 493 m | 353° |
| 636021202 | 233 m | 13.1 kn | 135° | 514 m | 175° |

The displacement is north or south whatever the ship's course, and which of the two depends on
whether the ship is closing on the sensor or opening from it. The vessel making 0.0 knots is not
displaced at all, and the one making 2.6 knots is displaced by 116 m. This is the SAR azimuth
shift: a moving target is imaged displaced along the azimuth direction by `(R / V) * v_radial`,
and the numbers above imply an `R / V` of about 115 s, which is Sentinel-1's.

**Why this is the interesting kind of wrong.** Nothing crashed. The two vessels that matched are
the two that were barely moving, and they matched at 41 m and 116 m — well inside the tolerance,
so the chain looks calibrated. The four that did not are the four doing 12 knots, and a layer of
"dark vessels" over a shipping lane is precisely the finding this project exists to produce. It
would have read as a result.

**Why it could not have been found before.** The old study area had no moving ships in it. Every
declared vessel that ever crossed it was a pleasure craft the radar could not see, so the one
term that dominates the error budget here never entered it. The move onto the lane is what
surfaced this, which is the argument for the move restated as a finding.

**What was done.** Recorded, and nothing else. The tolerance stays at 200 m and stays labelled
provisional. Widening it to 600 m would make these four match and would be wrong twice: it would
match them for the wrong reason, and it would hand every genuinely undeclared vessel a 600 m
radius in which to find an explanation. The answer is to predict the shift from each vessel's own
declared course and speed and compare against the position the radar would have imaged it at —
which is a level of its own, and now has its measurements.

**What this run does establish.** The study area works. Six commercial ships in one frame, wakes
visible on four of them, against an area where the largest vessel ever seen was 15 m.

---

## 2026-08-14 — The same config, run twice, produced two different detectors

**What happened.** The first training run finished 12 epochs on a Kaggle T4 and the numbers were
read off the console. The saved `metrics.json` then disagreed with that console log on every
epoch — at epoch 12, score 0.50, the log said 1903 detections found against 1877 in the file;
epoch 2's loss was 0.1554 on screen and 0.15260 in the file.

**Why the two could not both be one run.** The journal entry and the printed lines come from a
single measurement object inside `_report`: within one run they are the same numbers written
twice, and there is no path where one says 1903 and the other 1877. Two sets of numbers meant two
runs.

**Where the second run came from.** Kaggle's *Save Version* is not a snapshot. It re-executes the
whole notebook in a fresh machine, and the saved artefacts come from that execution — so a
session watched interactively and the version saved from it are two complete runs of the same
code. Worth knowing before reading any Kaggle output as a record of the session that produced it.

**Why they diverged.** The config says one seed names the run. It named the *data*: the subset of
empty tiles, the orientation of each tile, the order they arrive in. It did not name the model.
The detection head is built fresh — two classes where COCO has 91 — from torch's global
generator, which nothing seeded, and the anchor and proposal sampling inside each epoch draws
from it too. Two runs therefore started from two different models and sampled differently
throughout.

**What it would have cost.** Nothing was wrong with either run; both are valid. What was wrong is
that neither could be reproduced, and that the difference between them was recorded nowhere. A
number in the README with no run behind it that can be re-created is a number nobody can check —
including the next ticket, which is supposed to measure its changes against exactly these.

**What was done instead.** `detector_model` seeds before it builds, so the head is a function of
the run's seed. Each epoch seeds from the seed and the epoch number, the same derivation the
augmentation uses, so a session resumed at epoch 7 samples what an uninterrupted run would have
sampled there rather than restarting a stream. A test builds two heads from one seed and requires
them equal, and two heads from different seeds and requires them different.

**What it did not fix, and is a separate finding.** See below.

---

## 2026-08-14 — The detector did not converge, it oscillated, and the loss did not say so

**What happened.** Over 12 epochs the training loss fell from 0.1809 to 0.1357 — 25%, and most of
it in the first three epochs — while precision on the held-out split at a fixed score threshold
of 0.50 went: 0.547, 0.741, 0.746, 0.410, 0.642, 0.844, 0.654, **0.283**, 0.800, 0.629, 0.529,
0.808. Adjacent epochs differ by a factor of three. Epoch 8 collapses to a precision of 0.283
while its recall stays at 0.897 — the model spends that epoch predicting far too much.

**Why it is not noise.** The same shape appeared in both runs of the same configuration, on
different initial weights: epoch 8 is the outlier in both. A defect that survives a change of
seed is a property of the configuration, not of a draw.

**What it is.** `learning_rate: 0.005`, constant, with no decay anywhere in the schedule. The
model reaches the neighbourhood of a minimum within about three epochs and then bounces around it
for nine more, and what moves epoch to epoch is not the quality of the detector but the
calibration of its scores. The loss is nearly flat across all of it, which is why nothing in the
training output gave the game away — it is the held-out split, scored every epoch, that showed it.

**What it cost.** The last epoch is not the best epoch. Epoch 9 scored an F1 of 0.817 at a
threshold of 0.75 against epoch 12's 0.807, and `keep: 2` had already deleted epoch 9's
checkpoint by the time the run finished. The gap is small enough not to matter here, and it is
exactly the cost the "keep the last, not the best" decision was written down to accept.

**What has not been done.** No learning-rate schedule has been added. It changes what the numbers
mean, so it belongs to a run that can be compared against this one rather than to a patch on top
of it — and this baseline exists to be that comparison. Recorded here so that the next run starts
from a diagnosis rather than from a surprise.

---

## 2026-08-16 — The chain calls a declared, transponder-on vessel dark because SAR moves it

**What happened.** The trained detector was swapped into the chain and appeared to miss four of
the six declared vessels standing inside the real scene — hulls of 274, 244, 233 and 140 m,
which are the last things a detector should lose. The threshold baseline missed the same four.
Two detectors failing identically on the largest targets in the image is not a detector problem.

**What is actually there.** Nothing at the declared positions: within 100 m of each, the scene
peaks at −8.7, −11.7, −12.1 and −10.3 dB against a sea of about −18 dB. Six to nine decibels
above the water is clutter, not a hull. Searched wider, every one of the four has an unmistakable
return of 14 to 27 dB — between **420 and 490 m away, and almost purely north–south**. Easting
offsets never exceed 30 m.

**Why.** Azimuth displacement of moving targets. A target with a radial velocity is imaged
displaced along the satellite's track by roughly `R · v_r / V_sat`, and Sentinel-1's near-polar
orbit puts that track almost exactly north–south. Each vessel's velocity was taken from its own
AIS reports around the acquisition, and the ratio of northing offset to easting velocity is
consistent across all six:

| MMSI | Length | v east (m/s) | Offset north | Ratio m/(m/s) |
| --- | --- | --- | --- | --- |
| 636026410 | 274 m | +3.98 | −440 m | −111 |
| 667002360 | 244 m | −4.25 | +420 m | −99 |
| 636021202 | 233 m | +4.40 | −490 m | −111 |
| 538002621 | 228 m | −0.01 | **0 m** | — |
| 255805577 | 140 m | −4.62 | +450 m | −97 |
| 219025245 | 24 m | −1.30 | +120 m | −92 |

The fourth row is the control and it settles the argument: the one vessel with no east–west
velocity is displaced by nothing at all.

**What it costs.** The chain matches at 200 m. The two vessels it matched are exactly the two
whose displacement stayed under that — the stationary 228 m hull and the slow 24 m one. The other
four are declared, their transponders are on, they are plainly visible, and the chain calls them
*dark*. That is this project's central claim being manufactured by geometry: any vessel under way
with more than about 2 m/s of east–west velocity will be denounced.

It is systematic rather than occasional, and it scales the wrong way — the traffic this study
area was chosen for is exactly the traffic that moves. The 14 dark detections the threshold
baseline reported on this scene rest on it too.

**What was done.** Corrected on 2026-08-16 — see docs/decisions.md. Originally recorded as: nothing yet, deliberately. The fix belongs to the fusion stage, not to the
detector, and the ticket that swapped the detector in is not allowed to modify another stage. It
is recorded here with the measurements, and carried into a ticket of its own. What the swap
ticket reports instead is the detector scored against the positions the radar actually shows,
which is a statement about the detector and not about the matching.

---

## 2026-08-17 — The dual-polarisation stem has no data, on either side of the chain

**What was asked for.** Issue #11's first acceptance criterion is an input stage adapted to radar
polarisation channels, and `model.py` sharpened it before the work began: "a dual-polarisation
stem trained as one".

**Why it is not here.** There is no second polarisation anywhere in this project. LS-SSDD-v1.0 is
VV, all 9000 sub-images of it. The scene the chain runs on is VV, and `configs/kattegat-lane.yaml`
records why: Earth Engine answers a single download up to 48 MiB, the box came back at 57 MB in VV
and VH, and the area was the one thing that had been measured and argued for, so the polarisation
was what gave way. Building a dual stem now would mean fitting its second channel on a copy of the
first, which is not an adaptation — it is the null adaptation the repository already ships.

**What was done instead.** A single-channel stem: `conv1` takes one channel of radar amplitude
rather than three copies of it, its weights folded down from the pretrained RGB kernels so that
the model agrees with the repeat inside the tile at initialisation. It is not numerically
identical to the repeat everywhere — `conv1` pads with three rings of zeros, and a zero means raw
amplitude under the repeat's normalisation and means nothing under this one, so the two agree
beyond three positions of the border and differ within it. Every parameter outside `conv1` is the
repeat's own, including which layers are trainable, so the rung that introduces this stem measures
what training does with one bank of kernels rather than a different starting point. That is an
input stage adapted to what the data actually is, and it is measured as rung 3 of the ladder.

**What it would take to do the thing that was asked.** A VV+VH export, which means either a
smaller study area or a coarser resolution — both already argued over once — and a
dual-polarisation training set, which means either finding one at Sentinel-1's resolution or
accepting a set whose physics is not this chain's physics. Both are decisions above this ticket's
level. Recorded here so that the day a dual-polarisation export exists, this is a task rather than
a rediscovery.

---

## 2026-08-23 — R2, the small anchors: rejected, exactly where the census said it would be

**What was asked for.** Issue #11's second adaptation: anchors sized for the vessels this data
actually holds. `configs/ladder/r2-anchors.yaml` takes `anchor_sizes` from the stock
`[[32], [64], [128], [256], [512]]` down to `[[4], [8], [16], [32], [64]]`, on the argument that
the smallest stock anchor is a 320 m vessel at 10 m resolution — longer than nearly anything in
the training set.

**What was measured.** Twelve epochs on a T4, one line different from R1, same seed and same
splits. Statistic **F1 0.7877** against a bar of **0.8454** — R1's 0.8356 plus R1's band of
0.0099 — so it is short by **0.0577**, and rejected.

It is not a near miss on a hard bar. R2's best epoch of the twelve *is* its last one, 0.7877, so
no epoch of this run reaches the threshold under any reading. And it lands below **R0's** 0.8074
as well: the small anchors are worse than the stock ones outright, not merely worse than the
cosine schedule that was kept before them.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 0.801 | 0.784 | 0.801 | 0.804 | 0.804 | 0.812 | 0.832 | 0.810 | 0.827 | 0.826 | 0.833 | 0.836 |
| R2 | 0.609 | 0.685 | 0.722 | 0.725 | 0.757 | 0.763 | 0.781 | 0.779 | 0.783 | 0.770 | 0.785 | 0.788 |

At epoch 12 and threshold 0.75, R2 finds 1744 of the 2378 held-out ships with 306 false
detections, against R1's 1959 with 352. It gives up 215 ships to save 46 false alarms.

**The prediction that called it, written first.** `docs/decisions.md`, 2026-08-19, closes the
anchor census with a reservation recorded *before* any rung had run: the small anchors give
twenty-seven times fewer positive anchors than the stock ones and a slightly worse rescue-only
rate, and under both sets almost no ship reaches an IoU of 0.7 — which points at the RPN's
foreground IoU threshold as the binding constraint rather than at the anchor sizes. That entry
says in as many words that if the small anchors do not help, it predicted so beforehand. They did
not, and it did.

The realised positive fraction is the mechanism: 16.8% under the stock anchors, 1.4% under these.
The RPN's sampler fills a batch of 256 with roughly 3.6 positives instead of roughly 43, so the
head has an order of magnitude fewer examples from which to learn confidence.

**A number that looks like success and is not.** R2's final training loss is **0.0441**, against
R1's **0.1174** — under a third, on a detector that is measurably worse. The loss is computed over
a sample the anchor change re-composed: almost all easy negatives. Training losses are therefore
not comparable across rungs that move `anchor_sizes`, and reading this one as progress would have
been the most natural mistake available on the evening the run finished.

**One thing that genuinely favours R2, and is not enough.** At threshold 0.05 it reports 9883
false detections against R1's 27039, for the same recall — F1 0.312 against 0.144. The small
anchors do suppress low-confidence noise. That is not where the operating point lives: every
rung's statistic is decided at 0.75 or 0.90, where R2 loses.

**A caveat recorded rather than argued away.** R2 is still climbing at epoch 12 — 0.770, 0.785,
0.788 over its last three — and its band of 0.0173 is nearly twice R1's, both consistent with a
configuration that has not converged in the twelve epochs it was given. Twelve is part of the
comparison rather than an accident of it, and the gap is 0.058 rather than a thousandth, so this
does not put the verdict in doubt. It does mean the rejection is of *these anchors under this
schedule*, which is the only thing any rung of this ladder ever measures.

**What was done.** `configs/ladder/r3-stem.yaml` is repointed from `r2-anchors.yaml` to
`r1-cosine.yaml`, committed with this entry. The ladder is greedy: R3 now stands on R1, is
measured against R1, and its bar is unchanged at 0.8454 — R2 having been rejected, it moves
neither the standing statistic nor the band.

**What is left standing.** The census's hypothesis — that the RPN's foreground IoU threshold, not
the anchor geometry, is what binds — is now the best available explanation for two runs rather
than one. This ladder does not test it: its five rungs were fixed on 2026-08-17, before the census
existed. It is the first thing a sixth rung should change.

---

## 2026-08-23 — R3, the single-channel stem: rejected as a draw, to five decimal places

**What was asked for.** Issue #11's first acceptance criterion, an input stage adapted to radar
polarisation channels. The dual-polarisation stem it names has no data on either side of the chain
— recorded above, 2026-08-17 — and the single-channel stem was shipped in its place: `conv1` takes
one channel of radar amplitude rather than three copies of it, its weights folded down from the
pretrained RGB kernels.

**What was measured.** Twelve epochs on a T4, R1 plus that one change. Statistic **F1 0.83556**
against R1's **0.83557**. The bar was 0.84543, and the raw difference against R1 is
**−0.000011** — eleven millionths, in the losing direction. Rejected.

The run block confirms the change is the only one: of the 24 fields R1 and R3 record, two differ,
and they are the same value written twice — `built.stem` and `stem`, `repeat` to `single`. Same
seed, same anchors, same cosine schedule, same reporting, same 2246 and 3000 tiles.

At epoch 12 and threshold 0.75, R1 finds 1959 ships with 352 false and R3 finds 1969 with 366.
Final training loss 0.1174 against 0.1170. There is no version of this comparison in which the two
stems are distinguishable.

**Which is what the stem's own design predicted.** `docs/decisions.md` records that the folded
stem agrees with the repeat *inside* the tile at initialisation and differs only over a
three-pixel border, where `conv1` pads with zeros and a zero does not mean the same raw amplitude
under the two normalisations. Every parameter outside `conv1` is the repeat's own, trainability
included. A near-null result was therefore the expected one, and it arrived nearer to null than
anyone would have bet.

**What that settles about criterion 1.** The three copies were not costing anything. The
adaptation asked for was dual-polarisation and is blocked on data; the adaptation shipped in its
place turns out to be, in effect, the null adaptation — measurably identical to the repeat it
replaces. That is a complete answer to the criterion rather than an evasion of it: the honest
report is that this axis has nothing in it at this resolution, on this data, and the measurement
says so at five decimal places.

**Where the two runs do differ, and it does not help.** At threshold 0.05 R3 reports 50439 false
detections against R1's 27039 — nearly double, for one more ship. The single stem is noisier where
the detector is not asked to be confident. R3's band is also wider, 0.0146 against R1's 0.0099.
Neither moves the statistic, which is decided at 0.75.

**A reproducibility datapoint, recorded because this project has almost none.** Epoch 1 of R3 is
0.762 against R1's 0.801, and by epoch 12 they agree to five decimals. The same shape appeared
between R0 and R1, whose epoch 1 differed by 0.316 under conditions that were identical. Two pairs
now say the same thing: this pipeline's early epochs diverge substantially between runs and its
late epochs do not. That does not make 0.000011 a measurement of the noise floor — a change and a
noise can cancel — but it is the second observation pointing the same way, and it is why the
ladder reads its statistic at epoch 12 rather than at the best epoch.

**What was done.** `configs/ladder/r4-sampler.yaml` is repointed from `r3-stem.yaml` to
`r1-cosine.yaml`, committed with this entry. Its old comment named `r2-anchors.yaml` as the
fallback if R3 fell, on the assumption that only one of R2 and R3 could — both did, so the last
kept rung is R1.

The check added with R2's rejection caught this one without being touched:
`test_no_rung_of_the_shipped_ladder_stands_on_one_the_rule_rejected` failed by name the moment
`r3-stem.json` landed in `docs/runs/`, naming r4 and the rung it could no longer stand on. It
derives the rejected set from `judge`, which is why a guard written for one verdict held for the
next.

**What this costs rung 4, and what was deliberately not done about it.** R4's stated reason for
`rpn_batch_size_per_image: 32` is the census's finding that a tile offers 3.6 positive anchors, so
the sampler's ceiling is idle and only the batch size moves the realised fraction — 1.4% at 256,
11.3% at 32. Those counts are measured **under R2's small anchors**, and R2 is not on this branch.
Under the stock anchors R4 now inherits, the census puts the realised positive fraction at 16.8%:
the sampler is not idle, and 32 is not the number that argument produces.

The value is left at 32. It was fixed on 2026-08-17, and choosing it again now, with four runs'
results in hand, is precisely what this ladder's rule exists to forbid. What rung 4 will measure
is a smaller RPN batch under the stock anchors — a real question, and not the question its comment
poses. Recorded here, before the run, so that the gap between the rung's justification and the
rung's actual condition is on the record rather than discovered in its verdict.

---

## 2026-08-23 — R4, the RPN sampler: rejected inside the noise, and the ladder closes

**What was asked for.** Issue #11's third adaptation: the foreground/background imbalance at the
RPN's loss, addressed by shrinking `rpn_batch_size_per_image` from torchvision's 256 to 32.

**What was measured.** Twelve epochs on a T4, R1 plus that one change — verified as one change:
of the 24 fields the two run blocks record, exactly one differs. Statistic **F1 0.82689** against
R1's **0.83557**, so **−0.00869**, against a bar of 0.84543. Rejected.

**How this rejection differs from R2's, and why that matters.** R2 lost 0.048, five times R1's
noise band, and fell below even R0. R4 loses **0.0087 against a band of 0.0099** — the change is
*smaller than the noise it had to beat*. The honest statement is not "the smaller sampler batch
hurts"; it is that R4 and R1 are indistinguishable on this data, and the rule keeps a rung only
for beating the noise rather than matching it. Recorded as such, because "rejected" covers both
outcomes in the table and they are not the same finding.

At epoch 12 and threshold 0.75, R4 finds 1913 ships with 336 false against R1's 1959 with 352 —
46 fewer ships for 16 fewer false alarms.

**What the smaller batch demonstrably did do.** R4's band is **0.0190**, the widest of any rung on
the kept branch and nearly double R1's. Sixteen positives and sixteen negatives per image is a
noisier gradient than the 43-odd positives a batch of 256 draws under these anchors, and the
epoch-to-epoch score shows it: 0.792, 0.795, 0.798, 0.779, 0.808, 0.804, 0.817, 0.797, 0.826,
0.808, 0.823, 0.827. The training loss moves the same way and for the same reason — 0.2511 at
epoch 1 against R1's 0.1813, the highest first epoch of the five runs, because a balanced sample
is a harder sample. As with R2 in the opposite direction, that loss is not comparable across a
rung that changes the sampling, and reading it as failure would be the mirror of reading R2's
0.0374 as success.

**One consistent side-effect across both rejected RPN rungs.** At threshold 0.05, R4 reports 14887
false detections against R1's 27039, and R2 reported 9883. Both changes that reduce the RPN's
negative-heaviness cut low-confidence noise, and neither moves the statistic, which is decided at
0.75. That is now two observations of the same shape rather than one.

**The caveat this run carried before it started.** Recorded with R3's rejection and repeated here:
R4's stated justification — 3.6 positive anchors per tile, an idle sampler ceiling, and 32 as the
batch that lifts the realised fraction to 11.3% — is measured under R2's small anchors, and R2 was
rejected, so R4 inherited the stock anchors instead. Under those the census puts the realised
fraction at 16.8%, the ceiling is not idle, and 32 is not the number that argument produces. The
value was left at 32 because it was fixed on 2026-08-17 and rechoosing it with four runs in hand
is what the ladder's rule forbids. So what this rung measured is a smaller RPN batch under the
stock anchors — a real question, answered, and not the question its config's comment poses. The
question the comment poses has not been tested by anything.

**Nothing is repointed.** R4 is the last rung; no config extends it.

**What the five runs say together.** One change of five was kept, and it was the one that was not
among the ticket's three adaptations: cosine decay of the learning rate, +0.028 over the baseline
and a noise band cut from 0.026 to 0.010. The three domain adaptations produced, in order, a
clear harm (−0.048), a draw to five decimal places (−0.000011), and a draw inside the noise
(−0.0087). On this data, at this resolution, the schedule was the only lever that moved.

The standing explanation for the two RPN rungs remains the census's, written on 2026-08-19 before
any of them ran: almost no ship reaches an IoU of 0.7 against any anchor in either set, which
points at the RPN's foreground IoU threshold rather than at anchor geometry or sampler batch size.
Two rungs have now failed in the region that hypothesis describes, and neither tested it. It is
the first thing a sixth rung should change, and this ladder deliberately does not have one.
