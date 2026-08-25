# Evaluation — what the detector finds, what it does not, and what has never been asked of it

**2026-08-25. R1, epoch 12** — the configuration the ladder of issue #11 kept and the weights
`configs/kattegat-lane.yaml` loads. Scored over the whole of LS-SSDD's held-out split: scenes 11
to 15, 3000 sub-images, 2378 ships, empty tiles included.

Everything below is derived from `docs/runs/r1-cosine.json`, which is committed, by
`darkvessel evaluate --config configs/ladder.yaml`. Nothing here needs a GPU, a network or a
checkpoint to check, and the last section says plainly which questions that leaves unanswerable.

---

## What is being counted

Three decisions decide what a precision on this page means, and all three make it a more generous
number than an object-detection benchmark would report. They are stated first for that reason.

**A detection is credited by distance, not by overlap.** A detection lands on a ship if its centre
falls within **200 m** of the ship's centre — the same tolerance the AIS fusion downstream applies
to a declared position, read into pixels through the 10 m resolution. Overlap is the wrong
instrument here: a 60 m vessel is six pixels, so a box two pixels out already fails at half IoU,
and the score would be measuring a box no stage of this chain ever uses. The reasoning is in
`metrics.py`; the consequence is that these numbers are not comparable to a published mAP.

**One ship can be found once.** Detections claim ships in order of confidence and a claimed ship
is gone, so a detector that returns one hull four times is credited with one hit and charged three
false alarms — which is what those four would be to anyone sent out to look at them.

**The split is drawn by scene.** LS-SSDD's own published split, scenes 01–10 against 11–15, so
two cuts of one acquisition never straddle it. The training side was cut down to 2246 tiles to fit a
free tier; the held-out side is scored entire, empty tiles included, because empty tiles are
exactly where a false positive happens.

## The curve

![Precision against recall for R1 at epoch 12](figures/precision-recall-r1.svg)

| Score threshold | Precision | Recall | F1 | Found | False | Missed |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.078 (0.041–0.078) | 0.957 (0.954–0.960) | 0.144 | 2275 | 27039 | 103 |
| 0.10 | 0.336 (0.334–0.379) | 0.948 (0.944–0.951) | 0.496 | 2255 | 4458 | 123 |
| 0.25 | 0.529 (0.525–0.594) | 0.928 (0.911–0.928) | 0.674 | 2206 | 1964 | 172 |
| 0.50 | 0.710 (0.699–0.767) | 0.886 (0.855–0.887) | 0.789 | 2107 | 859 | 271 |
| 0.75 | 0.848 (0.840–0.894) | 0.824 (0.770–0.826) | **0.836** | 1959 | 352 | 419 |
| **0.90** | **0.950** (0.943–0.964) | **0.713** (0.637–0.730) | 0.814 | 1695 | 90 | 683 |

Six points, because six thresholds are what the run scored. It is a coarse curve and it is the
whole of one: nothing between 0.75 and 0.90 has been measured, and the chain operates in exactly
that gap's upper end.

**The interval beside each figure is where it went over the run's last four epochs** — epochs that
differ from one another by nothing except where the run was stopped. It is the real range and not
a symmetric error bar: the final epoch usually sits at one end of it. At the threshold the chain
runs, recall is 0.713 and the last four epochs covered 0.637 to 0.730 — 0.637 at epoch 9, 0.730
at epoch 11, 0.713 at the epoch that was kept. The same config, the same seed and the same data
report a recall nine points apart depending on which epoch the session happened to end on.

That band is the single most important number on this page. The run of 2026-08-14 moved precision
at a fixed threshold from 0.28 to 0.80 between two adjacent epochs while its training loss sat
still;
cosine decay, the one change the ladder kept, cut that wander from 0.026 to 0.010 measured on the
best-F1 statistic, and did not remove it. **Nothing here is stable to three decimal places, and no
figure on this page should be quoted to three.**

## Where the chain sits, and what it pays

`configs/kattegat-lane.yaml` runs at **0.90**, which is not where F1 is best. F1 peaks at 0.75, at
0.836; the chain gives up 0.022 of it to buy precision from 0.848 to 0.950 — 262 fewer false
alarms for 264 fewer ships found. That is a deliberate asymmetry and the reason is downstream
rather than statistical: every detection this chain fails to match against AIS is published as a
dark vessel, which is an accusation someone may be sent out on. A miss costs a ship nobody looked
at. A false alarm costs an inspection and a claim about a named vessel. The reasoning, and the
numbers the choice was made on, are in `docs/decisions.md` under 2026-08-16 and 2026-08-25.

The cost of that choice is stated here rather than left implicit: **at 0.90 the detector misses
28.7% of the ships in the split**, and the operating point is also the least stable one on the
curve: across the last four epochs precision moved by 0.021 and recall by 0.093.

## What the misses are made of

The 683 ships missed at 0.90 are not 683 ships the detector failed to see.

At 0.05 the same weights over the same split miss **103**. Every one of the other **580** produced
a detection within 200 m of the ship and was thrown away by the score threshold rather than by the
detector. Put the other way: **the detector's proposals cover 95.7% of the ships in the split, and
the reported recall is 71.3%. Twenty-four points of recall are lost to confidence, not to
detection.**

Two honest deductions from that number before anything is built on it.

- Some of those 580 are luck. At 0.05 the detector raises 27039 false alarms over 3000 tiles —
  9.0 per tile, 0.141 per km², against a tolerance disc of 0.126 km². Of order **twelve** of the
  683 would have a spurious detection land within tolerance by chance alone. Twelve, not 580.
- "The threshold throws them away" is not the same as "lowering the threshold would recover them".
  The curve says what lowering it costs: 0.75 buys 264 of those ships back for 262 extra false
  alarms, and 0.05 buys 580 back for 26949.

What this points at is **score calibration** rather than detection capacity, and it is the second
independent line of evidence pointing at the region
[issue #24](https://github.com/esamoun/dark-vessel-detection/issues/24) describes. The first was
the anchor census.

## Failure modes, by cause

Each one names what it is evidenced by, because two of these are measured over the held-out split
and three are single observations on one scene.

**1. Ships are matched by a rescue rule rather than by fitting anchors.** Over the 1123
ship-bearing training tiles and their 3637 ships, ninety percent of ships never reach the RPN's
0.7 foreground IoU threshold against *any* stock anchor. They become positives only because
`allow_low_quality_matches` guarantees every box its best anchor, and when a 16 px ship sits
inside a 32 px anchor the overlap is `256/1024 = 0.25` for every anchor containing it,
identically, so they tie at the maximum and the rescue rule forces all of them positive at once —
one tile produced 3098 that way. *Evidence: the census of 2026-08-19, `docs/decisions.md`.* This
is the standing explanation for the weak, badly separated confidences the section above measures,
and it is untested: two rungs failed in the region it describes (R2's anchor geometry, −0.048;
R4's sampler batch, −0.009) and neither changed the threshold itself. That is issue #24.

**2. Confidence wanders between epochs that are otherwise identical.** Recall at 0.90 covered
0.637–0.730 over the last four epochs; precision at 0.25 covered 0.525–0.594. *Evidence: the
per-epoch journal, `docs/runs/r1-cosine.json`.* The mechanism recorded on 2026-08-14 is that the
model reaches the neighbourhood of a minimum in about three epochs and then moves its score
calibration rather than its detections; the loss curve says nothing about it, and it was found
only because the held-out split is scored every epoch.

**3. One hull reported many times.** The bright-pixel stand-in reported a 274 m vessel as **eight
detections** — the cross of sidelobes around a large ship is bright enough to threshold as
separate targets — and 16 detections over the Kattegat scene resolved to 6 objects. The trained
detector reported the same six hulls exactly six times. *Evidence: the scene checked by eye on
2026-08-14 and the swap of 2026-08-16, both in the README.* Under the counting rule above, every
duplicate is a false alarm, so this failure mode is already priced into the false-alarm column;
what the scene shows is that the trained detector does not have it.

**4. A moving ship is not where it declared itself.** Four of the six declared vessels in the
Kattegat scene were imaged **341–632 m** from their AIS positions, almost purely north–south,
because a target's along-track velocity displaces it in azimuth. Before the correction the chain
called four declared, transponder-on vessels dark. *Evidence: `docs/failures.md`, 2026-08-16.*
Fixed by moving the declaration to where the radar would have drawn it before matching: 2 matched
became 5. This is a fusion failure rather than a detector failure, and it is in this report
because it is the largest single source of false dark vessels this project has found.

**5. Low-confidence noise, at a scale the statistic does not see.** 27039 false alarms at 0.05
against 90 at 0.90. Both rejected RPN rungs cut it — R2 to 9883, R4 to 14887 — and neither moved
the statistic, which is decided at 0.75. *Evidence: `docs/failures.md`, 2026-08-23 and 2026-08-25.*
It matters here only as a description of the proposal distribution: the detector is generous and
its scores, not its proposals, are what separate.

## Conditions under which this has never been tested

The list is long, and its length is the point.

- **One acquisition.** Every scene-level number in this project comes from a single Sentinel-1
  IW GRD frame over the northern Kattegat, on one date, in one sea state. There is no second scene,
  no second season and no repeat pass.
- **VV only, at both ends.** LS-SSDD is VV throughout and the Kattegat export is VV because Earth
  Engine's 48 MiB limit forced a choice between area and polarisation. Dual-polarisation is
  untested because there is no data for it — `docs/failures.md`, 2026-08-17.
- **No land in the frame, ever.** The study area is open water by construction. Nothing in this
  chain masks land, and no coastline has been put in front of it.
- **No fixed structures in the frame.** The study area was moved off the Anholt wind farm on
  2026-08-14. Offshore turbines are bright point scatterers that a ship detector returns happily;
  separating them is issue #14 and it has not been built, so a dark candidate here is not yet a
  finding about the sea.
- **Calm water only.** High wind roughens the sea surface and raises the clutter floor. No scene
  in a high sea state has been run, and the behaviour is not extrapolable from these numbers.
- **Sparse traffic only.** Six vessels in the frame. Nothing has been run over a port approach,
  an anchorage, or a convoy where hulls are within a tolerance disc of one another — and the
  counting rule, one ship claimed once in confidence order, is exactly the rule most likely to
  behave differently there.
- **A fixed incidence angle.** The azimuth correction assumes **38.5°**, the middle of an IW
  swath, because the product does not carry the angle. 34° gives four matches on this scene, 38.5°
  and 43° give five, 46° gives four again.
- **A stretch fitted on the scene it is reported on.** The window between calibrated decibels and
  the 8-bit amplitude the model was trained on could not be measured end to end; its floor is
  measured, its width was swept from 25 to 60 dB against this scene's own declared vessels, and
  40 dB was taken as the middle of what worked. That is one free parameter tuned on one scene, and
  it is written down as that.
- **Truthful AIS.** A vessel that spoofs its identity or position while transmitting is matched
  and reported as declared. This chain detects silence, not lies.
- **The shipped weights, on the scene.** R1 has never been run over the Kattegat frame. The
  scene-level numbers in causes 3 and 4 above are the 2026-08-14 detector's. See
  `docs/decisions.md`, 2026-08-25.

## What this report cannot say, and what would let it

Every failure mode above is evidenced by a count, a mechanism or a named vessel. **None is
evidenced by a picture of a tile the detector got wrong**, and no analysis here relates a miss to a
property of the ship that was missed — its length, its heading, its distance to another ship, the
scene it came from.

That is not an oversight in the writing; it is what this machine can reach. LS-SSDD is not on this
machine and neither are R1's weights, so no crop of a missed ship or a false alarm can be produced
here, and the 683 misses cannot be broken down by anything.

One session on a rented GPU closes it, and the shape of it is known:

1. Score the held-out split once with R1 at a fine grid of thresholds rather than six, so the
   curve between 0.75 and 0.90 — where the chain lives — stops being an interpolation.
2. For each held-out ship, record the score of the best detection within tolerance, and join that
   against the ship's box size and scene. The 580/103 split above becomes a distribution instead
   of two numbers.
3. Write out the crops: the worst false alarms at 0.90, and the misses at both ends of the size
   distribution.

Until that runs, this report stands as what the committed measurements support, and it is stated
here which of its criteria that leaves partly met.
