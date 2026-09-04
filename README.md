# Dark Vessel Detection

**Detecting undeclared vessels by fusing Sentinel-1 SAR imagery with AIS records over Danish waters.**

[![CI](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml)

[![189 detections over the northern Kattegat, 40 of them undeclared](docs/figures/map-kattegat.png)](https://esamoun.github.io/dark-vessel-detection/)

**[See the detections on a live map](https://esamoun.github.io/dark-vessel-detection/).** The chain
ran over 50 Sentinel-1 acquisitions of the northern Kattegat, 49 of which carried at least one
detection: 189 detections in all, matched against AIS, 40 of them undeclared. Static page, no
backend.

---

## The problem

Ships are legally required to broadcast their position over AIS (Automatic Identification
System). Some do not: the transponder is switched off, spoofed, or simply absent. These are
*dark vessels*, and they matter: illegal fishing, sanctions evasion, unreported transfers at sea.

Radar sees them anyway. Sentinel-1 acquires C-band SAR regardless of cloud or darkness, and a
metal hull on water is a strong scatterer against a near-black background. Detect every vessel in
the radar scene, match those detections against what AIS declared at the exact moment of
acquisition, and whatever is left over is a vessel that did not announce itself.

## Approach

```mermaid
flowchart LR
  S1["Sentinel-1 GRD<br/>one acquisition"] --> TI["overlapping tiles<br/>one owner per target"]
  TI --> DE["detector<br/>trained on LS-SSDD"]
  DE --> GE["pixel to ground<br/>EPSG:25832"]
  AI["Danish AIS archive<br/>the day of the acquisition"] --> IN["interpolated to<br/>the acquisition instant"]
  GE --> MA{"within the<br/>stated tolerance?"}
  IN --> MA
  MA -->|yes| OUT["GeoPackage, GeoJSON,<br/>the live map"]
  MA -->|no| FX{"stands at a known<br/>fixed structure?"}
  FX -->|yes| EXC["excluded, and the<br/>layer says so"]
  FX -->|no| DK["dark candidate"]
  EXC --> OUT
  DK --> OUT
```

The pipeline is built in four levels, each one shippable on its own.

| Level | What it does | Status |
| --- | --- | --- |
| **1. Detector** | Supervised CNN detector trained on labelled SAR scenes; honest precision/recall and failure analysis | trained, then measured a rung at a time: R1 gives 0.95 precision at 0.73 recall over a held-out split of 3000 sub-images, and the four changes that did not clear the noise are written up rather than removed |
| **2. Full-scene chain** | Inference over an entire Sentinel-1 scene: overlapping tiles, cross-tile deduplication, georeferenced GeoPackage output | runs on a real scene with the trained detector in it, since 2026-08-16 |
| **3. AIS fusion** | AIS positions interpolated to acquisition time, spatio-temporal matching, unmatched detections flagged as dark | **complete.** Runs on real Danish archives over a measured study area, with the azimuth shift of a moving ship compensated before matching. Offshore structures are excluded from the dark count without a single label: 65 fixed positions found by recurrence across 47 acquisitions, each verified against published coordinates to 5.1 m |
| **4. Spatial analysis** | Where dark vessels concentrate: distance to shore, bathymetry, EEZ boundaries, fishing effort | **complete.** 189 detections over 50 acquisitions, 40 of them undeclared: 21.2%, and [13.6%, 29.4%] once the interval is resampled over acquisitions rather than over detections. Of the four contextual variables one separates and three do not, [below](#what-the-archive-shows) |

The chain that carries these was built first, deliberately, with a deterministic stand-in where
the detector would go; the stand-in is still there, behind the same parameter, and is what the
tests and the synthetic run use. What runs today: scene in, detector injected at the pipeline
boundary, the scene cut into overlapping tiles and the targets they see reconciled into one list,
pixel coordinates converted to ground coordinates, each declared vessel interpolated along its
track to the moment of acquisition and the detections matched against those positions within a
stated tolerance, GeoPackage out. Both ends of that are real now: a Sentinel-1 acquisition fetched
clipped from Earth Engine, and a day of the Danish AIS archive streamed, filtered and cleaned
with every removal counted, and the detector between them is the trained one. Detections standing
at a known fixed structure are taken out of the dark count and say so in the layer. What keeps a
dark result from being a finding about the sea is now one thing rather than three: it has run over
one study area.

A vessel moves between its last AIS report and the instant the radar images it: at 12 knots, some
370 m a minute, which is more than the match tolerance. Comparing a detection against a report
taken as it stands therefore manufactures dark vessels that were never there, and that is the
most likely way for this project to produce a confidently wrong answer. Each vessel is placed at
the acquisition timestamp before anything is compared. Where its track gives nothing to
interpolate between, because it ends before the radar looks or the vessel reported once, the nearest
report is used and the row says so, in `position_basis`. Nothing is extrapolated past a track:
prolonging one from a course and speed derived from earlier points would manufacture a position
where no measurement exists.

A vessel on a tile boundary is seen by two tiles and must be reported once. That is done by
ownership rather than by merging detections after the fact: each tile answers for one slice of
the scene and stays quiet about the rest, so the count is right by construction and there is no
merge radius to tune. The reasoning, and the one condition it places on the config, are in
[`docs/decisions.md`](docs/decisions.md).

Two deep learning components sit inside this:

- **Supervised object detection** on SAR. The hard parts are genuinely hard: vessels are a few
  pixels wide at 10 m resolution, the background/foreground imbalance is extreme, pretrained RGB
  backbones have to be adapted to single-channel radar amplitude, and only geometry-preserving
  augmentations are physically valid on SAR.
- **Self-supervised contrastive embeddings** over detection crops. Offshore wind turbines are
  bright point scatterers that look a great deal like ships, and an unsupervised embedding space
  does separate them into distinct clusters without any additional labelling: measurably, at
  0.768 against 0.5 at chance. It turned out not to separate them *well enough to delete a
  detection on*, so the exclusion is built on where a thing stands over ten weeks rather than on
  what it looks like, and the measurement that settled it is
  [below](CHANGELOG.md#telling-a-turbine-from-a-ship-2026-08-27). The embedding earns its place as a
  similarity-search index over the detection archive and as the evidence that the clusters exist.

![Precision against recall for R1 over the held-out split, six thresholds from 0.05 to 0.90](docs/figures/precision-recall-r1.svg)

*The detector, measured rather than asserted. Six points, because six thresholds are what the run
scored. The chain runs at the last of them, 0.90: 0.946 precision at 0.726 recall, giving up 0.016
of F1 against the peak at 0.75 to buy 246 fewer false alarms. What the 651 misses are made of, and
the ten conditions this has never been tested under, are in
[`docs/evaluation.md`](docs/evaluation.md).*

![Six query crops and their four nearest neighbours in the embedding space, with cosine similarities](docs/figures/retrieval-archive.svg)

*The embedding, asked for the nearest neighbours of six detections across ten weeks of
acquisitions. It returns the same object 71% of the time against 0.02% at chance. It does not
separate turbines from ships well enough to delete a detection on, which is why the exclusion is
built on recurrence instead.*

Everything else (AIS interpolation, spatio-temporal matching, contextual analysis) is
geospatial data engineering, not deep learning, and is described as such.

## What the archive shows

The chain ran across all 50 acquisitions of the study area and 189 detections accumulated into one
layer, 40 of them undeclared: 21.2%, and [13.6%, 29.4%] once the interval is resampled over
acquisitions rather than over detections. Four contextual variables were sampled at each
detection. One of them separates; three do not, and are reported as not separating rather than
left out.

| Variable | What the archive shows |
| --- | --- |
| **Distance to shore** | The one that separates. The 770 m band carrying the declared lane holds 61 detections per kilometre against 8.8 in the widest band, and is 2.1% dark [0.0%, 6.5%] where the archive is 21.2% |
| **Water depth** | Every band's interval overlaps every other. Nothing is claimed |
| **Recorded fishing effort** | Every band's interval overlaps every other. Nothing is claimed |
| **EEZ** | 158 detections in Danish water, 31 in Swedish: 19.6% [11.5%, 28.1%] dark against 29.0% [10.0%, 48.6%]. The intervals overlap and nothing is claimed from the difference |

![Share undeclared against distance to shore, four bands with their intervals, against the archive rate of 21.2%](docs/figures/concentration-distance_to_shore_m.svg)

*The bands are equal-count, not equal-width, which is what makes the lane visible: 47 detections
inside 770 m of it against 48 spread over the 5.5 km band beside it. Every number here comes out
of [`docs/runs/analysis-archive.json`](docs/runs/analysis-archive.json), written by `darkvessel
analyse`, so it is re-derivable by anyone holding the GeoPackage.*

## What this does not support

It has run over one study area. Nothing here is evidence about any other water, and no claim of
transfer is made. The detector's precision and recall are measured on a held-out split of LS-SSDD
rather than on the Kattegat, so the numbers above describe a chain whose detector was never scored
on the water it ran over. An unmatched detection is published as a **candidate** at a stated
tolerance, not as an established dark vessel.

[`docs/evaluation.md`](docs/evaluation.md) is the honest account of where the detector breaks and
the ten conditions it has never been asked to work under.


## Data

| Source | Use | Access |
| --- | --- | --- |
| Sentinel-1 GRD | SAR imagery | Copernicus Data Space / Earth Engine `COPERNICUS/S1_GRD` |
| Danish Maritime Authority AIS | Declared vessel positions | open daily archives, `aisdata.ais.dk` |
| LS-SSDD-v1.0 | Detector training | 15 large Sentinel-1 scenes, VV, cut into 9000 labelled sub-images |
| Earth Engine catalogue | Bathymetry, coastline, fishing effort | Google Earth Engine |
| Marine Regions (VLIZ) | EEZ boundaries | Maritime Boundaries Geodatabase v12, CC-BY, fetched per run and not redistributed |

Study area: **Danish waters**, chosen for dense and varied traffic, excellent Sentinel-1 revisit
as a Copernicus priority zone, and freely available raw AIS.

Within them, a 17 km box in the northern Kattegat on the approach to Skagen, and the box is
measured rather than picked. `darkvessel survey` streams a day of Danish AIS and ranks every
rectangle of that size in the Kattegat by how many vessels of 100 m or more, under way, stand
inside it in a given half hour. This one holds five or six at an arbitrary instant and is never
empty. The first study area was chosen off a map for its wind farm, which put it in quiet water
where the largest vessel ever imaged was 15 m; the whole argument, and what the move gives up,
is in [`docs/decisions.md`](docs/decisions.md).

## Repository layout

```
src/darkvessel/
  pipeline.py the single seam: scene + AIS + injected detector -> classified detections
  cli.py      the one command; builds the detector and hands it to the pipeline
  data/       study area and the survey that chose it, Sentinel-1 export, Danish AIS archives
              and ingestion, published offshore-structure coordinates and EEZ boundaries,
              tiling, fixtures
  detect/     detector contract, labelled dataset and augmentations, model, training,
              checkpoints and resume, precision/recall, inference, pixel->geo
  embed/      detection crops, contrastive views and training, the archive they
              accumulate in, nearest-neighbour retrieval and its checks, finding the
              fixed structures the archive holds and verifying them
  fusion/     AIS interpolation to acquisition time, spatio-temporal matching, the
              register of fixed structures a run will not call dark vessels
  context/    contextual variables at each detection: distance to shore, water depth and
              fishing effort sampled from the catalogue, EEZ membership joined locally
  analysis/   the distribution of dark candidates against each of those variables, with
              intervals resampled over acquisitions rather than over detections
  viz/        the GeoJSON export and the static page it is drawn on
configs/      pipeline configuration
data/reference/  published structure coordinates, and the register built from the archive
notebooks/    exploration and Kaggle/Colab training entry points
tests/        unit tests for the geometry-critical paths
docs/         decision log, failure log, and the published page
```

## Related work

This is a known and actively worked problem, not an invented one. The task is the subject of a
public detection challenge on Sentinel-1 with AIS-derived labels, of an established research
literature on SAR ship detection, and of operational commercial services. Prior art is listed
in [`docs/related-work.md`](docs/related-work.md) as it accumulates.

What is specific here is the instance rather than the task: this study area, an AIS ingestion and
interpolation pipeline built from raw national archives, the contextual analysis layer, and
embedding-based disambiguation of vessels from fixed offshore structures.

## Engineering notes

Constraints are stated rather than hidden. Training runs on free-tier cloud GPUs with short,
resumable sessions and checkpointing from the first epoch. The training subset is deliberately
scoped and documented. Where results are modest, they are reported as modest. A detector that
usefully ranks candidates for inspection is a different and more honest claim than a detector
that maps them.

- [`docs/evaluation.md`](docs/evaluation.md): how well the detector works, and where it breaks
- [`docs/decisions.md`](docs/decisions.md): why each choice was made
- [`docs/failures.md`](docs/failures.md): what was tried and did not work

## How this was built

I used Claude Code throughout, as an assistant. It wrote a share of the code and much of the
prose, under review.

What it did not do is decide. The study area, the choice to train a detector rather than
threshold the imagery, the ablation ladder and the decision to reject R5, the tolerance the
matching runs at, and the decision to publish unmatched detections as candidates rather than as
findings, are mine. [`docs/decisions.md`](docs/decisions.md) is the record of that reasoning, and
it is the honest place to judge whether I understand what is here.

## Setup

```bash
conda env create -f environment.yml
conda activate darkvessel
pip install -e ".[dev]"
```

Training and Earth Engine dependencies are extras, `".[detector]"` and `".[gee]"`, and are not
needed to run the pipeline.

## Running the chain

No credentials, no downloads, no weights:

```bash
darkvessel synthesise --out data/synthetic
darkvessel run --config configs/pipeline.yaml
```

```
5 detections in EPSG:25832 -> outputs/detections.gpkg
  4 matched, 1 dark at a tolerance of 200 m
  of those matches, 1 on a position interpolated to the acquisition and 3 on a report taken as it stands
```

One of those five is a vessel under way, 900 m west of its target three minutes before the
acquisition and 600 m east of it two minutes after. Neither report stands within the tolerance;
the interpolated position lands on the target. Matched against a report as it stands, it comes
back as a dark vessel that was never there.

### On a real Sentinel-1 scene

This one needs Earth Engine credentials, and is the only part of the repository that does.

```bash
pip install -e ".[gee]"
earthengine authenticate                              # once; set your project in the config
darkvessel survey --config configs/survey.yaml        # where the traffic is
darkvessel export --config configs/kattegat-lane.yaml # the scene
darkvessel ais    --config configs/kattegat-lane.yaml # what declared itself in it
darkvessel run    --config configs/kattegat-lane.yaml # the chain
```

The embedding level is six more commands over the same water. Only `scenes` needs Earth Engine
credentials and only `known` needs any other network; the last one needs neither, and neither a
GPU nor the framework:

```bash
pip install -e ".[detector]"
darkvessel scenes     --config configs/embeddings.yaml  # ten weeks, two rectangles
darkvessel crops      --config configs/embeddings.yaml  # every detection, cut out
darkvessel embed      --config configs/embeddings.yaml  # fitted, without labels
darkvessel retrieve   --config configs/embeddings.yaml  # what resembles what
darkvessel known      --config configs/embeddings.yaml  # published structure coordinates
darkvessel structures --config configs/embeddings.yaml  # the register, verified
```

Level 4 runs the same chain across every acquisition of the archive rather than one, because a
distribution cannot be read off six detections. `archive-ais` is the long one (21 GB, a few
hours) and it resumes where it stopped:

```bash
darkvessel archive-ais --config configs/kattegat-lane.yaml  # declarations, one day per download
darkvessel archive-run --config configs/kattegat-lane.yaml  # the chain over all 50 acquisitions
darkvessel context --config configs/kattegat-lane.yaml --archive  # the layers, in one round trip
darkvessel eez     --config configs/kattegat-lane.yaml      # the published EEZ boundaries, once
darkvessel zones   --config configs/kattegat-lane.yaml --archive  # whose water each one is in
darkvessel analyse --config configs/kattegat-lane.yaml      # the distribution, and its intervals
darkvessel map     --config configs/kattegat-lane.yaml      # the static page, and the GeoJSON
```

`zones`, `analyse` and `map` are the three of the seven that need neither credentials nor a
network:
everything they read is already on the row. `analyse` writes
[`docs/runs/analysis-archive.json`](docs/runs/analysis-archive.json) and one figure per variable,
so every number in the section below is re-derivable by anyone holding the GeoPackage; `map`
writes the page in [`docs/map/`](docs/map/), described at the foot of this file.

```
4676 crops from 96 scene(s): 318 distinct positions, 65 of them standing in 20+ acquisitions
  kattegat-lane: nothing published in this box, and 0 structure(s) registered from it
  anholt: 65 of 66 published positions carry a registered structure and 65 of 65 registered structures stand at a published position, 5.1 m apart at the median, within 200 m
  published at 0.9: 782 of 972 detections stand at a registered structure (80.5%), leaving 190
```

`survey` is the command that chose the study area, and it needs no credentials, only the AIS
archive. It streams one day of Danish AIS and ranks every rectangle of the study area's size in
the Kattegat by how many vessels of 100 m or more, **under way**, stand inside it during a half
hour (the same half hour `ais` fetches) averaged over every half hour of the day, empty ones
included. Each of those qualifications is load-bearing, and each of them is a way the first study
area was chosen wrongly; the argument is in [`docs/decisions.md`](docs/decisions.md).

```
vessels of 100 m or more, under way, in 0.3 x 0.15 degree rectangles over 2026-08-09
   11.00  57.55  to  11.30  57.70     91 over the day   4.75 in a window  fewest   2     0 windows empty
   11.45  57.25  to  11.75  57.40     88 over the day   4.54 in a window  fewest   2     0 windows empty
   10.95  57.55  to  11.25  57.70     91 over the day   4.50 in a window  fewest   2     0 windows empty
```

`export` asks Earth Engine for one acquisition over that rectangle, already clipped to it and
reprojected into the working CRS, and writes a single GeoTIFF carrying its acquisition time,
scene id, polarisations and orbit pass. Clipping and reprojection happen on Google's machines,
and no GRD product reaches the local disk: a single response is two orders of magnitude smaller
than a whole product, and an area that would ask for one is refused before the request is sent.
The shipped area came back as 1845 x 1727 px in VV: 22 MB, and sixteen tiles at 512/64 with real
seams between them rather than the four the synthetic scene has. VV only is what the larger box
costs; the trade is stated in the config and in the decision log.

`ais` fetches the Danish Maritime Authority's archive for the day of that acquisition (the
acquisition instant is read off the scene, so the two cannot describe different moments) and
filters it down to the study area and a quarter of an hour either side. The archive for this day
is 662 MB compressed and 3.3 GB of CSV; it is inflated off the network a chunk at a time and
never stored, so what stays on disk is the reports that survive:

```
declared positions around 2026-08-09T05:31:24+00:00, from kattegat-lane.tif
  29718190 position reports read, 53320 of them with no usable position
  3687 in the study area and the window, 0 more inside the area with no readable timestamp
  of those, 2135 removed by cleaning: 0 not a vessel, 0 with no nine-digit identifier, 2129
  duplicated, 6 contradicting another report of the same instant, 0 at a position the rest of
  their own track cannot reach
  1552 declared positions kept
```

More than half of what reached the cleaning was the archive repeating itself. Every rule's count
is printed because a slice is a claim about which vessels declared themselves, and it is only as
good as what was thrown away on the way to it.

```
16 detections in EPSG:25832 -> outputs/kattegat-lane.gpkg
  2 matched, 14 dark at a tolerance of 200 m, against 12 declared positions
  of those matches, 2 on a position interpolated to the acquisition and 0 on a report taken as it stands
```

#### What the first run on the lane showed, 2026-08-14

Scene `S1C_IW_GRDH_1SDV_20260809T053124_…`, acquired 2026-08-09 05:31:24 UTC, descending, VV,
1845 x 1727 px over the northern Kattegat, against the Danish archive for that day.

**The study area works.** Six bright objects in the frame, four of them trailing a visible wake.
Twelve vessels declared themselves inside the searched area, ten of them 100 m or longer and the
largest 337 m; six stood inside the study rectangle itself at the acquisition instant, five of
those 100 m or more and the largest 274 m. Against the old box, where the largest vessel ever
imaged over five weeks was a 15 m sailing boat, this is the difference the move was made for.

**Checked by eye, 2026-08-14.** `outputs/kattegat-lane.gpkg` opened over the scene itself, VV
rendered from −25 to 0 dB and the detections drawn as hollow outlines so the pixel under each one
stays visible. Every detection sits on a bright object against a uniform speckled sea at −21.8 dB
median, and there is no land and no fixed structure anywhere in the frame. The 16
detections resolve to **6 objects** when anything within 200 m is treated as one, which matches
the six bright objects visible in the image exactly. The extra rows are one object each: the
larger ships come with a cross of sidelobes bright enough for the threshold to report the arms as
separate targets, so a 274 m vessel arrives as eight detections. That is the placeholder
detector's problem, and it is the same one the wind farm showed at Anholt.

**Two matched, and both of them make sense.** A 228 m vessel making 0.0 knots matched at 41 m,
which is geolocation error and a centroid, and nothing else. A 24 m vessel making 2.6 knots
matched at 116 m.

**The other four are not dark, and finding out why is what this scene was for.** The fourteen
dark detections belong to four vessels, and every one of them stands 341-632 m from a declared
vessel of 140 m or more. The offsets are not scattered. The first two rows below are the two
matches, shown because they are the contrast that makes the pattern readable:

| MMSI | Length | Speed | Course | Offset | Bearing of the offset |
| --- | --- | --- | --- | --- | --- |
| 538002621 | 228 m | 0.0 kn | — | 41 m | 248° |
| 219025245 | 24 m | 2.6 kn | 285° | 116 m | 001° |
| 255805577 | 140 m | 13.4 kn | 317° | 475 m | 000° |
| 636026410 | 274 m | 12.8 kn | 137° | 480 m | 176° |
| 667002360 | 244 m | 11.6 kn | 316° | 493 m | 353° |
| 636021202 | 233 m | 13.1 kn | 135° | 514 m | 175° |

Every displacement points north or south whatever the ship's course, and which of the two depends
on whether the ship is closing on the sensor or opening from it. The vessel making no way is not
displaced; the one making 2.6 knots is displaced by 116 m; the four making twelve knots are
displaced by half a kilometre. This is the SAR azimuth shift: a moving target is imaged
displaced along the azimuth direction by `(R / V) · v_radial`, and the numbers above imply an
`R / V` of about 115 s, which is Sentinel-1's.

So the honest reading of `2 matched, 14 dark` is that the chain is correct, the tolerance is not,
and the term that dominates the error budget is one that could not be measured until there were
moving ships in the frame. The tolerance stays at 200 m and stays labelled provisional: widening
it to 600 m would match these four for the wrong reason and hand every genuinely undeclared vessel
a 600 m radius in which to find an explanation. Predicting the shift from each vessel's own
declared course and speed is a level of its own, and it now has its measurements.
[`docs/failures.md`](docs/failures.md) has the full account.

What earlier real runs caught is in [`docs/failures.md`](docs/failures.md), one entry each: the
chain read the product's nodata fill as the brightest targets in the scene; the export's size
guard was sized from an assumed dtype and then, on a second reading, from a ceiling nobody had
measured, so it let through the very request Earth Engine refused; the AIS outlier rule removed
the evidence along with the noise; and the first AIS slice ingested was empty, which the chain
correctly, and unreadably, reported as 115 dark vessels.

`outputs/detections.gpkg` opens directly in QGIS, in EPSG:25832. Each detection carries its
`status` (`matched` or `dark`), the `mmsi` that explains it if one does, that vessel's declared
`length_m`, the distance to its declared position, the `tolerance_m` the decision was made at,
and `declarations_searched`: how many declared positions that radius was applied to. Both
numbers are part of the result, because "dark" is a claim about a search: without the radius it
cannot be read at all, and without the count a scene where nobody declared themselves is
indistinguishable from a scene full of ships that switched their transponders off. The length is
there for the same reason from the other side: at 10 m pixels a 15 m hull is a pixel and a half,
so a scene of small craft and a scene of cargo are different claims about what the radar could
have seen at all. A match also carries what the position it
matched was built from: `position_basis` is `interpolated` or `reported`, and `position_age_s`
is how far the nearest real report sits from the acquisition. A match against a position
constructed at the acquisition instant and one against a report five minutes old are different
claims, and the row says which it is rather than leaving it to be assumed.

The run is defined by the config file. `configs/pipeline.yaml` names the scene, the AIS slice,
the output, the tile size and overlap to run the detector at, which detector to inject, the match
tolerance and the widest gap in a track a position may be interpolated across; the pipeline itself
never knows which detector it got. One of the five synthetic targets stands exactly where the
tiles that config cuts the scene into meet, so the shipped run crosses a seam rather than only the
tests.

```bash
make test    # the seam: georeferencing, tiling, matching and export, offline and deterministic
make lint
```

The export is tested with Earth Engine faked: the catalogue is a parameter, the same seam that
lets the pipeline run without a detector. What that cannot check is whether Earth Engine's own
filters select what this code believes they select; that is verified by hand on the first real
export and recorded here rather than asserted in a test that could not fail.

Both run on every push and pull request, from
[`.github/workflows/ci.yml`](.github/workflows/ci.yml): the same two commands, not a second
definition of them. Lint runs on Python 3.11; the tests run on 3.11 and 3.13.

## Training the detector

This is the one part of the project that needs a GPU, and it does not run here: the development
machine is an 8 GB M1 laptop, so training happens on a Kaggle free tier where the labelled data
is already attached and never touches the local disk. What is in the repository is the run:
`darkvessel train --config configs/train.yaml`, driven by a config file like every other stage,
with [`notebooks/kaggle-train.ipynb`](notebooks/kaggle-train.ipynb) as a four-cell wrapper that
clones, installs and calls it.

The run-by-run record of the training runs, the numbers each one produced and the changes that did
not clear the noise is in [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT, see [LICENSE](LICENSE).
