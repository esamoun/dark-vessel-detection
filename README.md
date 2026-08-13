# Dark Vessel Detection

**Detecting undeclared vessels by fusing Sentinel-1 SAR imagery with AIS records over Danish waters.**

[![CI](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml)

> **Status — work in progress.** The chain runs end to end today on real data at both ends: a real
> Sentinel-1 scene, and a real day of the Danish Maritime Authority's AIS archive, with a
> threshold on bright pixels standing in for the detector. Three commands in — the scene, the
> declarations, the chain — and a georeferenced GeoPackage out that opens in QGIS where it should.
> It tiles a scene larger than one tile and reports a target sitting on a tile boundary exactly
> once. The study area is now measured rather than picked, and sits on the Kattegat shipping lane:
> the latest run has six commercial ships in one frame, four of them trailing a wake. The detector
> is still the placeholder, and the run surfaced the next real problem — a ship making twelve
> knots is imaged half a kilometre from where it is. See [Approach](#approach) for what is real and
> what is not, and [what the first run on the lane showed](#what-the-first-run-on-the-lane-showed--2026-08-14).

---

## The problem

Ships are legally required to broadcast their position over AIS (Automatic Identification
System). Some do not: the transponder is switched off, spoofed, or simply absent. These are
*dark vessels*, and they matter — illegal fishing, sanctions evasion, unreported transfers at sea.

Radar sees them anyway. Sentinel-1 acquires C-band SAR regardless of cloud or darkness, and a
metal hull on water is a strong scatterer against a near-black background. Detect every vessel in
the radar scene, match those detections against what AIS declared at the exact moment of
acquisition, and whatever is left over is a vessel that did not announce itself.

## Approach

The pipeline is built in four levels, each one shippable on its own.

| Level | What it does | Status |
| --- | --- | --- |
| **1 — Detector** | Supervised CNN detector trained on labelled SAR scenes; honest precision/recall and failure analysis | in progress |
| **2 — Full-scene chain** | Inference over an entire Sentinel-1 scene: overlapping tiles, cross-tile deduplication, georeferenced GeoPackage output | runs on a real scene; awaiting the trained detector |
| **3 — AIS fusion** | AIS positions interpolated to acquisition time, spatio-temporal matching, unmatched detections flagged as dark | runs on real Danish archives over a measured study area; the tolerance does not yet account for the azimuth shift of a moving ship |
| **4 — Spatial analysis** | Where dark vessels concentrate: distance to shore, bathymetry, EEZ boundaries, fishing effort | planned |

The chain that carries these exists first, deliberately, with a deterministic stand-in where the
detector will go. What runs today: scene in, detector injected at the pipeline boundary, the
scene cut into overlapping tiles and the targets they see reconciled into one list, pixel
coordinates converted to ground coordinates, each declared vessel interpolated along its track to
the moment of acquisition and the detections matched against those positions within a stated
tolerance, GeoPackage out. Both ends of that are real now: a Sentinel-1 acquisition fetched
clipped from Earth Engine, and a day of the Danish AIS archive streamed, filtered and cleaned
with every removal counted. What is still a placeholder is the detector — a threshold on bright
pixels — so dark results are not yet findings about the sea.

A vessel moves between its last AIS report and the instant the radar images it: at 12 knots, some
370 m a minute, which is more than the match tolerance. Comparing a detection against a report
taken as it stands therefore manufactures dark vessels that were never there, and that is the
most likely way for this project to produce a confidently wrong answer. Each vessel is placed at
the acquisition timestamp before anything is compared. Where its track gives nothing to
interpolate between — it ends before the radar looks, or the vessel reported once — the nearest
report is used and the row says so, in `position_basis`. Nothing is extrapolated past a track:
prolonging one from a course and speed derived from earlier points would manufacture a position
where no measurement exists.

A vessel on a tile boundary is seen by two tiles and must be reported once. That is done by
ownership rather than by merging detections after the fact: each tile answers for one slice of
the scene and stays quiet about the rest, so the count is right by construction and there is no
merge radius to tune. The reasoning, and the one condition it places on the config, are in
[`docs/decisions.md`](docs/decisions.md).

Two deep learning components sit inside this:

- **Supervised object detection** on SAR. The hard parts are genuinely hard — vessels are a few
  pixels wide at 10 m resolution, the background/foreground imbalance is extreme, pretrained RGB
  backbones have to be adapted to single-channel radar amplitude, and only geometry-preserving
  augmentations are physically valid on SAR.
- **Self-supervised contrastive embeddings** over detection crops. Offshore wind turbines are
  bright point scatterers that look a great deal like ships; an unsupervised embedding space
  separates them into distinct clusters without any additional labelling, and doubles as a
  similarity-search index over the detection archive.

Everything else — AIS interpolation, spatio-temporal matching, contextual analysis — is
geospatial data engineering, not deep learning, and is described as such.

## Data

| Source | Use | Access |
| --- | --- | --- |
| Sentinel-1 GRD | SAR imagery | Copernicus Data Space / Earth Engine `COPERNICUS/S1_GRD` |
| Danish Maritime Authority AIS | Declared vessel positions | open daily archives, `aisdata.ais.dk` |
| Labelled SAR ship datasets | Detector training | public research datasets |
| Earth Engine catalogue | Bathymetry, EEZ, coastline, fishing effort | Google Earth Engine |

Study area: **Danish waters** — dense and varied traffic, excellent Sentinel-1 revisit as a
Copernicus priority zone, and freely available raw AIS.

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
              and ingestion, tiling, fixtures
  detect/     detector contract, dataset, model, training, inference, pixel->geo
  embed/      contrastive representation learning, clustering
  fusion/     AIS interpolation to acquisition time, spatio-temporal matching
  context/    Earth Engine contextual layers
  viz/        GeoJSON export for the web map
configs/      pipeline configuration
notebooks/    exploration and Kaggle/Colab training entry points
tests/        unit tests for the geometry-critical paths
docs/         decision log and failure log
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
scoped and documented. Where results are modest, they are reported as modest — a detector that
usefully ranks candidates for inspection is a different and more honest claim than a detector
that maps them.

- [`docs/decisions.md`](docs/decisions.md) — why each choice was made
- [`docs/failures.md`](docs/failures.md) — what was tried and did not work

## Setup

```bash
conda env create -f environment.yml
conda activate darkvessel
pip install -e ".[dev]"
```

Training and Earth Engine dependencies are extras — `".[detector]"` and `".[gee]"` — and are not
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

`survey` is the command that chose the study area, and it needs no credentials — only the AIS
archive. It streams one day of Danish AIS and ranks every rectangle of the study area's size in
the Kattegat by how many vessels of 100 m or more, **under way**, stand inside it during a half
hour — the same half hour `ais` fetches — averaged over every half hour of the day, empty ones
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
The shipped area came back as 1845 x 1727 px in VV — 22 MB, and sixteen tiles at 512/64 with real
seams between them rather than the four the synthetic scene has. VV only is what the larger box
costs; the trade is stated in the config and in the decision log.

`ais` fetches the Danish Maritime Authority's archive for the day of that acquisition — the
acquisition instant is read off the scene, so the two cannot describe different moments — and
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

#### What the first run on the lane showed — 2026-08-14

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

**Two matched, and both of them make sense.** A 228 m vessel making 0.0 knots matched at 41 m —
which is geolocation error and a centroid, and nothing else. A 24 m vessel making 2.6 knots
matched at 116 m.

**The other four are not dark, and finding out why is what this scene was for.** The fourteen
dark detections belong to four vessels, and every one of them stands 341–632 m from a declared
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
displaced by half a kilometre. This is the SAR azimuth shift — a moving target is imaged
displaced along the azimuth direction by `(R / V) · v_radial` — and the numbers above imply an
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
the evidence along with the noise; and the first AIS slice ingested was empty — which the chain
correctly, and unreadably, reported as 115 dark vessels.

`outputs/detections.gpkg` opens directly in QGIS, in EPSG:25832. Each detection carries its
`status` (`matched` or `dark`), the `mmsi` that explains it if one does, that vessel's declared
`length_m`, the distance to its declared position, the `tolerance_m` the decision was made at,
and `declarations_searched` — how many declared positions that radius was applied to. Both
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

The export is tested with Earth Engine faked — the catalogue is a parameter, the same seam that
lets the pipeline run without a detector. What that cannot check is whether Earth Engine's own
filters select what this code believes they select; that is verified by hand on the first real
export and recorded here rather than asserted in a test that could not fail.

Both run on every push and pull request, from
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) — the same two commands, not a second
definition of them. Lint runs on Python 3.11; the tests run on 3.11 and 3.13.

## Licence

MIT — see [LICENSE](LICENSE).
