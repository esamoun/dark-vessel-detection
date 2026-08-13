# Dark Vessel Detection

**Detecting undeclared vessels by fusing Sentinel-1 SAR imagery with AIS records over Danish waters.**

[![CI](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml)

> **Status — work in progress.** The chain runs end to end today on real data at both ends: a real
> Sentinel-1 scene, and a real day of the Danish Maritime Authority's AIS archive, with a
> threshold on bright pixels standing in for the detector. Three commands in — the scene, the
> declarations, the chain — and a georeferenced GeoPackage out that opens in QGIS where it should. It tiles a scene larger than one tile and
> reports a target sitting on a tile boundary exactly once. What it finds so far is a wind farm,
> and what declared itself in that scene is a pair of sailing yachts no radar at 10 m pixels can
> show — the detector is the placeholder and the study area turns out to be the wrong water, both
> of which are findings rather than faults. See [Approach](#approach) for what is real and what is
> not, and [what the first real fusion run showed](#what-the-first-real-fusion-run-showed--2026-08-13).

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
| **3 — AIS fusion** | AIS positions interpolated to acquisition time, spatio-temporal matching, unmatched detections flagged as dark | runs on real Danish archives; the study area has no radar-visible traffic |
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
Copernicus priority zone, freely available raw AIS, and enough offshore wind farms to make the
false-positive problem real.

## Repository layout

```
src/darkvessel/
  pipeline.py the single seam: scene + AIS + injected detector -> classified detections
  cli.py      the one command; builds the detector and hands it to the pipeline
  data/       study area, Sentinel-1 export, Danish AIS archives and ingestion, tiling, fixtures
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
earthengine authenticate          # once; set your project in configs/anholt.yaml
darkvessel export --config configs/anholt.yaml   # the scene
darkvessel ais    --config configs/anholt.yaml   # what declared itself in it
darkvessel run    --config configs/anholt.yaml   # the chain
```

`export` asks Earth Engine for one acquisition over the Anholt wind farm, already clipped to the
area and reprojected into the working CRS, and writes a single GeoTIFF carrying its acquisition
time, scene id, polarisations and orbit pass. Clipping and reprojection happen on Google's
machines, and no GRD product reaches the local disk: a single response is two orders of magnitude
smaller than a whole product, and an area that would ask for one is refused before the request is
sent. The shipped area, about 15 km square, came back as 1582 x 1498 px in VV and VH — 33 MB, and
sixteen tiles at 512/64 with real seams between them rather than the four the synthetic scene has.

`ais` fetches the Danish Maritime Authority's archive for the day of that acquisition — the
acquisition instant is read off the scene, so the two cannot describe different moments — and
filters it down to the study area and a quarter of an hour either side. The archive for this day
is 662 MB compressed and 3.3 GB of CSV; it is inflated off the network a chunk at a time and
never stored, so what stays on disk is the few hundred reports that survive:

```
declared positions around 2026-06-21T05:32:30+00:00, from anholt.tif
  26366160 position reports read, 54798 of them with no usable position
  415 in the study area and the window, 0 more inside the area with no readable timestamp
  of those, 237 removed by cleaning: 0 not a vessel, 0 with no nine-digit identifier, 235
  duplicated, 2 contradicting another report of the same instant, 0 at a position the rest of
  their own track cannot reach
  178 declared positions kept
```

More than half of what reached the cleaning was the archive repeating itself. Every rule's count
is printed because a slice is a claim about which vessels declared themselves, and it is only as
good as what was thrown away on the way to it.

```
90 detections in EPSG:25832 -> outputs/anholt.gpkg
  0 matched, 90 dark at a tolerance of 200 m, against 5 declared positions
```

#### What the first real fusion run showed — 2026-08-13

Scene `S1A_IW_GRDH_1SDV_20260621T053230_…`, acquired 2026-06-21 05:32:30 UTC, descending, VV+VH,
1582 x 1498 px over the Anholt area, against the Danish archive for that day.

**The detections are the wind farm, again — and they say so twice over.** 39 of the 90 sit within
100 m of another — one bright object split in two by a threshold — and merging those leaves 70
objects with a median spacing of 604 m. Ships do not arrange themselves on a 600 m lattice.
Turbines are bright point scatterers, and separating them from vessels is the detector's problem
rather than the fusion's.

That lattice is also the georeferencing check for this scene, and a stronger one than a basemap.
Anholt Offshore Wind Farm is published at 56.60°N 11.21°E, 111 turbines over a footprint up to
8 km wide. The detections come out centred on 56.63°N **11.2075°E** — 150 m from the published
longitude — and 9.4 km wide, with the latitude pulled north because the scene clips the southern
end of a farm 20 km long. A transform out by anything would not land a 600 m lattice on the
published position of a real structure. (The earlier scene was checked by eye in QGIS over an
OpenStreetMap basemap; that check belongs to that acquisition, and this is what replaces it here.)

**Nothing matched, and the interesting part is why.** Five vessels declared themselves inside the
searched area. Three were in the margin outside the scene, where a declaration is kept so that a
vessel at the edge of the frame keeps the reports either side of it. The other two were inside
the frame, and neither is visible to this chain:

| MMSI | Type | Length | What the radar has there |
| --- | --- | --- | --- |
| 219032944 | Sailing | — | inside a hole in the product; no data within 100 m |
| 244001536 | Sailing | 8 m | peak −12.1 dB within 100 m, against a sea at −16.7 dB and a threshold at 0 dB |

An 8 m glassfibre sailing boat is not a strong scatterer, and at 10 m pixels it is barely one
pixel. No threshold that finds it would leave a scene rather than a speckle map. So the honest
reading of `0 matched, 90 dark` is neither a finding nor a bug: it is a scene with no
radar-visible declared traffic in it.

**Which is a fact about the study area, and the most useful thing this run produced.** Anholt was
chosen for its wind farm, and that put the box in quiet water off the main Kattegat lane. All 30
Sentinel-1 acquisitions over it between 21 June and 28 July 2026 were checked against the
archive: 19 had no declared vessel inside the frame at all, and across the other 11 the largest
vessel ever standing in the scene at the instant it was taken was **15 m** — every one of them a
sailing boat or a pleasure craft. There is no commercial traffic in this box. An area chosen to
make the detector's false-positive problem visible turns out to make the fusion's problem
invisible, and no luckier acquisition exists to be found. Level 3 needs a box on the shipping
lane; that is recorded in [`docs/failures.md`](docs/failures.md) rather than papered over.

Three things earlier real runs caught, all in [`docs/failures.md`](docs/failures.md): the chain
read the product's nodata fill as the brightest targets in the scene, the export's size guard had
been sized from an assumed dtype rather than a measured one, and the first AIS slice ingested was
empty — which the chain correctly, and unreadably, reported as 115 dark vessels.

`outputs/detections.gpkg` opens directly in QGIS, in EPSG:25832. Each detection carries its
`status` (`matched` or `dark`), the `mmsi` that explains it if one does, the distance to that
declared position, the `tolerance_m` the decision was made at, and `declarations_searched` —
how many declared positions that radius was applied to. Both numbers are part of the result,
because "dark" is a claim about a search: without the radius it cannot be read at all, and
without the count a scene where nobody declared themselves is indistinguishable from a scene
full of ships that switched their transponders off. A match also carries what the position it
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
