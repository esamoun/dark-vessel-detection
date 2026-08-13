# Dark Vessel Detection

**Detecting undeclared vessels by fusing Sentinel-1 SAR imagery with AIS records over Danish waters.**

[![CI](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml)

> **Status — work in progress.** The chain runs end to end today, on a real Sentinel-1 scene, with
> a threshold on bright pixels standing in for the detector: one command in, a georeferenced
> GeoPackage out that opens in QGIS where it should. It tiles a scene larger than one tile and
> reports a target sitting on a tile boundary exactly once. What it finds so far is mostly a wind
> farm — the detector is the placeholder, and that is the point of building it in this order. See
> [Approach](#approach) for what is real and what is not.

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
| **3 — AIS fusion** | AIS positions interpolated to acquisition time, spatio-temporal matching, unmatched detections flagged as dark | interpolation and matching done; awaiting real Danish AIS |
| **4 — Spatial analysis** | Where dark vessels concentrate: distance to shore, bathymetry, EEZ boundaries, fishing effort | planned |

The chain that carries these exists first, deliberately, with a deterministic stand-in where the
detector will go. What runs today: scene in, detector injected at the pipeline boundary, the
scene cut into overlapping tiles and the targets they see reconciled into one list, pixel
coordinates converted to ground coordinates, each declared vessel interpolated along its track to
the moment of acquisition and the detections matched against those positions within a stated
tolerance, GeoPackage out. What is still a placeholder: the detector is a threshold on bright
pixels, and the scene and the AIS slice are synthetic — so dark results from this level are wiring
tests, not findings.

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
| Danish Maritime Authority AIS | Declared vessel positions | open daily archives |
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
  data/       AOI selection, Sentinel-1 export, AIS ingestion, tiling, synthetic inputs
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
darkvessel export --config configs/anholt.yaml
darkvessel run --config configs/anholt.yaml
```

`export` asks Earth Engine for one acquisition over the Anholt wind farm, already clipped to the
area and reprojected into the working CRS, and writes a single GeoTIFF carrying its acquisition
time, scene id, polarisations and orbit pass. Clipping and reprojection happen on Google's
machines, and no GRD product reaches the local disk: a single response is two orders of magnitude
smaller than a whole product, and an area that would ask for one is refused before the request is
sent. The shipped area, about 15 km square, came back as 1582 x 1498 px in VV and VH — 33 MB, and
sixteen tiles at 512/64 with real seams between them rather than the four the synthetic scene has.

That run has no AIS to match against; real Danish declarations arrive with Level 3. Its
detections come back marked `unsearched` rather than `dark`, because nothing was searched:

```
115 detections in EPSG:25832 -> outputs/anholt.gpkg
  no AIS supplied: nothing was searched, so no detection here is a dark vessel
```

#### What the first real run showed — checked on a basemap, 2026-08-13

Scene `S1C_IW_GRDH_1SDV_20260702T170036_…`, acquired 2026-07-02 17:00:36 UTC, ascending, VV+VH,
1582 x 1498 px over the Anholt area. `outputs/anholt.gpkg` was opened in QGIS over an
OpenStreetMap basemap: the detections fall at sea east of Grenaa, in the water the config asks
for, with no offset visible against the coastline.

They are not vessels. Nearest-neighbour distances are bimodal with nothing at all between 100 m
and 500 m: 93 of the 115 sit within 100 m of another — one bright object reported twice by a
threshold that splits it — and merging those leaves **60 objects spaced 680 m apart** (p10 554,
p90 957). Ships do not arrange themselves on a 680 m lattice. That is the Anholt wind farm, whose
turbines are bright point scatterers, and it is the false-positive problem this project has to
solve rather than a fault in the chain. Separating fixed structures from vessels is Level 3.

Two things this run caught, both recorded in [`docs/failures.md`](docs/failures.md): the chain was
reading the product's nodata fill as the brightest targets in the scene, and the export's size
guard had been sized from an assumed dtype rather than a measured one.

`outputs/detections.gpkg` opens directly in QGIS, in EPSG:25832. Each detection carries its
`status` (`matched` or `dark`), the `mmsi` that explains it if one does, the distance to that
declared position, and the `tolerance_m` the decision was made at — the radius is part of the
result, because "dark" means nothing without it. A match also carries what the position it
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
