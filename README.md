# Dark Vessel Detection

**Detecting undeclared vessels by fusing Sentinel-1 SAR imagery with AIS records over Danish waters.**

[![CI](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/esamoun/dark-vessel-detection/actions/workflows/ci.yml)

> **Status — work in progress.** The chain runs end to end today, on a synthetic scene, with a
> threshold on bright pixels standing in for the detector: one command in, a georeferenced
> GeoPackage of matched and dark detections out. It tiles a scene larger than one tile and
> reports a vessel sitting on a tile boundary exactly once. Nothing inside it is good yet — that
> is the point of building it in this order. See [Approach](#approach) for what is real and what
> is a placeholder.

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
| **2 — Full-scene chain** | Inference over an entire Sentinel-1 scene: overlapping tiles, cross-tile deduplication, georeferenced GeoPackage output | tiling and output done; awaiting a real scene |
| **3 — AIS fusion** | AIS positions interpolated to acquisition time, spatio-temporal matching, unmatched detections flagged as dark | planned |
| **4 — Spatial analysis** | Where dark vessels concentrate: distance to shore, bathymetry, EEZ boundaries, fishing effort | planned |

The chain that carries these exists first, deliberately, with a deterministic stand-in where the
detector will go. What runs today: scene in, detector injected at the pipeline boundary, the
scene cut into overlapping tiles and the targets they see reconciled into one list, pixel
coordinates converted to ground coordinates, detections matched against declared AIS positions
within a stated tolerance, GeoPackage out. What is still a placeholder: the detector is a
threshold on bright pixels, the scene is synthetic, and AIS matching uses the nearest report in
time rather than a position interpolated to the moment of acquisition — which means dark results
from this level are wiring tests, not findings.

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
4 detections in EPSG:25832 -> outputs/detections.gpkg
  3 matched, 1 dark at a tolerance of 200 m
```

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
machines: no GRD product reaches the local disk, and the 32 MB a direct download returns is two
orders of magnitude smaller than one. About 15 km square at 10 m is 1485 x 1497 px — sixteen
tiles with real seams between them, rather than the four the synthetic scene has.

That run has no AIS to match against; real Danish declarations arrive with Level 3. Its
detections come back marked `unsearched` rather than `dark`, because nothing was searched:

```
… detections in EPSG:25832 -> outputs/anholt.gpkg
  no AIS supplied: nothing was searched, so no detection here is a dark vessel
```

`outputs/detections.gpkg` opens directly in QGIS, in EPSG:25832. Each detection carries its
`status` (`matched` or `dark`), the `mmsi` that explains it if one does, the distance to that
declared position, and the `tolerance_m` the decision was made at — the radius is part of the
result, because "dark" means nothing without it.

The run is defined by the config file. `configs/pipeline.yaml` names the scene, the AIS slice,
the output, the tile size and overlap to run the detector at, and which detector to inject; the
pipeline itself never knows which detector it got. One of the four synthetic targets stands
exactly where the tiles that config cuts the scene into meet, so the shipped run crosses a seam
rather than only the tests.

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
