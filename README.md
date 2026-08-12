# Dark Vessel Detection

**Detecting undeclared vessels by fusing Sentinel-1 SAR imagery with AIS records over Danish waters.**

> **Status — work in progress.** Level 2 in development. See [Roadmap](#roadmap).

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
| **2 — Full-scene chain** | Inference over an entire Sentinel-1 scene: overlapping tiles, cross-tile deduplication, georeferenced GeoPackage output | planned |
| **3 — AIS fusion** | AIS positions interpolated to acquisition time, spatio-temporal matching, unmatched detections flagged as dark | planned |
| **4 — Spatial analysis** | Where dark vessels concentrate: distance to shore, bathymetry, EEZ boundaries, fishing effort | planned |

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
  data/       AOI selection, Sentinel-1 export, AIS ingestion, tiling
  detect/     dataset, model, training, full-scene inference, pixel->geo
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
pip install -e .
```

## Licence

MIT — see [LICENSE](LICENSE).
