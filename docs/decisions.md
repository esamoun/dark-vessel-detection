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
