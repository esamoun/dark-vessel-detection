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
