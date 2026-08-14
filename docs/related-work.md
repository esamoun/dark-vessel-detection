# Related work

Dark vessel detection is an established problem with an active literature and operational
commercial services. This file records what exists, what was reused, and what was built here —
so that the boundary between the two is never in doubt.

Populate as sources are actually read. An unread citation is worse than no citation.

## Format

For each entry: reference, what it establishes, and its relationship to this project
(reused / adapted / rejected / background).

---

## Zhang et al. (2020), LS-SSDD-v1.0

Zhang, T. et al., "LS-SSDD-v1.0: A Deep Learning Dataset Dedicated to Small Ship Detection from
Large-Scale Sentinel-1 SAR Images", *Remote Sensing* 12(18), 2997.
[Dataset](https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN) ·
[Paper](https://doi.org/10.3390/rs12182997)

**What it establishes.** A labelled set built for the case this project is in rather than for the
case that is easy to label: small ships under large-scale backgrounds, on Sentinel-1 IW at 10 m,
with abundant pure-background tiles kept in rather than filtered out. 15 large VV acquisitions,
cut into 9000 sub-images of 800 x 800, ground truth drawn by SAR experts against AIS and Google
Earth. Its published split is scenes 01–10 for training and 11–15 for test.

**Relationship to this project: reused.** It is the training set for the detector, used with its
own split so that results here can be put beside the paper's baselines. Read as far as its
construction, its split and its layout — enough to use it correctly and to say why it was chosen
over HRSID, SSDD and xView3-SAR; not yet its baseline results, which are worth reading once this
project has numbers of its own to compare.
