# Related work

Dark vessel detection is an established problem with an active literature and operational
commercial services. This file records what exists, what was reused, and what was built here,
so that the boundary between the two is never in doubt.

Every entry below was read. An unread citation is worse than no citation.

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
Earth. Its published split is scenes 01–10 for training and 11–15 for test. 6015 ship instances in
all — the total LS-SSDD publishes, and the number this project counted for itself across its own
ingested split. At 11.52 gigapixels it is the largest SAR vessel-detection set other than
xView3-SAR, by that paper's own comparison table.

**Relationship to this project: reused.** It is the training set for the detector, used with its
own split so that results here can be put beside the paper's baselines. Read as far as its
construction, its split and its layout: enough to use it correctly and to say why it was chosen
over HRSID, SSDD and xView3-SAR; not yet its baseline results, which are worth reading once this
project has numbers of its own to compare.

---

## Paolo et al. (2022), xView3-SAR

Paolo, F., Lin, T. T., Gupta, R., Goodman, B., Patel, N., Kuster, D., Kroodsma, D., Dunnmon, J.,
"xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery", NeurIPS
2022. [Paper](https://arxiv.org/abs/2206.00897)

**What it establishes.** That this exact task is a public benchmark, and how large the labelled
version of it gets. 991 full-size Sentinel-1 acquisitions averaging 29,400 x 24,400 px, 243,018
verified maritime objects over 43.2 million km², split 554 train / 50 validation / 150 public
test / 237 holdout. Labels come from CFAR, a probabilistic AIS-to-SAR matcher and human analysts,
and the paper says which is which: 39.1% of labels carry both automated and manual annotation,
33.4% manual only, 27.5% automated only. 1,421.81 gigapixels, against 11.52 for LS-SSDD-v1.0.

**Relationship to this project: rejected**, for the reason recorded under 2026-08-14 in
[`decisions.md`](decisions.md) — scoping a subset of it is a piece of work in its own right, and it
stays in reserve for the level where the detector is the bottleneck rather than the chain. It is
kept here for two things it says about this project rather than about itself. It is the public
challenge the README claims exists. And its labelling section states that its probabilistic
matcher "performs significantly better than conventional approaches such as interpolation based on
speed and course" — which is exactly the matcher this chain uses. That method is the entry below.

---

## Kroodsma et al. (2022), probabilistic AIS-to-SAR matching

Kroodsma, D. A., Hochberg, T., Davis, P. B., Paolo, F. S., Joo, R., Wong, B. A., "Revealing the
global longline fleet with satellite radar", *Scientific Reports* 12, 21004.
[Paper](https://doi.org/10.1038/s41598-022-23688-7)

**What it establishes.** The method that supersedes interpolation, and by how much. Rather than
projecting a track to the acquisition instant and matching inside a radius, it mines a year of
global AIS — roughly 10 billion positions — into probability rasters of where a vessel of a given
class, speed and elapsed time is likely to be: 10 km by 10 km, 1296 of them, six vessel classes by
thirty-six time intervals by six speeds. The raster from before the image is multiplied by the one
after it and renormalised. Every candidate pair is then scored by the probability at the detection,
adjusted for how likely that vessel was to be detected at all and for whether its declared length
agrees with the length read off the image, and matches are assigned iteratively from the resulting
score matrix. The paper reports this "far outperformed conventional approaches such as simple
interpolation based on the vessel's speed and course", and that the score supplies a criterion for
accepting or rejecting a match, which a radius does not. Its imagery and detections came from a
commercial provider, Kongsberg Satellite Services.

**Relationship to this project: background, and the honest name for this chain's ceiling.** This
chain interpolates each track to the acquisition instant and matches inside a fixed 200 m: the
approach this paper measures itself against and beats. Two consequences of that are already in the
output rather than hidden. The tolerance is labelled provisional because the azimuth displacement
measured on the first real scene is larger than the radius. And a match carries `position_basis`
and `position_age_s` because a radius on its own cannot say how much of the match to believe. The
raster approach needs a year of global AIS to fit and is out of reach here; predicting each
vessel's own azimuth shift from its declared course and speed is the version of it this project
can reach, and it is the level named at the end of [`running.md`](running.md).

---

## Raney (1971), moving targets in synthetic-aperture imagery

Raney, R. K., "Synthetic Aperture Imaging Radar and Moving Targets", *IEEE Transactions on
Aerospace and Electronic Systems* AES-7(3), 499–505.
[Paper](https://doi.org/10.1109/TAES.1971.310292)

**What it establishes.** The two things a moving target does to a synthetic-aperture image: radial
velocity displaces it along the azimuth direction, along-track velocity defocuses it. The
displacement is the ratio of slant range to platform velocity, times the radial velocity.

**Relationship to this project: background, read after the measurement rather than before it.** The
first real scene on the lane returned four dark detections standing 341–632 m from declared vessels
of 140 m or more. Every displacement pointed north or south whatever the ship's course, its sign
set by whether the ship was closing on the sensor or opening from it; the vessel making no way was
not displaced, the one making 2.6 knots by 116 m, the four making twelve knots by half a kilometre.
The implied `R / V` is about 115 s, which is Sentinel-1's. The table is in
[`running.md`](running.md); this is the reference that names what it found. It is also why the
tolerance stays at 200 m rather than being widened to 600 m to make the problem go away: that would
match those four for the wrong reason, and hand every genuinely undeclared vessel a 600 m radius in
which to find an excuse.

---

## Paolo et al. (2024), the size of the gap

Paolo, F. S. et al., "Satellite mapping reveals extensive industrial activity at sea", *Nature*
625(7993), 85–91. [Paper](https://doi.org/10.1038/s41586-023-06825-8)

**What it establishes.** Two petabytes of Sentinel-1 and Sentinel-2 over 2017–2021, against 53
billion AIS positions, put a number on the gap: 72–76% of the world's industrial fishing vessels
are not publicly tracked, against 21–30% of transport and energy vessels. It uses "dark vessels"
for the same thing this project does.

Its usable half is the calibration. Detection rate above 70% at 25 m and above 90% at 50 m or more,
most objects under 15 m missed, and the IW GRD product's resolution put at about 20 m: that is the
outside view of why every row here carries the declared `length_m` rather than a bare status. It
also removes azimuth ambiguities geometrically, taking every detection within 200 m of the azimuth
line through another and testing the azimuth-angle difference against the off-nadir angle. The
eight rows this project's placeholder detector returns for a single 274 m vessel are a different
mechanism — sidelobes, not pulse-repetition aliasing — but the same failure, and they are handled
here by clustering within 200 m rather than by geometry.

**Relationship to this project: background.** Nothing is reused from it.
