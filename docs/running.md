# Running the chain

The synthetic run, which needs no credentials and no weights, is in
[README.md](../README.md#running-the-chain). This file is everything past it: a real
Sentinel-1 scene, the embedding level, the whole archive, and the map.

## On a real Sentinel-1 scene

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
darkvessel failures --config configs/kattegat-lane.yaml     # the report's failure modes, over 49
darkvessel map     --config configs/kattegat-lane.yaml      # the static page, and the GeoJSON
```

`zones`, `analyse`, `failures` and `map` are the four of the eight that need neither credentials
nor a network: everything they read is already on the row, or in the products the archive holds.
`analyse` writes [`docs/runs/analysis-archive.json`](runs/analysis-archive.json) and one figure per
variable, so every number in the section below is re-derivable by anyone holding the GeoPackage;
`failures` writes [`docs/runs/failures-archive.json`](runs/failures-archive.json), which is where
the two archive-wide failure modes in [`docs/evaluation.md`](evaluation.md) come from — it re-runs
the matching stage twice over every acquisition, once with the azimuth correction and once without,
and takes about eight seconds; `map` writes the page in [`docs/map/`](map/), described at the foot
of this file.

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
area was chosen wrongly; the argument is in [`docs/decisions.md`](decisions.md).

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

### What the first run on the lane showed, 2026-08-14

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
[`docs/failures.md`](failures.md) has the full account.

**What happened next, 2026-08-17.** The level was built. `fusion/azimuth.py` displaces each
declared position along the satellite's ground track before matching, and on this same scene the
matches go from 2 to 5 of the 6 hulls in frame. The tolerance stayed at 200 m, which is the whole
point of moving the declaration rather than widening the radius. What the correction approximates
is stated where it is made: the incidence angle is declared at the middle of the swath rather than
read off the product, and it is a fifth of the correction. The reasoning, the sixth vessel no
incidence angle in the swath recovers, and what the export still owes are in
[`docs/decisions.md`](decisions.md) under 2026-08-16. The measurements above stay as they were
recorded: the correction is only readable against them.

What earlier real runs caught is in [`docs/failures.md`](failures.md), one entry each: the
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
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml): the same two commands, not a second
definition of them. Lint runs on Python 3.11; the tests run on 3.11 and 3.13.
