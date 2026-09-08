"""The two failure modes `docs/evaluation.md` had evidenced on one frame, measured over 49.

The report names five failure modes. Three of them are measured over the held-out split and stay
there, because they are properties of the detector rather than of this water. The other two were
single observations on the Kattegat scene of 2026-08-09 — one hull reported many times, and a
moving ship not being where it declared itself — and the report says so about itself, in the
sentence admitting that fifty acquisitions were sitting in the repository while its failure modes
rested on one of them. This module is what closes that sentence.

**Cause 4 is answered by a counterfactual, not by a description.** "The azimuth correction
matters" is not a measurement; "without it this archive would have reported 129 dark candidates
instead of 40, and 89 of them would have been transponder-on ships" is. So the matching stage is
run twice over every acquisition — once with the orbit geometry the chain uses, once with
`geometry=None`, which is the switch `fusion/match.py` already carries — and the two verdicts are
compared row by row. Re-running `classify` is the only honest way to do it: assignment is
confidence-ordered and one-to-one, so removing the correction can in principle hand a declaration
to a different detection, and recomputing distances on the published rows would miss that.

**The recomputation checks itself against what was published.** The corrected pass has to return
the layer that is on disk — the same status and the same MMSI on all 189 rows — and
`reproduces_published` is that check. Without it the counterfactual would be a number from a
second, unverified implementation of the run, which is worth nothing. With it, the "without" column
is known to differ from the published layer by exactly one thing: the correction.

**Cause 3 is answered by a separation, and the answer is a clean negative.** A hull reported twice
must show as two detections closer together than the hull is long, so the smallest separation
between any two detections of one acquisition, against the longest hull the archive estimates, is
the whole test. It needs no re-run and no scene, only the layer.

No network, no torch and no credentials. The counterfactual needs each acquisition's moment and
pass direction, which come off the products the archive already holds, and its declarations, which
are the slices `archive-ais` wrote.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.match import MATCHED, classify


@dataclass(frozen=True)
class Acquisition:
    """One acquisition, reduced to what the counterfactual needs and nothing else.

    Assembled by the command from a product and a slice, and by the tests from two frames, which
    is the point of it being a dataclass rather than a path: the measurement below never opens a
    file, so it can be exercised on four detections in a millisecond.

    `detections` carries the published `status` and `mmsi` alongside the geometry and the score,
    because those two columns are what the corrected pass is checked against.
    """

    scene: str
    acquired_at: datetime
    detections: gpd.GeoDataFrame
    ais: gpd.GeoDataFrame | None
    geometry: Geometry | None


@dataclass(frozen=True, kw_only=True)
class Correction:
    """What the azimuth correction is worth over the archive: failure mode 4.

    `recovered` is the number this exists to report — detections the correction turns from dark
    into matched. `lost` is its price, and a non-zero one would be a finding rather than a bug:
    the assignment is one-to-one, so moving the declarations can in principle take a match away.
    """

    detections: int
    scenes: int
    matched: int
    matched_uncorrected: int
    recovered: int
    lost: int
    matched_either_way: int
    dark_either_way: int
    # Whether the corrected pass returned the layer that is on disk, status and MMSI, on every
    # row. False makes every other number here a number about some other run.
    reproduces_published: bool
    scenes_with_recovery: int
    vessels_recovered: int
    shift_median_m: float
    shift_max_m: float

    @property
    def dark(self) -> int:
        return self.detections - self.matched

    @property
    def dark_uncorrected(self) -> int:
        return self.detections - self.matched_uncorrected

    @property
    def rate(self) -> float:
        return self.dark / self.detections

    @property
    def rate_uncorrected(self) -> float:
        return self.dark_uncorrected / self.detections

    def lines(self) -> list[str]:
        return [
            f"cause 4, the azimuth correction, over {self.scenes} acquisitions",
            f"  {self.matched} of {self.detections} detections matched a declaration; "
            f"without the correction, {self.matched_uncorrected}",
            f"  {self.recovered} recovered, {self.lost} lost, "
            f"{self.matched_either_way} matched either way, "
            f"{self.dark_either_way} dark either way",
            f"  dark rate {self.rate:.1%} against {self.rate_uncorrected:.1%} uncorrected",
            f"  the recovered are {self.vessels_recovered} distinct vessels over "
            f"{self.scenes_with_recovery} acquisitions, displaced "
            f"{self.shift_median_m:.0f} m median and {self.shift_max_m:.0f} m at most",
            "  the corrected pass reproduces the published layer"
            if self.reproduces_published
            else "  THE CORRECTED PASS DOES NOT REPRODUCE THE PUBLISHED LAYER",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "detections": self.detections,
            "scenes": self.scenes,
            "matched": self.matched,
            "matched_uncorrected": self.matched_uncorrected,
            "recovered": self.recovered,
            "lost": self.lost,
            "matched_either_way": self.matched_either_way,
            "dark_either_way": self.dark_either_way,
            "dark": self.dark,
            "dark_uncorrected": self.dark_uncorrected,
            "rate": self.rate,
            "rate_uncorrected": self.rate_uncorrected,
            "reproduces_published": self.reproduces_published,
            "scenes_with_recovery": self.scenes_with_recovery,
            "vessels_recovered": self.vessels_recovered,
            "shift_median_m": self.shift_median_m,
            "shift_max_m": self.shift_max_m,
        }


@dataclass(frozen=True, kw_only=True)
class Duplication:
    """Whether one hull is reported many times, over the archive: failure mode 3.

    A duplicate is two detections of the same object, so it cannot be further apart than the
    object is long. `closest_pair_m` against `longest_hull_m` is therefore the whole finding, and
    `candidate_duplicates` counts the pairs that fail it — pairs closer together than the longer
    of the two hulls they estimate.

    `declarations_matched_twice` is the other shape the same failure would take: one declaration
    claimed by two detections. It is zero by construction of the assignment, and reported because
    a rule holding by construction is worth a number that would move if the construction changed.
    """

    scenes: int
    detections: int
    scenes_with_pairs: int
    candidate_duplicates: int
    declarations_matched_twice: int
    closest_pair_m: float
    median_closest_pair_m: float
    longest_hull_m: float

    def lines(self) -> list[str]:
        return [
            f"cause 3, one hull reported many times, over {self.scenes} acquisitions",
            f"  {self.scenes_with_pairs} acquisitions carry two detections or more; the closest "
            f"pair in any of them is {self.closest_pair_m:.0f} m apart, "
            f"{self.median_closest_pair_m:.0f} m at the median",
            f"  the longest hull the archive estimates is {self.longest_hull_m:.0f} m, so "
            f"{self.candidate_duplicates} pairs are close enough to be one object",
            f"  {self.declarations_matched_twice} declarations were matched by two detections",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenes": self.scenes,
            "detections": self.detections,
            "scenes_with_pairs": self.scenes_with_pairs,
            "candidate_duplicates": self.candidate_duplicates,
            "declarations_matched_twice": self.declarations_matched_twice,
            "closest_pair_m": self.closest_pair_m,
            "median_closest_pair_m": self.median_closest_pair_m,
            "longest_hull_m": self.longest_hull_m,
        }


@dataclass(frozen=True)
class Failures:
    """Both answers, ready to be printed or written as JSON."""

    correction: Correction
    duplication: Duplication

    def lines(self) -> list[str]:
        return self.duplication.lines() + self.correction.lines()

    def as_dict(self) -> dict[str, Any]:
        return {
            "duplication": self.duplication.as_dict(),
            "correction": self.correction.as_dict(),
        }


def counterfactual(
    acquisitions: list[Acquisition],
    *,
    tolerance_m: float,
    max_gap: timedelta,
) -> Correction:
    """Run the matching twice over every acquisition and compare the two verdicts row by row.

    The second pass differs from the first in one argument, `geometry=None`, and that is the whole
    experiment: the tolerance, the interpolation window and the declarations are the ones the
    published run used, so a difference between the two columns cannot be anything but the
    correction. Widening the tolerance instead would answer a different question — see
    `docs/decisions.md`, 2026-08-16.
    """
    scenes = 0
    detections = 0
    matched = 0
    matched_uncorrected = 0
    recovered = 0
    lost = 0
    matched_either_way = 0
    faithful = True
    recovering_scenes: set[str] = set()
    recovered_vessels: set[str] = set()
    recovered_shifts: list[float] = []

    for acquisition in acquisitions:
        rows = acquisition.detections
        if rows.empty:
            continue
        scenes += 1
        detections += len(rows)
        asked = rows[["score", "geometry"]].reset_index(drop=True)
        settings = {
            "ais": acquisition.ais,
            "acquired_at": acquisition.acquired_at,
            "tolerance_m": tolerance_m,
            "max_gap": max_gap,
        }
        corrected = classify(asked.copy(), geometry=acquisition.geometry, **settings)
        uncorrected = classify(asked.copy(), geometry=None, **settings)

        if not _same_verdict(corrected, rows):
            faithful = False

        was = np.asarray(corrected["status"] == MATCHED)
        now = np.asarray(uncorrected["status"] == MATCHED)
        matched += int(was.sum())
        matched_uncorrected += int(now.sum())
        matched_either_way += int((was & now).sum())
        gained = was & ~now
        recovered += int(gained.sum())
        lost += int((now & ~was).sum())
        if gained.any():
            recovering_scenes.add(acquisition.scene)
            recovered_vessels.update(str(m) for m in corrected.loc[gained, "mmsi"])
            recovered_shifts.extend(
                float(shift) for shift in corrected.loc[gained, "azimuth_shift_m"]
            )

    return Correction(
        detections=detections,
        scenes=scenes,
        matched=matched,
        matched_uncorrected=matched_uncorrected,
        recovered=recovered,
        lost=lost,
        matched_either_way=matched_either_way,
        dark_either_way=detections - matched - matched_uncorrected + matched_either_way,
        reproduces_published=faithful,
        scenes_with_recovery=len(recovering_scenes),
        vessels_recovered=len(recovered_vessels),
        shift_median_m=float(np.median(recovered_shifts)) if recovered_shifts else 0.0,
        shift_max_m=float(np.max(recovered_shifts)) if recovered_shifts else 0.0,
    )


def duplication(detections: gpd.GeoDataFrame) -> Duplication:
    """The closest two detections of one acquisition ever come, against the longest hull.

    Pairs are formed inside an acquisition and never across two, because the same vessel imaged
    twice a week apart is not a duplicate, it is the traffic. The hull length is the estimate the
    detector's box carries; taking the longer of the two is the generous reading, which is the one
    worth taking when the finding is a negative.
    """
    longest = float(np.nanmax(np.asarray(detections["length_m"], dtype=float)))
    candidates = 0
    with_pairs = 0
    closest: list[float] = []

    for _, frame in detections.groupby("scene"):
        rows = frame.reset_index(drop=True)
        if len(rows) < 2:
            continue
        with_pairs += 1
        lengths = np.asarray(rows["length_m"], dtype=float)
        separations = []
        for first, second in combinations(range(len(rows)), 2):
            apart = float(rows.geometry.iloc[first].distance(rows.geometry.iloc[second]))
            separations.append(apart)
            if apart < max(lengths[first], lengths[second]):
                candidates += 1
        closest.append(min(separations))

    matched_rows = detections[detections["mmsi"].notna()]
    twice = int((matched_rows.groupby(["scene", "mmsi"]).size() > 1).sum())

    return Duplication(
        scenes=int(detections["scene"].nunique()),
        detections=len(detections),
        scenes_with_pairs=with_pairs,
        candidate_duplicates=candidates,
        declarations_matched_twice=twice,
        closest_pair_m=float(min(closest)) if closest else float("nan"),
        median_closest_pair_m=float(np.median(closest)) if closest else float("nan"),
        longest_hull_m=longest,
    )


def _same_verdict(recomputed: gpd.GeoDataFrame, published: gpd.GeoDataFrame) -> bool:
    """Did the corrected pass return the layer that is on disk, status and MMSI, on every row?"""
    status = np.asarray(recomputed["status"]) == np.asarray(published["status"])
    mine = recomputed["mmsi"].fillna("").astype(str).to_numpy()
    theirs = published["mmsi"].fillna("").astype(str).to_numpy()
    return bool(status.all() and (mine == theirs).all())


def report(result: Failures) -> str:
    return json.dumps(result.as_dict(), indent=2) + "\n"


def write(result: Failures, *, report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(result))
    return report_path


def failures_request_from(config: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    """What the measurement is asked for, read out of a config file.

    The same shape as the other `*_request_from` functions, and here for the same reason: the
    counterfactual reads fifty products and fifty slices before it can fail, and a mistyped key
    should not surface at the end of that.
    """
    failures = config.get("failures", {})
    archive = config["archive"]
    return {
        "scenes": (relative_to / str(archive["scenes"])).resolve(),
        "ais": (relative_to / str(archive["ais"])).resolve(),
        "detections": (relative_to / str(archive["detections"])).resolve(),
        "report": (relative_to / str(failures.get("report", "failures.json"))).resolve(),
    }
