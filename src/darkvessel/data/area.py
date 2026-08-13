"""The study area, described once.

Two stages of the chain cut something to the same rectangle: the export asks Earth Engine for a
scene clipped to it, and the ingestion filters a day of Danish AIS down to it. They read the same
`area.bounds` out of the same config file, and sharing the type is what stops the two readings of
it from drifting apart — an AIS slice filtered to a rectangle the scene was not cut to produces
dark vessels along whichever edge the two disagree on.
"""

from dataclasses import dataclass

from pyproj import Geod

# Distances on the ellipsoid the coordinates here are expressed on, rather than on a sphere or on
# degrees taken as metres. Shared with the ingestion, which measures a report's distance from its
# own track against it: two modules with their own ellipsoid are two answers to the same question.
WGS84 = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Bounds:
    """An area of interest, in WGS84 degrees, as the catalogue and the archive both expect it."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError(
                f"bounds must run west to east and south to north, got {self.as_rectangle()}"
            )
        if not (-180 <= self.west and self.east <= 180 and -90 <= self.south and self.north <= 90):
            raise ValueError(f"bounds are outside the WGS84 range: {self.as_rectangle()}")

    def as_rectangle(self) -> list[float]:
        """The order Earth Engine's `ee.Geometry.Rectangle` takes."""
        return [self.west, self.south, self.east, self.north]

    def grown_by(self, margin_m: float) -> "Bounds":
        """The same area with `margin_m` of slack on every side.

        The AIS filter runs on this rather than on the area itself. A vessel imaged just inside
        the edge of the scene reported, minutes earlier, from just outside it, and cutting that
        report away leaves the vessel with nothing to interpolate between — the failure this
        level exists to remove, reintroduced by the filter meant to feed it.

        Longitude is grown from whichever edge lies nearer the pole, where a degree is shortest
        and a metre is therefore worth most: grown from the other, the margin would come up short
        along one edge of the rectangle.
        """
        if margin_m < 0:
            raise ValueError(f"a margin cannot be negative, got {margin_m}")
        if margin_m == 0:
            return self

        nearest_the_pole = max((self.south, self.north), key=abs)
        west, _, _ = WGS84.fwd(self.west, nearest_the_pole, 270, margin_m)
        east, _, _ = WGS84.fwd(self.east, nearest_the_pole, 90, margin_m)
        _, south, _ = WGS84.fwd(self.west, self.south, 180, margin_m)
        _, north, _ = WGS84.fwd(self.west, self.north, 0, margin_m)

        return Bounds(
            west=max(west, -180.0),
            south=max(south, -90.0),
            east=min(east, 180.0),
            north=min(north, 90.0),
        )
