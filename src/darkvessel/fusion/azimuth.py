"""Where the radar draws a moving vessel, as against where the vessel is.

Synthetic-aperture radar reads position along the satellite's track out of Doppler shift. A
target that is itself moving toward or away from the sensor adds Doppler of its own, and nothing
in the processing can tell that apart from Doppler caused by position — so the target is drawn
displaced along the track, by an amount proportional to how fast it is closing on the radar.

On the first real scene this chain ran, four declared vessels of 140 to 274 m were imaged 420 to
490 m from where AIS put them, and the chain called all four dark. The measurement, including the
one vessel with no cross-track velocity and no displacement at all, is in docs/failures.md.

Two numbers decide the correction. The direction is the satellite's ground track, which follows
from the orbit's inclination and the latitude, and Sentinel-1's pass direction is recorded on
every product this chain exports. The magnitude is the slant range over the platform speed, and
slant range follows from the incidence angle — which is *not* on the product, and which matters:
across an IW swath it moves the constant from 50 to 90 seconds, and the difference between those
is the difference between a vessel matching and not. It is a config key for that reason, and the
run states it.

Nothing here interpolates a track or matches anything. It converts a velocity into an offset,
and it is a pure function of the geometry so that it can be checked against a scene rather than
fitted to one.
"""

import math
from dataclasses import dataclass

ASCENDING = "ASCENDING"
DESCENDING = "DESCENDING"

# Sentinel-1's orbit. Retrograde by a little over eight degrees, which is what makes a descending
# track lean *west* of due south rather than east of it.
INCLINATION_DEG = 98.18
ALTITUDE_M = 693_000.0
EARTH_RADIUS_M = 6_371_000.0

# The platform's own speed along its orbit, not the speed of the beam over the ground. It is the
# one that appears in the Doppler geometry.
PLATFORM_SPEED_MS = 7590.0

# The middle of a Sentinel-1 IW swath, which spans roughly 29 to 46 degrees. The default, and a
# stated approximation rather than a measurement: the products this chain exports carry no
# incidence angle, so until they do, a run either accepts the middle of the swath or says which
# part of it the scene came from.
IW_MID_SWATH_DEG = 38.5

# What is plausibly an incidence angle at all. Outside this, the number is a look angle, or
# radians, or a mistake — and all three would produce a correction that looks like a correction.
_PLAUSIBLE_INCIDENCE = (20.0, 50.0)


@dataclass(frozen=True)
class Geometry:
    """The orbit geometry that decides where a moving target is drawn.

    `incidence_deg` is the angle the radar meets the ground at, which is what sets the slant
    range. It is separate from the look angle at the satellite, and the two differ by several
    degrees at this altitude — enough that swapping them silently changes the correction by a
    fifth.
    """

    orbit_pass: str
    incidence_deg: float = IW_MID_SWATH_DEG

    def __post_init__(self) -> None:
        if self.orbit_pass not in (ASCENDING, DESCENDING):
            raise ValueError(
                f"{self.orbit_pass!r} is not an orbit pass; it is {ASCENDING} or {DESCENDING}, "
                "as the ORBIT_PASS tag on the product spells them"
            )
        low, high = _PLAUSIBLE_INCIDENCE
        if not low <= self.incidence_deg <= high:
            raise ValueError(
                f"an incidence of {self.incidence_deg} degrees is outside a Sentinel-1 IW swath "
                f"({low} to {high}); this is a look angle, or radians, or a slip"
            )

    def bearing(self, latitude_deg: float) -> float:
        """Which way the ground track runs here, in degrees clockwise from north.

        A function of latitude rather than a constant: a retrograde orbit crosses the meridian at
        an angle that opens up the further from the equator it gets. At the Kattegat's 57.6
        degrees a descending track runs at about 195, which is 15 degrees west of due south.
        """
        crossing = math.cos(math.radians(INCLINATION_DEG)) / math.cos(math.radians(latitude_deg))
        drift = math.degrees(math.asin(crossing))
        return (drift if self.orbit_pass == ASCENDING else 180.0 - drift) % 360.0

    def seconds(self) -> float:
        """Metres of displacement per metre per second of speed toward the radar.

        Slant range over platform speed, times the sine that turns a velocity across the ground
        into a velocity along the line of sight. The units work out to seconds, which is the
        honest name for it: the interval by which the target's Doppler history is offset.
        """
        incidence = math.radians(self.incidence_deg)
        orbit = EARTH_RADIUS_M + ALTITUDE_M
        look = math.asin(math.sin(incidence) * EARTH_RADIUS_M / orbit)
        slant = orbit * math.cos(look) - math.sqrt(
            EARTH_RADIUS_M**2 - (orbit * math.sin(look)) ** 2
        )
        return slant / PLATFORM_SPEED_MS * math.sin(incidence)

    def displacement(
        self, velocity_east: float, velocity_north: float, latitude_deg: float
    ) -> tuple[float, float]:
        """How far along the track this velocity throws a target, as an offset in metres.

        Only the component aimed at the radar counts. Sentinel-1 looks to the right of its track,
        so that component is the velocity projected onto the bearing ninety degrees clockwise
        from the heading; the part running parallel to the track contributes nothing, which is
        the property the tests assert rather than any particular number.

        The sign is the one the Kattegat scene shows, and it is the single place in this module
        where a convention could have been flipped without the arithmetic complaining. Six
        vessels, three of them thrown each way, agree with it.
        """
        bearing = math.radians(self.bearing(latitude_deg))
        along = (math.sin(bearing), math.cos(bearing))
        toward_radar = velocity_east * math.cos(bearing) - velocity_north * math.sin(bearing)

        shift = -self.seconds() * toward_radar
        return (shift * along[0], shift * along[1])
