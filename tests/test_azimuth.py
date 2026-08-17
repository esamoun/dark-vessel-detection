"""Where a moving vessel is drawn, as against where it is.

Every expectation here is a property of the geometry — a direction, a proportionality, a case
where the effect must vanish — rather than a number read off the Kattegat scene. The scene is
what the model is *checked* against, in `docs/failures.md` and in the run itself; a test that
pinned its six vessels would turn one acquisition's measurement into the definition of the
physics, and the next scene would fail for being a different scene.
"""

import math

import pytest

from darkvessel.fusion.azimuth import (
    ASCENDING,
    DESCENDING,
    IW_MID_SWATH_DEG,
    Geometry,
)

# The study area's latitude, and the pass the scene this chain runs on was acquired from.
KATTEGAT_LAT = 57.62
GEOMETRY = Geometry(orbit_pass=DESCENDING, incidence_deg=IW_MID_SWATH_DEG)


def test_a_descending_pass_travels_south_and_slightly_west():
    """Sentinel-1's orbit is retrograde at 98.18 degrees, so a descending track leans west of
    due south — not east of it, which is the sign this got wrong the first time."""
    bearing = GEOMETRY.bearing(KATTEGAT_LAT)

    assert 180.0 < bearing < 210.0
    assert bearing == pytest.approx(195.4, abs=0.3)


def test_an_ascending_pass_is_the_mirror_of_it():
    ascending = Geometry(orbit_pass=ASCENDING, incidence_deg=IW_MID_SWATH_DEG)

    assert ascending.bearing(KATTEGAT_LAT) == pytest.approx(344.6, abs=0.3)


def test_the_track_leans_further_from_the_meridian_the_further_north_it_goes():
    """A retrograde orbit crosses the meridian at an angle that grows with latitude, which is why
    the bearing is a function of where in the scene the vessel is and not a constant."""
    assert GEOMETRY.bearing(70.0) > GEOMETRY.bearing(57.62) > GEOMETRY.bearing(20.0)


def test_a_vessel_standing_still_is_drawn_where_it_is():
    """The control the Kattegat scene supplied, as a property: no velocity, no displacement."""
    assert GEOMETRY.displacement(0.0, 0.0, KATTEGAT_LAT) == (0.0, 0.0)


def test_a_vessel_running_along_the_track_is_not_displaced():
    """Only the part of the velocity aimed at the radar shifts a target. A vessel steaming
    parallel to the satellite's own path contributes no line-of-sight Doppler at all."""
    bearing = math.radians(GEOMETRY.bearing(KATTEGAT_LAT))
    east, north = 6.0 * math.sin(bearing), 6.0 * math.cos(bearing)

    shift_east, shift_north = GEOMETRY.displacement(east, north, KATTEGAT_LAT)

    assert math.hypot(shift_east, shift_north) == pytest.approx(0.0, abs=1e-9)


def test_a_vessel_running_straight_at_the_radar_is_displaced_the_full_amount():
    bearing = math.radians(GEOMETRY.bearing(KATTEGAT_LAT))
    east, north = 6.0 * math.cos(bearing), -6.0 * math.sin(bearing)

    shift = math.hypot(*GEOMETRY.displacement(east, north, KATTEGAT_LAT))

    assert shift == pytest.approx(6.0 * GEOMETRY.seconds(), rel=1e-9)


def test_the_displacement_lies_along_the_track_whatever_the_course():
    """Doppler is read as position along the track, so the shift can only ever be along it."""
    bearing = math.radians(GEOMETRY.bearing(KATTEGAT_LAT))
    for east, north in ((3.0, 4.0), (-5.0, 1.0), (0.5, -6.0)):
        shift_east, shift_north = GEOMETRY.displacement(east, north, KATTEGAT_LAT)
        across = shift_east * math.cos(bearing) - shift_north * math.sin(bearing)
        assert across == pytest.approx(0.0, abs=1e-9)


def test_the_displacement_is_proportional_to_speed():
    one = math.hypot(*GEOMETRY.displacement(2.0, 0.0, KATTEGAT_LAT))
    three = math.hypot(*GEOMETRY.displacement(6.0, 0.0, KATTEGAT_LAT))

    assert three == pytest.approx(3.0 * one, rel=1e-9)


def test_opposite_courses_are_thrown_opposite_ways():
    """Which is why a shipping lane carrying traffic both ways scatters its vessels in both
    directions along the track rather than shifting the whole scene one way."""
    outbound = GEOMETRY.displacement(4.0, 0.0, KATTEGAT_LAT)
    inbound = GEOMETRY.displacement(-4.0, 0.0, KATTEGAT_LAT)

    assert outbound[0] == pytest.approx(-inbound[0])
    assert outbound[1] == pytest.approx(-inbound[1])


def test_a_steeper_look_throws_a_vessel_further():
    """The far edge of the swath is both further away and looked at more obliquely, and both
    push the same way. It is the reason the incidence angle cannot be shrugged off."""
    near = Geometry(orbit_pass=DESCENDING, incidence_deg=29.1)
    far = Geometry(orbit_pass=DESCENDING, incidence_deg=46.0)

    assert far.seconds() > GEOMETRY.seconds() > near.seconds()


def test_the_mid_swath_constant_is_the_one_the_scene_was_corrected_with():
    """Not an assertion about the world — an assertion that the shipped default is the figure
    docs/decisions.md reports, so the two cannot drift apart unnoticed."""
    assert GEOMETRY.seconds() == pytest.approx(70.5, abs=0.5)


def test_a_pass_direction_nobody_recognises_is_refused():
    with pytest.raises(ValueError, match="orbit pass"):
        Geometry(orbit_pass="SIDEWAYS", incidence_deg=IW_MID_SWATH_DEG)


def test_an_incidence_angle_outside_the_swath_is_refused():
    """A number outside 20-50 degrees is not an IW incidence angle; it is a units mistake, or a
    look angle written where an incidence angle was asked for."""
    with pytest.raises(ValueError, match="incidence"):
        Geometry(orbit_pass=DESCENDING, incidence_deg=0.67)
