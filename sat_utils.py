import numpy as np
from rubin_sim.utils import gnomonic_project_toxy, _angularSeparation, Site
from skyfield.api import load, wgs84, EarthSatellite
from astropy import units as u
from astropy import constants as const


MJDOFFSET = 2400000.5


def satellite_mean_motion(altitude, mu=const.GM_earth, r_earth=const.R_earth):
    '''
    Compute mean motion of satellite at altitude in Earth's gravitational field.

    See https://en.wikipedia.org/wiki/Mean_motion#Formulae
    '''
    no = np.sqrt(4.0 * np.pi ** 2 * (altitude + r_earth) ** 3 / mu).to(u.day)
    return 1 / no


def tle_from_orbital_parameters(sat_name, sat_nr, epoch, inclination, raan,
                                mean_anomaly, mean_motion):
    '''
    Generate TLE strings from orbital parameters.

    Note: epoch has a very strange format: first two digits are the year, next three
    digits are the day from beginning of year, then fraction of a day is given, e.g.
    20180.25 would be 2020, day 180, 6 hours (UT?)
    '''

    # Note: RAAN = right ascention (or longitude) of ascending node

    def checksum(line):
        s = 0
        for c in line[:-1]:
            if c.isdigit():
                s += int(c)
            if c == "-":
                s += 1
        return '{:s}{:1d}'.format(line[:-1], s % 10)

    tle0 = sat_name
    tle1 = checksum(
        '1 {:05d}U 20001A   {:14.8f}  .00000000  00000-0  50000-4 '
        '0    0X'.format(sat_nr, epoch))
    tle2 = checksum(
        '2 {:05d} {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} '
        '{:11.8f}    0X'.format(
            sat_nr, inclination.to_value(u.deg), raan.to_value(u.deg),
            mean_anomaly.to_value(u.deg), mean_motion.to_value(1 / u.day)
        ))

    return '\n'.join([tle0, tle1, tle2])


def create_constellation(altitudes, inclinations, nplanes, sats_per_plane,
                         epoch=22050.1, name='Test', seed=42):

    rng = np.random.default_rng(seed)

    my_sat_tles = []
    sat_nr = 8000
    for alt, inc, n, s in zip(
            altitudes, inclinations, nplanes, sats_per_plane):

        if s == 1:
            # random placement for lower orbits
            mas = rng.uniform(0, 360, n) * u.deg
            raans = rng.uniform(0, 360, n) * u.deg
        else:
            mas = np.linspace(0.0, 360.0, s, endpoint=False) * u.deg
            mas += rng.uniform(0, 360, 1) * u.deg
            raans = np.linspace(0.0, 360.0, n, endpoint=False) * u.deg
            mas, raans = np.meshgrid(mas, raans)
            mas, raans = mas.flatten(), raans.flatten()

        mm = satellite_mean_motion(alt)
        for ma, raan in zip(mas, raans):
            my_sat_tles.append(
                tle_from_orbital_parameters(
                    name+' {:d}'.format(sat_nr), sat_nr, epoch,
                    inc, raan, ma, mm))
            sat_nr += 1

    return my_sat_tles


def starlink_constellation_v1():
    """
    Create a list of satellite TLE's
    """
    altitudes = np.array([550, 540, 570, 560, 560]) * u.km
    inclinations = np.array([53, 53.2, 70, 97.6, 97.6]) * u.deg
    nplanes = np.array([72, 72, 36, 6, 4])
    sats_per_plane = np.array([22, 22, 20, 58, 43])

    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name="starV1")

    return my_sat_tles


class Constellation(object):
    """
    Have a class to hold ephem satellite objects

    Parameters
    ----------
    sat_tle_list : list of str
        A list of satellite TLEs to be used
    tstep : float (5)
        The time step to use when computing satellite positions in an exposure
    """

    def __init__(self, sat_tle_list, alt_limit=15., fov=3.5, mjd0=60218.):
        self.alt_limit_rad = np.radians(alt_limit)
        self.fov_radius_rad = np.radians(fov/2.)

        # Load ephemeris for sun position
        self.eph = load('de421.bsp')

        self.sat_list = []
        self.ts = load.timescale()
        for tle in sat_tle_list:
            name, line1, line2 = tle.split('\n')
            self.sat_list.append(EarthSatellite(line1, line2, name, self.ts))

        self._make_location()

    def _make_location(self):
        telescope = Site(name='LSST')

        self.observatory_site = wgs84.latlon(telescope.latitude,
                                             telescope.longitude,
                                             telescope.height)

    def update_mjd(self, mjd):
        """
        observer : ephem.Observer object
        """
        jd = mjd + MJDOFFSET
        t = self.ts.ut1_jd(jd)
        
        self.altitudes_rad = []
        self.azimuth_rad = []
        self.illum = []
        for sat in self.sat_list:
            current_sat = sat.at(t)
            illum = current_sat.is_sunlit(self.eph)
            self.illum.append(illum.copy())
            if illum:
                topo = current_sat - self.observatory_site.at(t)
                alt, az, dist = topo.altaz()  # this returns an anoying Angle object
                self.altitudes_rad.append(alt.radians + 0)
                self.azimuth_rad.append(az.radians + 0)
            else:
                self.altitudes_rad.append(np.nan)
                self.azimuth_rad.append(np.nan)

        self.altitudes_rad = np.array(self.altitudes_rad)
        self.azimuth_rad = np.array(self.azimuth_rad)
        self.illum = np.array(self.illum)
        # Keep track of the ones that are up and illuminated
        self.visible = np.where((self.altitudes_rad >= self.alt_limit_rad) & (self.illum == True))[0]


    def paths_array(mjds):
        """Maybe pass in an arary of MJD vallues and return the RA,Dec (and illumination) arrays for each satellite
        """
        pass
        