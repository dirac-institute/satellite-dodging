import numpy as np
from rubin_sim.utils import gnomonic_project_toxy, _angularSeparation, Site
from skyfield.api import load, wgs84, EarthSatellite
from astropy import units as u
from astropy import constants as const
from part1 import pointToLineDistance
from shapely.geometry import LineString, Point
from shapely import geometry


MJDOFFSET = 2400000.5
MJD0 = (
    60218  # is 23274 for a TLE via http://www.decimaltime.hynes.net/p/conversions.html
)


def satellite_mean_motion(altitude, mu=const.GM_earth, r_earth=const.R_earth):
    """
    Compute mean motion of satellite at altitude in Earth's gravitational field.
    See https://en.wikipedia.org/wiki/Mean_motion#Formulae
    """
    no = np.sqrt(4.0 * np.pi**2 * (altitude + r_earth) ** 3 / mu).to(u.day)
    return 1 / no


def tle_from_orbital_parameters(
    sat_name, sat_nr, epoch, inclination, raan, mean_anomaly, mean_motion
):
    """
    Generate TLE strings from orbital parameters.
    Note: epoch has a very strange format: first two digits are the year, next three
    digits are the day from beginning of year, then fraction of a day is given, e.g.
    20180.25 would be 2020, day 180, 6 hours (UT?)
    """

    # Note: RAAN = right ascention (or longitude) of ascending node

    # I suspect this is filling in 0 eccentricity everywhere.

    def checksum(line):
        s = 0
        for c in line[:-1]:
            if c.isdigit():
                s += int(c)
            if c == "-":
                s += 1
        return "{:s}{:1d}".format(line[:-1], s % 10)

    tle0 = sat_name
    tle1 = checksum(
        "1 {:05d}U 20001A   {:14.8f}  .00000000  00000-0  50000-4 "
        "0    0X".format(sat_nr, epoch)
    )
    tle2 = checksum(
        "2 {:05d} {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} "
        "{:11.8f}    0X".format(
            sat_nr,
            inclination.to_value(u.deg),
            raan.to_value(u.deg),
            mean_anomaly.to_value(u.deg),
            mean_motion.to_value(1 / u.day),
        )
    )

    return "\n".join([tle0, tle1, tle2])


def create_constellation(
    altitudes,
    inclinations,
    nplanes,
    sats_per_plane,
    epoch=23274.0,
    name="Test",
    seed=42,
):
    """Create a set of orbital elements for a satellite constellation then
    convert them to TLEs.
    """

    rng = np.random.default_rng(seed)

    my_sat_tles = []
    sat_nr = 8000
    for alt, inc, n, s in zip(altitudes, inclinations, nplanes, sats_per_plane):

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
                    name + " {:d}".format(sat_nr), sat_nr, epoch, inc, raan, ma, mm
                )
            )
            sat_nr += 1

    return my_sat_tles


def starlink_constellation_v1():
    """
    Create a list of satellite TLE's. 
    For starlink v1 (as of July 2022). Should create 4,408 orbits
    """
    altitudes = np.array([550, 540, 570, 560, 560]) * u.km
    inclinations = np.array([53, 53.2, 70, 97.6, 97.6]) * u.deg
    nplanes = np.array([72, 72, 36, 6, 4])
    sats_per_plane = np.array([22, 22, 20, 58, 43])

    my_sat_tles = create_constellation(
        altitudes, inclinations, nplanes, sats_per_plane, name="starV1"
    )

    return my_sat_tles


def starlink_constellation_v2():
    """
    Create a list of satellite TLE's
    For starlink v2 (as of July 2022). Should create 29,988 orbits
    """
    altitudes = np.array([340, 345, 350, 360, 525, 530, 535, 604, 614]) * u.km
    inclinations = np.array([53, 46, 38, 96.9, 53, 43, 33, 148, 115.7]) * u.deg
    nplanes = np.array([48, 48, 48, 30, 28, 28, 28, 12, 18])
    sats_per_plane = np.array([110, 110, 110, 120, 120, 120, 120, 12, 18])

    my_sat_tles = create_constellation(
        altitudes, inclinations, nplanes, sats_per_plane, name="starV2"
    )

    return my_sat_tles


def oneweb_constellation():
    """
    Create a list of satellite TLE's
    for OneWeb plans (as of July 2022). Should create 6,372 orbits
    """
    altitudes = np.array([1200, 1200, 1200]) * u.km
    inclinations = np.array([87.9, 40, 55]) * u.deg
    nplanes = np.array([49, 72, 72])
    sats_per_plane = np.array([36, 32, 32])

    my_sat_tles = create_constellation(
        altitudes, inclinations, nplanes, sats_per_plane, name="oneWe"
    )

    return my_sat_tles


class Constellation(object):
    """
    Have a class to hold satellite constellation
    Parameters
    ----------
    sat_tle_list : list of str
        A list of satellite TLEs to be used
    alt_limit : float (15)
        Altitude limit below which satellites can be ignored (degrees)
    fov : float (3.5)
        The field of view diameter (degrees)
    """

    def __init__(self, sat_tle_list, alt_limit=20.0, fov=3.5):
        self.alt_limit_rad = np.radians(alt_limit)
        self.fov_radius_rad = np.radians(fov / 2.0)

        # Load ephemeris for sun position
        self.eph = load("de421.bsp")

        self.sat_list = []
        self.ts = load.timescale()
        for tle in sat_tle_list:
            name, line1, line2 = tle.split("\n")
            self.sat_list.append(EarthSatellite(line1, line2, name, self.ts))

        self._make_location()

    def _make_location(self):
        telescope = Site(name="LSST")

        self.observatory_site = wgs84.latlon(
            telescope.latitude, telescope.longitude, telescope.height
        )

    def update_mjd(self, mjd):
        """
        Record the alt,az position and illumination status for all the satellites at a given time
        XXX--need to update so this will work with an array of MJD values, so we can avoid mjd loops.
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
        self.visible = np.where(
            (self.altitudes_rad >= self.alt_limit_rad) & (self.illum == True)
        )[0]

    def paths_array(self, mjds):
        """For an array of MJD values, compute the resulting RA,Dec and illumination status of 
        the full constellation at each MJD."""
        
        jd = mjds + MJDOFFSET
        t = self.ts.ut1_jd(jd)

        ras = []
        decs = []
        alts = []
        illums = []
        for sat in self.sat_list:
            current_sat = sat.at(t)
            illum = current_sat.is_sunlit(self.eph)
            illums.append(illum.copy())
            topo = current_sat - self.observatory_site.at(t)
            ra, dec, distance = topo.radec()
            alt, az, dist = topo.altaz()
            ras.append(ra.radians)
            decs.append(dec.radians)
            alts.append(alt.radians)
        return np.vstack(ras), np.vstack(decs), np.vstack(alts), np.vstack(illums)

    def check_pointing(self, pointing_alt, pointing_az, mjd, exposure_time, fov_radius=1.75):
        """Calculates the length of satellite streaks in a pointing. 
        Parameters
        ----------
        Param1 : float 
            the altitude of the pointing (degrees).
        Param2 : float
            the azimuth of the pointing (degrees).
        Param3 : float
            the current mjd (days).
        Param4: float 
            the length of exposure (seconds).
        fov_radius : float (1.75)
            The radius of the field of view (degrees), default 1.75.
        Returns
        -------
        list
            list of streak length in the given pointing (degrees)
            and the number of satellites that contributed to the length"""
        
        fov_radius = np.radians(fov_radius)
        pointing_alt = np.radians(pointing_alt)
        pointing_az = np.radians(pointing_az)
        exposure_time = exposure_time/86400
        streak_len_rad = 0.
        n_streaks = 0

        self.update_mjd(mjd)
        inAlt_list = self.altitudes_rad + 0
        inAz_list = self.azimuth_rad + 0
        illum1 = self.visible.copy()
        
        self.update_mjd(mjd+exposure_time)
        finAlt_list = self.altitudes_rad + 0 
        finAz_list = self.azimuth_rad + 0

        vis_sometime = np.unique(np.hstack([illum1, self.visible]))

        for index in vis_sometime: 
            distance = pointToLineDistance(inAz_list[index], inAlt_list[index], 
                                           finAz_list[index], finAlt_list[index], 
                                           pointing_az, pointing_alt)

            if distance < fov_radius:
                streak_len_rad += calculate_length(inAlt_list[index], inAz_list[index],
                                                   finAlt_list[index], finAz_list[index],
                                                   pointing_alt, pointing_az, fov_radius)
                n_streaks += 1
        return np.degrees(streak_len_rad), n_streaks

    def check_pointings(self, pointing_ras, pointing_decs, mjds, visit_time, fov_radius=1.75):
        """Just like `check_pointing`, but now use arrays for all the things
        Parameters
        ----------
        pointing_ras : array
            The RA for each pointing (degrees)
        pointing_decs : array
            The dec for each pointing (degres)
        mjds : array
            The MJD for the (start) of each pointing (days)
        visit_time : array
            The entire time a visit happend (seconds). We'll assume
        fov_radius : float (1.75)
            The radius of the science field of view (degrees)
        """

        # Arrays to hold results
        lengths_rad = np.zeros(pointing_ras.size, dtype=float)
        n_streaks = np.zeros(pointing_ras.size, dtype=int)

        input_id_indx_oned = np.arange(pointing_ras.size, dtype=int)

        # Convert everything to radians for internal computations
        pointing_ras = np.radians(pointing_ras)
        pointing_decs = np.radians(pointing_decs)
        fov_radius = np.radians(fov_radius)

        # Note self.paths_array should return an array that is N_sats x N_mjds in shape
        # And all angles in radians.
        sat_ra_1, sat_dec_1, sat_alt_1, sat_illum_1 = self.paths_array(mjds)
        mjd_end = mjds + visit_time/3600./24.
        sat_ra_2, sat_dec_2, sat_alt_2, sat_illum_2 = self.paths_array(mjd_end)

        # broadcast the pointings to be the same shape as the satellite arrays.
        pointing_ras = np.broadcast_to(pointing_ras, sat_ra_1.shape)
        pointing_decs = np.broadcast_to(pointing_decs, sat_ra_1.shape)
        input_id_indx = np.broadcast_to(input_id_indx_oned, sat_ra_1.shape)

        # Which satellites are above the altitude limit and illuminated
        # np.where confuses me when used on a 2d array. 
        above_illum_indx = np.where(((sat_alt_1 > self.alt_limit_rad) | (sat_alt_2 > self.alt_limit_rad)) &
                                    ((sat_illum_1 == True) | (sat_illum_2 == True)))

        # pointToLineDistance can take arrays, but they all need to be the same shape,
        # thus why we broadcasted pointing ra and dec above.
        distances = pointToLineDistance(sat_ra_1[above_illum_indx], sat_dec_1[above_illum_indx],
                                        sat_ra_2[above_illum_indx], sat_dec_2[above_illum_indx],
                                        pointing_ras[above_illum_indx], pointing_decs[above_illum_indx])

        close = np.where(distances < fov_radius)[0]

        # ok, this is pretty ugly, but should get the job done
        # Loop over all the potential collisions we have found
        for sat_ra1, sat_dec1, sat_ra2, sat_dec2, p_ra, p_dec, ob_indx in zip(sat_ra_1[above_illum_indx][close],
                                                                              sat_dec_1[above_illum_indx][close],
                                                                              sat_ra_2[above_illum_indx][close],
                                                                              sat_dec_2[above_illum_indx][close],
                                                                              pointing_ras[above_illum_indx][close],
                                                                              pointing_decs[above_illum_indx][close],
                                                                              input_id_indx[above_illum_indx][close]):        
            length = calculate_length(sat_dec1, sat_ra1, sat_dec2, sat_ra2, p_dec, p_ra, fov_radius)
            if length > 0:
                lengths_rad[ob_indx] += length
                n_streaks[ob_indx] += 1
        # Since we had degrees in, do degrees out. Probably poor form that we don't have 
        # uniform behavior over all methods. Maybe change methods that are radians in/out
        # to have a leading underscore _ in name to make clear.
        return np.degrees(lengths_rad), n_streaks


def calculate_length(initial_alt, initial_az, end_alt, end_az, pointing_alt, pointing_az, radius):
    """Helper funciton for check_pointing. 
    calculate the length of a streak after projecting the locations of the satellite and the pointing onto 2D.
    Parameters
    ----------
    Param1 : float 
        the initial altitude of the satellite (radians)
    Param2 : float
        the initial azimuth of the satellite (radians)
    Param3 : float
        the end altitude of the satellite (radians)
    Param4: float 
        the end azimuth of the satellite (radians)
    Param5 : float
        the altitude of the pointing (radians)
    Param6: float 
        the azimuth of the pointing(radians)
    Param7 : float
        the radius of the pointing (radians)
    Returns
    -------
    float
        the length of the satellite streak in the pointing (radians)
    """
    #start location 
    x1,y1=gnomonic_project_toxy(initial_az, initial_alt, pointing_az, pointing_alt)
    #end location
    x2,y2=gnomonic_project_toxy(end_az, end_alt, pointing_az, pointing_alt)

    # create your two points
    point_1 = geometry.Point(x1, y1)
    point_2 = geometry.Point(x2, y2)
    
    #from https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
    p = Point(0, 0)
    circle = p.buffer(radius).boundary
    circle_buffer=p.buffer(radius)
    line = LineString([(x1,y1), (x2,y2)])
    intersection = circle.intersection(line)
    try:
        if circle_buffer.contains(point_1) and circle_buffer.contains(point_2): 
            len=np.sqrt((x1-x2)**2+(y1-y2)**2)
        elif circle_buffer.contains(point_1):
            x_2=intersection.coords[0][0]
            y_2=intersection.coords[0][1]
            len=np.sqrt((x1-x_2)**2+(y1-y_2)**2)
        elif circle_buffer.contains(point_2):
            x_1=intersection.coords[0][0]
            y_1=intersection.coords[0][1]
            len=np.sqrt((x_1-x2)**2+(y_1-y2)**2)
        else:  
            p1=intersection.geoms[0].coords[0]
            p2=intersection.geoms[1].coords[0]
            len=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        return len 
    except:
        return 0