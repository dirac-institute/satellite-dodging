from part1 import pointToLineDistance
import numpy as np 
from shapely.geometry import LineString
from shapely.geometry import Point
from rubin_sim.utils import Site
import ephem
from rubin_sim.utils import _angularSeparation, _buildTree, xyz_angular_radius
from rubin_sim.scheduler.utils import read_fields
from astropy import constants as const
from astropy import units as u
from astropy import time
from shapely import geometry

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

def create_constellation(altitudes, inclinations, nplanes, sats_per_plane, epoch=22050.1, name='Test'):

    my_sat_tles = []
    sat_nr = 8000
    for alt, inc, n, s in zip(
            altitudes, inclinations, nplanes, sats_per_plane):

        if s == 1:
            # random placement for lower orbits
            mas = np.random.uniform(0, 360, n) * u.deg
            raans = np.random.uniform(0, 360, n) * u.deg
        else:
            mas = np.linspace(0.0, 360.0, s, endpoint=False) * u.deg
            mas += np.random.uniform(0, 360, 1) * u.deg
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

def starlink_constellation(supersize=False, fivek=False):
    """
    Create a list of satellite TLE's
    """
    altitudes = np.array([550, 1110, 1130, 1275, 1325, 345.6, 340.8, 335.9])
    inclinations = np.array([53.0, 53.8, 74.0, 81.0, 70.0, 53.0, 48.0, 42.0])
    nplanes = np.array([72, 32, 8, 5, 6, 2547, 2478, 2493])
    sats_per_plane = np.array([22, 50, 50, 75, 75, 1, 1, 1])

    if supersize:
        # Let's make 4 more altitude and inclinations
        new_altitudes = []
        new_inclinations = []
        new_nplanes = []
        new_sat_pp = []
        for i in np.arange(0, 4):
            new_altitudes.append(altitudes+i*20)
            new_inclinations.append(inclinations+3*i)
            new_nplanes.append(nplanes)
            new_sat_pp.append(sats_per_plane)

        altitudes = np.concatenate(new_altitudes)
        inclinations = np.concatenate(new_inclinations)
        nplanes = np.concatenate(new_nplanes)
        sats_per_plane = np.concatenate(new_sat_pp)

    altitudes = altitudes * u.km
    inclinations = inclinations * u.deg
    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name='Starl')

    if fivek:
        stride = round(len(my_sat_tles)/5000)
        my_sat_tles = my_sat_tles[::stride]

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

    def __init__(self, sat_tle_list, alt_limit=30., fov=3.5, tstep=5., exptime=30., seed=42):
        np.random.seed(seed)
        self.sat_list = [ephem.readtle(tle.split('\n')[0], tle.split('\n')[1], tle.split('\n')[2]) for tle in sat_tle_list]
        self.alt_limit_rad = np.radians(alt_limit)
        self.fov_rad = np.radians(fov)
        self._make_observer()
        self._make_fields()
        self.tsteps = np.arange(0, exptime+tstep, tstep)/3600./24.  # to days

        self.radius = xyz_angular_radius(fov)

    def _make_fields(self):
        """
        Make tesselation of the sky
        """
        # RA and dec in radians
        fields = read_fields()

        # crop off so we only worry about things that are up
        good = np.where(fields['dec'] > (self.alt_limit_rad - self.fov_rad))[0]
        self.fields = fields[good]

        self.fields_empty = np.zeros(self.fields.size)

        # we'll use a single tessellation of alt az
        leafsize = 100
        self.tree = _buildTree(self.fields['RA'], self.fields['dec'], leafsize, scale=None)

    def _make_observer(self):
        telescope = Site(name='LSST')

        self.observer = ephem.Observer()
        self.observer.lat = telescope.latitude_rad
        self.observer.lon = telescope.longitude_rad
        self.observer.elevation = telescope.height

    def advance_epoch(self, advance=100):
        """
        Advance the epoch of all the satellites
        """

        # Because someone went and put a valueError where there should have been a warning
        # I prodly present the hackiest kludge of all time
        for sat in self.sat_list:
            sat._epoch += advance

    def set_epoch(self, mjd):
        for sat in self.sat_list:
            sat._epoch = mjd

    #self.update_mjd gives a bunch of positions 
    def update_mjd(self, mjd, indx=None):
        """
        mjd : float
            The MJD to advance the satellites to (days)
        indx : list-like of ints
            Only propigate a subset of satellites. 
        """
        self.active_indx = indx

        self.observer.date = ephem.date(time.Time(mjd, format='mjd').datetime)

        self.altitudes_rad = []
        self.azimuth_rad = []
        self.eclip = []
        if self.active_indx is None:
            indx = np.arange(len(self.sat_list))
        else:
            indx = self.active_indx
        for i in indx:
            sat = self.sat_list[i]
            try:
                sat.compute(self.observer)
            except ValueError:
                self.set_epoch(self.observer.date+np.random.uniform()*10)
                sat.compute(self.observer)
            self.altitudes_rad.append(sat.alt)
            self.azimuth_rad.append(sat.az)
            self.eclip.append(sat.eclipsed)

        self.altitudes_rad = np.array(self.altitudes_rad)
        self.azimuth_rad = np.array(self.azimuth_rad)
        self.eclip = np.array(self.eclip)
        # Keep track of the ones that are up and illuminated
        self.above_alt_limit = np.where((self.altitudes_rad >= self.alt_limit_rad) & (self.eclip == False))[0]
    


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
            list of streak length in the given pointing"""
        
        fov_radius = np.radians(fov_radius)
        pointing_alt=np.radians(pointing_alt)
        pointing_az=np.radians(pointing_az)
        exposure_time=exposure_time/86400
        streak_len=[]

        self.update_mjd(mjd)
        inAlt_list=self.altitudes_rad + 0
        inAz_list=self.azimuth_rad + 0
        
        self.update_mjd(mjd+exposure_time)
        finAlt_list=self.altitudes_rad + 0 
        finAz_list=self.azimuth_rad + 0

        
        for index in self.above_alt_limit: 
            elem_list=list(zip(inAlt_list, inAz_list, finAlt_list, finAz_list))[index]
            initial_alt=elem_list[0]
            initial_az=elem_list[1]
            end_alt=elem_list[2]
            end_az=elem_list[3]

            distance=pointToLineDistance(initial_alt, initial_az, end_alt, end_az, pointing_alt, pointing_az)

            if distance<fov_radius:
                streak=calculate_length(initial_alt, initial_az, end_alt, end_az, pointing_alt, pointing_az, fov_radius)
                streak_len.append(streak)
        return streak_len


def calculate_length(initial_alt, initial_az, end_alt, end_az, pointing_alt, pointing_az, radius ):
    """Helper funciton for check_pointing. 
    calculate the length of a streak after projecting the locations of the satellite and the pointing onto 2D.
    Parameters
    ----------
    Param1 : float 
        the initial altitude of the satellite (degree)
    Param2 : float
        the initial azimuth of the satellite (degree)
    Param3 : float
        the end altitude of the satellite (degree)
    Param4: float 
        the end azimuth of the satellite (degree)
    Param5 : float
        the altitude of the pointing (degree)
    Param6: float 
        the azimuth of the pointing(degree)
    Param7 : float
        the radius of the pointing (degree)


    Returns
    -------
    float
        the length of the satellite streak in the pointing (not sure what the unit should be -- degrees? meters?)
    """
    #stsart location 
    x1,y1=gnomonic_project_toxy(initial_alt, initial_az, pointing_alt, pointing_az)
    #end location
    x2,y2=gnomonic_project_toxy(end_alt, end_az, pointing_alt, pointing_az)

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
            return len 
        elif circle_buffer.contains(point_1):
            x_2=intersection.coords[0][0]
            y_2=intersection.coords[0][1]
            len=np.sqrt((x1-x_2)**2+(y1-y_2)**2)
            return len 
        elif circle_buffer.contains(point_2):
            x_1=intersection.coords[0][0]
            y_1=intersection.coords[0][1]
            len=np.sqrt((x_1-x2)**2+(y_1-y2)**2) 
            return len  
        else:  
            p1=intersection.geoms[0].coords[0]
            p2=intersection.geoms[1].coords[0]
            len=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
            return len 
    except:
            print(f"edge case: line: {line}, radius: {radius}, intersection: {intersection}")
            pass







#need to project start point, end point, center, and radius 
def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal
    Parameters
    ----------
    Param1 : float 
        the right ascension of the object (degrees)
    Param2 : float
        the declination of the object (degrees)
    Param3 : float
        the right ascension of the center of the system (degrees)
    Param4: float 
        the declination of the center of the system (degrees)



    Returns
    -------
    list
        two element list that contains the x,y projection of the object 
    
    
    """
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(
        RA1 - RAcen
    )
    x = np.cos(Dec1) * np.sin(RA1 - RAcen) / cosc
    y = (
        np.cos(Deccen) * np.sin(Dec1)
        - np.sin(Deccen) * np.cos(Dec1) * np.cos(RA1 - RAcen)
    ) / cosc
    return x, y