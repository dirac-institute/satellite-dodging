from part1 import pointToLineDistance
import numpy as np 
from shapely.geometry import LineString
from shapely.geometry import Point
from rubin_sim.utils import Site
import ephem
from rubin_sim.utils import _angularSeparation, _buildTree, xyz_angular_radius
from rubin_sim.scheduler.utils import read_fields

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
            The MJD to advance the satellites to
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




##numpy doc docstring 
def check_pointing(self, pointing_alt, pointing_az, mjd, exposure_time):
    """Calculates the length of satellite streaks in a pointing. 
    Parameters
    ----------
    Param1 : float 
        the altitude of the pointing
    Param2 : float
        the azimuth of the pointing
    Param3 : float
        the current mjd
    Param4: float 
        the length of exposure.

    Returns
    -------
    list
        list of streak length in the given pointing"""
    streak_len=[]
    sat_list=self.sat_list
    self.update_mjd(mjd)
    inLat_list=self.latitude
    inLong_list=self.longtitude
    initialPositions=zip(inLat_list,inLong_list)
    self.updata_mjd(mjd+exposure_time)
    finLat_list=self.latitude
    finLong_list=self.longtitude   
    endPositions=zip(finLat_list,finLong_list)
    for i in range(len(initialPositions)):
        initial_lat, initial_lon=initialPositions[i]
        end_lat, end_lon = endPositions[i]
        distance=pointToLineDistance(initial_lat, initial_lon, end_lat, end_lon, pointing_alt, pointing_az)
        print(distance)
        print(self.radius)
        if distance<self.radius:
            streak=calculate_length(initial_lat, initial_lon, end_lat, end_lon, pointing_alt, pointing_az, self.radius)
            streak_len.append(streak)



    

def calculate_length(initial_lat, initial_lon, end_lat, end_lon, pointing_alt, pointing_az, radius ):
    """Helper funciton for check_pointing. 
    calculate the length of a streak after projecting the locations of the satellite and the pointing onto 2D.
    Parameters
    ----------
    Param1 : float 
        the initial latitude of the satellite
    Param2 : float
        the initial longitude of the satellite
    Param3 : float
        the end latitude of the satellite
    Param4: float 
        the end longitude of the satellite
    Param5 : float
        the altitude of the pointing 
    Param6: float 
        the azimuth of the pointing
    Param7 : float
        the radius of the pointing 


    Returns
    -------
    float
        the length of the satellite streak in the pointing 
    """
    #stsart location 
    x1,y1=gnomonic_project_toxy(initial_lat, initial_lon, pointing_alt, pointing_az)
    #end location
    x2,y2=gnomonic_project_toxy(end_lat, end_lon, pointing_alt, pointing_az)
    #center of pointing 
    x_c,y_c=gnomonic_project_toxy(pointing_alt, pointing_az, pointing_alt, pointing_az)
    #TODO: so is radius just the old radius?? 


    #from https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
    p = Point(x_c,y_c)
    circle = p.buffer(radius).boundary
    line = LineString([(x1,y1), (x2,y2)])
    i = circle.intersection(line)

    p1=i.geoms[0].coords[0]
    p2=i.geoms[1].coords[0]

    len=np.sqrt(p1**2+p2**2)

    
    return len 




#need to project start point, end point, center, and radius 
def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal
    Parameters
    ----------
    Param1 : float 
        the right ascension of the object
    Param2 : float
        the declination of the object
    Param3 : float
        the right ascension of the center of the system
    Param4: float 
        the declination of the center of the system



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