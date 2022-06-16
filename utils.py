from part1 import pointToLineDistance
import numpy as np 
from shapely.geometry import LineString
from shapely.geometry import Point








def check_pointing(self, pointing_alt, pointing_az, mjd, exposure_time):
    """Calculates the length of satellite streaks in a pointing. 
    Param: the altitude and azimuth of the pointing, the current mjd, and the length of exposure. 
    Returns a list of streak length in the given pointing"""
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
        if distance<self.radius:
            streak=calculate_length(initial_lat, initial_lon, end_lat, end_lon, pointing_alt, pointing_az)
            streak_len.append(streak)



    

def calculate_length(initial_lat, initial_lon, end_lat, end_lon, pointing_alt, pointing_az ):
    """Helper funciton for check_pointing. 
    calculate the length of a streak after projecting the locations of the satellite and the pointing onto 2D."""
    #stsart location 
    #TODO: what should be the center 
    x1,y1=gnomonic_project_toxy(initial_lat, initial_lon, RAcen, Deccen)
    #end location
    x2,y2=gnomonic_project_toxy(end_lat, end_lon, RAcen, Deccen)
    #center of pointing 
    x_c,y_c=gnomonic_project_toxy(pointing_alt, pointing_az, RAcen, Deccen)
    #TODO: find radius project. How? 


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
    Input radians. Grabbed from sims_selfcal"""
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