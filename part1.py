import numpy as np 

def bear(latA, lonA, latB, lonB):
    """All radians
    """
    # BEAR Finds the bearing from one lat / lon point to another.
    #bearing: The horizontal angle between the astronomic meridian and a line on the Earth
    #the angle between 2 points on the sky. northpole is always at 0. How far to you have to rotate 
    result = np.arctan2(np.sin(lonB - lonA) * np.cos(latB),
                        np.cos(latA) * np.sin(latB) - np.sin(latA) * np.cos(latB) * np.cos(lonB - lonA)
                        )

    return result

def _angularSeparation(long1, lat1, long2, lat2):
    """
    angle between 2 points
    """
    ## haversine distance 
    #how far apart two points on the sky are 
    t1 = np.sin(lat2/2.0 - lat1/2.0)**2
    t2 = np.cos(lat1)*np.cos(lat2)*np.sin(long2/2.0 - long1/2.0)**2
    _sum = t1 + t2

    if np.size(_sum) == 1:
        if _sum < 0.0:
            _sum = 0.0
    else:
        _sum = np.where(_sum < 0.0, 0.0, _sum)

    return 2.0*np.arcsin(np.sqrt(_sum))
    

def pointToLineDistance(lon1, lat1, lon2, lat2, lon3, lat3):
    """All radians
    points 1 and 2 define an arc segment,
    this finds the distance of point 3 to the arc segment. 
    """

    result = lon1*0
    needed = np.ones(result.size, dtype=bool)

    bear12 = bear(lat1, lon1, lat2, lon2)
    bear13 = bear(lat1, lon1, lat3, lon3)
    dis13 = _angularSeparation(lon1, lat1, lon3, lat3)

    # Is relative bearing obtuse?
    diff = np.abs(bear13 - bear12)
    if np.size(diff) == 1:
        if diff > np.pi:
            diff = 2*np.pi - diff
        if diff > (np.pi / 2):
            return dis13
    else:
        over = np.where(diff > np.pi)
        diff[over] = 2*np.pi - diff[over]
        solved = np.where(diff > (np.pi / 2))[0]
        result[solved] = dis13[solved]
        needed[solved] = False
    
    # Find the cross-track distance.
    dxt = np.arcsin(np.sin(dis13) * np.sin(bear13 - bear12))

    # Is p4 beyond the arc?
    dis12 = _angularSeparation(lon1, lat1, lon2, lat2)
    dis14 = np.arccos(np.cos(dis13) / np.cos(dxt))
    if np.size(dis14) == 1:
        if dis14 > dis12:
            return _angularSeparation(lon2, lat2, lon3, lat3)
    else:
        solved = np.where((dis14 > dis12) & needed)[0]
        result[solved] = _angularSeparation(lon2[solved], lat2[solved], lon3[solved], lat3[solved])
        needed[solved] = False

    if np.size(lon1) == 1:
        return np.abs(dxt)
    else:
        result[needed] = np.abs(dxt[needed])
        return result
