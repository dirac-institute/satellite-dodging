
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
        solved = np.where(diff > (np.pi / 2))[0]
        result[solved] = dis13[solved]
        needed[solved] = 0
    
    # Find the cross-track distance.
    dxt = np.arcsin(np.sin(dis13) * np.sin(bear13 - bear12))

    # Is p4 beyond the arc?
    dis12 = _angularSeparation(lon1, lat1, lon2, lat2)
    dis14 = np.arccos(np.cos(dis13) / np.cos(dxt))
    if np.size(dis14) == 1:
        if dis14 > dis12:
            return _angularSeparation(lon2, lat2, lon3, lat3)
    else:
        solved = np.where(dis14 > dis12)[0]
        result[solved] = _angularSeparation(lon2[solved], lat2[solved], lon3[solved], lat3[solved])

    if np.size(lon1) == 1:
        return np.abs(dxt)
    else:
        result[needed] = np.abs(dxt[needed])
        return result