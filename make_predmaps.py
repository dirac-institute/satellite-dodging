#!/usr/bin/env python

import numpy as np
import healpy as hp
from sat_utils import (starlink_constellation_v1, starlink_constellation_v2,
                       Constellation, oneweb_constellation)
from rubin_sim.utils import survey_start_mjd, _healbin
import argparse
from astropy.time import Time
from rubin_sim.utils import Site
from astropy.coordinates import get_sun, EarthLocation, AltAz


def generate_sat_maps(mjd_start_offset=0, mjd_end_offset=1,
                      nside=64, mjd_start=None,
                      timestep=20.,
                      constellation_name="starlink_v1",
                      sun_alt_limit=-11,
                      alt_limit=20):
    """Generate an array of mjds and healpix maps that have predicted satellite positions
    """
    timestep = timestep/3600./24  # Seconds to days
    alt_limit = np.radians(alt_limit)

    if constellation_name == 'starlink_v1':
        star_tles = starlink_constellation_v1()
    elif constellation_name == 'starlink_v2':
        star_tles = starlink_constellation_v2()
    elif constellation_name == 'oneweb':
        star_tles = oneweb_constellation()
    else:
        ValueError("No valid constellation name given")

    constellation = Constellation(star_tles)
    mjd0 = survey_start_mjd() if mjd_start is None else mjd_start
    mjds = np.arange(mjd0+mjd_start_offset,
                     mjd0+mjd_end_offset+timestep,
                     timestep)

    # Toss any mjds where the sun alt is higher than the limit
    site = Site("LSST")
    location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    t_sparse = Time(mjds, format="mjd", location=location)
    sun = get_sun(t_sparse)
    aa = AltAz(location=location, obstime=t_sparse)
    sun_aa = sun.transform_to(aa)
    sun_down = np.where(sun_aa.alt.deg <= sun_alt_limit)
    mjds = mjds[sun_down]

    # Compute RA and decs for when sun is down
    ras, decs, alts, illums = constellation.paths_array(mjds)

    below_limit = np.where(alts < alt_limit)

    weights = np.zeros(ras.shape, dtype=int)
    weights[illums] = 1
    weights[below_limit] = 0

    satellite_maps = []
    final_mjds = []
    for i, mjd in enumerate(mjds):

        spot_map = _healbin(ras[:,i][illums[:,i]], decs[:,i][illums[:,i]],
                            weights[:,i][illums[:,i]], nside, reduceFunc=np.sum, dtype=int,
                            fillVal=0)
        if np.sum(spot_map) > 0:
            satellite_maps.append(spot_map)
            final_mjds.append(mjd)

    satellite_maps = np.vstack(satellite_maps)

    np.savez(constellation_name+'_%i_%i.npz' % (mjd_start_offset, mjd_end_offset),
             satellite_maps=satellite_maps, mjds=final_mjds, all_mjds=mjds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=1)
    parser.add_argument("--name", type=str, default='starlink_v1')
    args = parser.parse_args()

    generate_sat_maps(mjd_start_offset=args.start, mjd_end_offset=args.end,
                      constellation_name=args.name)



