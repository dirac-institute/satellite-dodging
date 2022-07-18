#!/usr/bin/env python

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from rubin_sim.data import get_data_dir

import sqlite3
import argparse

from sat_utils import (Constellation, starlink_constellation_v1,
                       starlink_constellation_v2, oneweb_constellation,
                       sun_alt_limits)

def data_setup(datalocation, sun_alt_limit=-90): 
    """
    Loads up the dataframe given a datalocation 
    Parameters
    ----------
    Param1 : str
       the location of data 
    
    Returns
    -------
        dataframe 
        dataframe that contains the data in that location
    """
    limit = 900

    # Conenct to the sqlite database
    con = sqlite3.connect(datalocation)

    # Load up the first year
    df = pd.read_sql('select * from observations where night < 366 and sunAlt > %f;' % sun_alt_limit, con)
    
    con.close()

    return df


def compute_streak_len(datalocation, constellation,fieldRA='fieldRA',fieldDec='fieldDec',
                       mjd='observationStartMJD', exptime='visitTime', obsid='observationId',
                       sun_alt_limit=-90): 
    """
    computes the streak length of satellite crossing 
    Parameters
    ----------
    Param1 : datalocation
       the location of data that contains the pointing informations.
    Param2 : constellation
       the constellation of satellites.
    Param3 : fieldRA
       the name of the dataframe column that gives information about the RA of the pointing.
    Param4 : fieldDec
       the name of the dataframe column that gives information about the Dec of the pointing.
    Param5 : mjd
       the name of the dataframe column that gives information about the observation mjd of the pointing.
    Param6 : exptime
       the name of the dataframe column that gives information about the exposure time of the pointing.
    Param7 : obsid
       the name of the dataframe column that gives information about the observation id of the pointing.
    
    Returns
    -------
        list
        list with 3-Tuple elements containing observation id, the length of the streaks, and the number of streaks of each pointing.
    """
    if datalocation == None: 
        dd = get_data_dir()
        baseline_file = os.path.join(dd,'sim_baseline/baseline.db')
        datalocation = baseline_file

    df = data_setup(datalocation, sun_alt_limit=sun_alt_limit)

    fast_lengths, fast_nstreaks = constellation.check_pointings(df[fieldRA].values, df[fieldDec].values,
                                                                df[mjd].values,
                                                                df[exptime].values)
    ids=df.loc[:,obsid]

    res=list(zip(ids,fast_lengths,fast_nstreaks))

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--season_frac", type=float, default=0.1)
    parser.add_argument("--datalocation", type=str, default = None)
    parser.add_argument("--fieldRA", type=str, default = 'fieldRA')    
    parser.add_argument("--fieldDec", type=str, default = 'fieldDec') 
    parser.add_argument("--mjd", type=str, default = 'observationStartMJD') 
    parser.add_argument("--exptime", type=str, default = 'visitTime') 
    parser.add_argument("--obsid", type=str, default = 'observationId') 
    parser.add_argument("--constellation_name", type=str, default='slv1')
    args = parser.parse_args()
    season_frac = args.season_frac
    datalocation = args.datalocation
 
    if args.constellation_name == 'slv1':
      tles = starlink_constellation_v1()
    elif args.constellation_name == 'slv2':
      tles = starlink_constellation_v2()
    elif args.constellation_name == 'oneweb':
       tles = oneweb_constellation()
    else:
      ValueError('Constellation name unknown, use slv1, slv2 or oneweb')

    sun_alt_limit = sun_alt_limits()[args.constellation_name]

    constellation = Constellation(tles)
    fieldRA= args.fieldRA
    fieldDec=args.fieldDec
    mjd=args.mjd
    exptime=args.exptime
    obsid=args.obsid

    filename = datalocation.replace('.db', '') + '.npz'

    obs_array = compute_streak_len(datalocation, constellation, fieldRA,
                                   fieldDec, mjd, exptime, obsid,
                                   sun_alt_limit=sun_alt_limit)
    np.savez(filename, obs_array=obs_array)





