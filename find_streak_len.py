import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from rubin_sim.data import get_data_dir

import sqlite3
import argparse

from sat_utils import Constellation, starlink_constellation_v1

def data_setup(datalocation): 
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
    df = pd.read_sql('select * from observations where night < 366;', con)

    con.close()

    return df


def compute_streak_len(datalocation, constellation,fieldRA='fieldRA',fieldDec='fieldDec',mjd='observationStartMJD', exptime='visitTime', id='observationId'): 
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
    Param7 : id
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

    df= data_setup(datalocation)
    fast_lengths, fast_nstreaks = constellation.check_pointings(df[fieldRA].values, df[fieldDec].values,
                                                           df[mjd].values,
                                                           df[exptime].values)
    ids=df.loc[:,id]

    res=list(zip(ids,fast_lengths,fast_nstreaks))

    return res




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, default='streak_len.npz')
    parser.add_argument("--season_frac", type=float, default=0.1)
    parser.add_argument("--datalocation", type=str, default = None)
    parser.add_argument("--fieldRA", type=str, default = 'fieldRA')    
    parser.add_argument("--fieldDec", type=str, default = 'fieldDec') 
    parser.add_argument("--mjd", type=str, default = 'observationStartMJD') 
    parser.add_argument("--exptime", type=str, default = 'visitTime') 
    parser.add_argument("--id", type=str, default = 'observationId') 
    args = parser.parse_args()
    filename = args.out_file
    season_frac = args.season_frac
    datalocation = args.datalocation
    tles = starlink_constellation_v1()
    constellation = Constellation(tles)
    fieldRA= args.fieldRA
    fieldDec=args.fieldDec
    mjd=args.mjd
    exptime=args.exptime
    id=args.id

    obs_array = compute_streak_len(datalocation, constellation, fieldRA, fieldDec, mjd, exptime, id)
    np.savez(filename, obs_array=obs_array)





