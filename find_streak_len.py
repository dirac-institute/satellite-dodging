import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

from rubin_sim.data import get_data_dir
import sqlite3

from sat_utils import Constellation, starlink_constellation_v1

def data_setup(): 

    limit = 900

    dd = get_data_dir()
    baseline_file = os.path.join(dd,'sim_baseline/baseline.db')

    # Conenct to the sqlite database
    con = sqlite3.connect(baseline_file)

    # Load up the first year
    df = pd.read_sql('select * from observations where night < 1 and sunAlt > -24 ;', con)

    con.close()


    # Make a satellite constellation
    tles = starlink_constellation_v1()
    constellation = Constellation(tles)

    return df, constellation

def first_method(df): 
    ids=[]
    lengths = []
    nstreaks = []
    n_rows = len(df)
    t1 = time.time()
    for index, row in df.iterrows():
        length, streak = constellation.check_pointing(row['altitude'], row['azimuth'],
                                        row['observationStartMJD'], row['visitTime'])
        ids.append(df.loc[row,'observationId'])
        lengths.append(length)
        nstreaks.append(streak)
        # A simple progress bar
        progress = index/float(n_rows)*100
        text = "\rprogress = %.3f%%" % progress
        sys.stdout.write(text) 
        sys.stdout.flush() 
    t2 = time.time()
    # print('runtime = %.2f min' % ((t2-t1)/60.) )
    res=list(zip(ids,lengths,nstreaks))

    return res


def second_method(df): 
    t1 = time.time()

    fast_lengths, fast_nstreaks = constellation.check_pointings(df['fieldRA'].values, df['fieldDec'].values,
                                                           df['observationStartMJD'].values,
                                                           df['visitTime'].values)
    ids=df.loc[:,'observationId']
    t2 = time.time()
    # print('runtime = %.2f min' % ((t2-t1)/60.) )

    res=list(zip(ids,fast_lengths,fast_nstreaks))

    return fast_lengths 

