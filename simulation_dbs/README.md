Directory to hold survey simulations that attempt to avoid satellite streaks

Bad form to put giant files in git, so these can be downloaded from NCSA: https://lsst.ncsa.illinois.edu/sim-data/satellite-dodging/simulation_dbs/

files should have the format satellite_sim_X.X_YYYY_v2.2_Zyrs.db

* X.X is the weight given to the satellite avoidance basis function (so all sims with zero weight _should_ be identical. Good sanity check, but note that the scheduler is not guarenteed cross-platform repeatable because of floating point precision differences.)
* YYYY is an abreviation for the satellite constellation that was used to generate streak avoidance maps. (so slv1 is starlink_v1. Obviously that sim would not be expected to do a good job avoiding OneWeb satellites.)
* Z is the number of years the survey was run (Probably limiting things to 1 year for now. If we get ambitious we can go for a full 10).


Also in here is the `summary.h5` file. This has the results of running the standard MAF analysis pipeline on all the simulations. 