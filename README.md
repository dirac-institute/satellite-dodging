# satellite-dodging
To address the problem of bright commercial satellite streaks in images from the Vera C. Rubin Observatory, we create heuristics for satellite dodging strategies using the survey’s scheduling algorithm. We computationally forecast satellite trajectories to account for the growing satellite population during Rubin operations. Our results help maximize efficiency and quality of this flagship observatory.


Dependencies:
As usual, suggest using anaconda and creating a fresh environement for installing with something like: `conda create -n sat-env ; conda activate sat-env`

* rubin_sim: Maybe just `conda install -c conda-forge rubin_sim` Full instructions at:  https://github.com/lsst/rubin_sim
* shapely: `conda install -c conda-forge shapely`
* skyfield:  `pip install skyfield`

Files: 
Compare_sim_runs.ipynb contains all result plots, including evaluation of dodging efficiency, number of missed exposures, and trade-off between pixel loss and coadded depth. 
compare_methods.ipynb contains testing of the satellite streak length method and speed optimization testing 



