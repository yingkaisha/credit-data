# CREDIT-data

## About

This repository ties to the main repository of the the NSF NCAR MILES Community Runnable Earth Digital Intelligence Twin (CREDIT) and provides additional support in terms of data pre-processing, post-processing, and verification. It also hosts the result of CREDIT papers from the NSF NCAR MILES group.

## Python environment

Same as the main CREDIT repository. 

As of right now, it is only accessable to the NSF NCAR MILES group, but will be publicly available soon.

## Navigaition

* `libs`: A collection of functions/scripts used by this repository.
  * `verif_utils.py`: functions used for forecasts post-processing and verification.
  * `score_utils.py`: function sued for computing certain verification scores.
  * `graph_utils.py`: data visualization functions.

* `data_preprocessing`: this folder contains data pre-processing step for CREDIT model runs.
  * This folder is under construction; it contains ERA5 pressure level and model level preprocessing, static field preprocessing, zscore computation, and residual norm computation.   

* `verification`: this folder contains verification steps for CREDIT model runs. It can be implemented as follows:
  * Copy `verif_config_template.yml` to `verif_config.yml` and modify based on your file directories
  * Go through Jupyter notebooks from `STEP00` to `STEP02`
  * Access scripts folder for large-scale verification runs
  * Note: the verification setup has only been tested on the data analysis server of NSF NCAR: `casper.ucar.edu`

 * `visualization`: this folder hosts results of the CREDIT papers.
