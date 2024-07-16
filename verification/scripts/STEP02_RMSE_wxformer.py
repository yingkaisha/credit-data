import os
import sys
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

config_name = os.path.realpath('../verif_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_ind_start', help='verif_ind_start')
parser.add_argument('verif_ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

verif_ind_start = int(args['verif_ind_start'])
verif_ind_end = int(args['verif_ind_end'])
# ====================== #
model_name = 'wxformer'
# ====================== #

leads_do = np.arange(240)

path_verif = conf[model_name]['save_loc_verif']+'combined_rmse_{}_{}.nc'.format(verif_ind_start, verif_ind_end)

# ---------------------------------------------------------------------------------------- #
# ERA5 verif target
filename_ERA5 = sorted(glob(conf['ERA5']['save_loc']))

# pick years
year_range = conf['ERA5']['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_ERA5 = [fn for fn in filename_ERA5 if any(year in fn for year in years_pick)]

# merge yearly ERA5 as one
ds_ERA5 = [vu.get_forward_data(fn) for fn in filename_ERA5]
ds_ERA5_merge = xr.concat(ds_ERA5, dim='time')
    
# Select the specified variables and their levels
variables_levels = conf['ERA5']['verif_variables']

# subset merged ERA5 and unify coord names
ds_ERA5_merge = vu.ds_subset_everything(ds_ERA5_merge, variables_levels)
ds_ERA5_merge = ds_ERA5_merge.rename({'latitude':'lat','longitude':'lon'})

# ---------------------------------------------------------------------------------------- #
# forecast
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))
filename_OURS_backup = sorted(glob('/glade/campaign/cisl/aiml/ksha/CREDIT/gathered/*.nc'))

# manual input bad files in '/glade/campaign/cisl/aiml/gathered/'
# provide replacements in '/glade/campaign/cisl/aiml/ksha/CREDIT/gathered/'
# correct file info and rerun climo days/leads that touchs the bad files
ind_bad = [206, 209, 211, 215, 360, 390, 400]
filename_bad = []

for i, i_bad in enumerate(ind_bad):
    file_old = filename_OURS[i_bad]
    file_new = filename_OURS_backup[i]

    if os.path.basename(file_old) == os.path.basename(file_new):
        filename_bad.append(file_new)
        filename_OURS[i_bad] = filename_OURS_backup[i]
    else:
        print('Replacement of bad file {} not found'.format(file_old))
        raise
        
# pick years
year_range = conf[model_name]['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_OURS = [fn for fn in filename_OURS if any(year in fn for year in years_pick)]

L_max = len(filename_OURS)
assert verif_ind_end <= L_max, 'verified indices (days) exceeds the max index available'

filename_OURS = filename_OURS[verif_ind_start:verif_ind_end]

# latitude weighting
lat = xr.open_dataset(filename_OURS[0])["lat"]
w_lat = np.cos(np.deg2rad(lat))
w_lat = w_lat / w_lat.mean()

# ---------------------------------------------------------------------------------------- #
# RMSE compute
verif_results = []

for fn_ours in filename_OURS:
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = vu.ds_subset_everything(ds_ours, variables_levels)
    ds_ours = ds_ours.isel(time=leads_do)
    
    ds_target = ds_ERA5_merge.sel(time=ds_ours['time']).compute()

    # RMSE with latitude-based cosine weighting (check w_lat)
    RMSE = np.sqrt((w_lat* (ds_ours - ds_target)**2).mean(['lat', 'lon']))
    
    verif_results.append(RMSE.drop_vars('time'))

    print('Completed: {}'.format(fn_ours))
    
# Combine verif results
ds_verif = xr.concat(verif_results, dim='days')

# Save the combined dataset
print('Save to {}'.format(path_verif))
ds_verif.to_netcdf(path_verif)

