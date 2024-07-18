import os
import sys
import yaml
import argparse
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
verif_hour = conf[model_name]['verif_hour']
longest_hour = conf[model_name]['longest_hour']
leads_do = np.arange(verif_hour, longest_hour+verif_hour, verif_hour)
leads_do = leads_do - 1 # -1 for Python indexing
print('Verifying lead times: {}'.format(leads_do))
# ====================== #

def sp_avg(DS, wlat):
    return DS.weighted(wlat).mean(['lat', 'lon'], skipna=True)

path_verif = conf[model_name]['save_loc_verif']+'combined_acc_{}_{}_{}h.nc'.format(verif_ind_start, verif_ind_end, verif_hour)

# ERA5 and fcst climatology info
ERA5_path_string = conf['ERA5']['save_loc_clim'] + 'ERA5_DOY{:05}_HOD{:02}.nc'
OURS_clim_string = conf[model_name]['save_loc_clim'] + 'medium_boy_DOY{:03d}_LEAD{:03d}.nc'

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

acc_results = []

for fn_ours in filename_OURS:
    # --------------------------------------------------------------- #
    # import and subset forecast
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = vu.ds_subset_everything(ds_ours, variables_levels)
    ds_ours = ds_ours.isel(time=leads_do)
    dayofyear_ours = ds_ours['time.dayofyear']
    
    # --------------------------------------------------------------- #
    # get ERA5 verification target
    ds_target = ds_ERA5_merge.sel(time=ds_ours['time']).compute()
    
    # --------------------------------------------------------------- #
    # get ERA5 climatology
    # pull day of year for anomaly computation
    dayofyear_ERA5 = ds_target['time.dayofyear'].values
    hourofday_ERA5 = ds_target['time'].dt.hour
    
    required_ERA5_clim = [
        ERA5_path_string.format(day, hourofday_ERA5[i_day]) for i_day, day in enumerate(dayofyear_ERA5)]
    
    for fn_required in required_ERA5_clim:
        if os.path.exists(fn_required) is False:
            print('Missing: {}'.format(fn_required))
            raise
        
    print('ERA5 climatology file requirments fulfilled')
    
    # open all ERA5 climatology files and merge as one
    ds_ERA5_clim = [xr.open_dataset(fn) for fn in required_ERA5_clim]
    ds_clim_merge = xr.concat(ds_ERA5_clim, dim='time')
    
    # unify coord names
    #ds_clim_merge = ds_clim_merge.rename({'latitude':'lat','longitude':'lon'})
    ds_clim_merge['time'] = ds_target['time']
    
    # ========================================== #
    # ERA5 anomaly
    ds_anomaly_ERA5 = ds_target - ds_clim_merge
    # ========================================== #
    
    # --------------------------------------------------------------- #
    # get forecast climatology
    required_OURS_clim = [
        OURS_clim_string.format(day, (leads_do[i_day])+1) for i_day, day in enumerate(dayofyear_ours)]
    
    for fn_required in required_OURS_clim:
        if os.path.exists(fn_required) is False:
            print('Missing: {}'.format(fn_required))
            raise
        
    print('OURS climatology file requirments fulfilled')
    
    # open all fcst climatology files and merge as one
    datasets_f = [xr.open_dataset(fn) for fn in required_OURS_clim]

    # unify coord names
    fcst_clim = xr.concat(datasets_f, dim='time')
    fcst_clim = fcst_clim.drop_vars('level')
    fcst_clim['time'] = ds_ours['time']

    # ========================================== #
    # fcst anomaly
    ds_anomaly_OURS = ds_ours - fcst_clim
    
    # ========================================== #
    # anmalies --> ACC with latitude-based cosine weights (check sp_avg and w_lat)
    top = sp_avg(ds_anomaly_OURS*ds_anomaly_ERA5, w_lat)
    
    bottom = np.sqrt(
        sp_avg(ds_anomaly_OURS**2, w_lat) * sp_avg(ds_anomaly_ERA5**2, w_lat))
                
    acc_results.append((top/bottom).drop_vars('time'))
    
    print('ACC completed: {}'.format(fn_ours))
    
# Combine ACC results
ds_acc = xr.concat(acc_results, dim='days')

# Save
print('Save to {}'.format(path_verif))
ds_acc.to_netcdf(path_verif)













