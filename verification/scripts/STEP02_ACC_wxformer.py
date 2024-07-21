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
lead_range = conf[model_name]['lead_range']
verif_lead_range = conf[model_name]['verif_lead_range']

leads_exist = list(np.arange(lead_range[0], lead_range[-1]+lead_range[0], lead_range[0]))
leads_verif = list(np.arange(verif_lead_range[0], verif_lead_range[-1]+verif_lead_range[0], verif_lead_range[0]))
ind_lead = vu.lead_to_index(leads_exist, leads_verif)

print('Verifying lead times: {}'.format(leads_verif))
print('Verifying lead indices: {}'.format(ind_lead))
# ====================== #

def sp_avg(DS, wlat):
    return DS.weighted(wlat).mean(['lat', 'lon'], skipna=True)

path_verif = conf[model_name]['save_loc_verif']+'combined_acc_{:04d}_{:04d}_{:03d}h_{:03d}h_{}.nc'.format(
                                                                                            verif_ind_start, 
                                                                                            verif_ind_end,
                                                                                            verif_lead_range[0],
                                                                                            verif_lead_range[-1],
                                                                                            model_name)

# ERA5 climatology info
ERA5_path_string = conf['ERA5_weatherbench']['save_loc_clim'] + 'ERA5_clim_1990_2019_6h_interp.nc'
ds_ERA5_clim = xr.open_dataset(ERA5_path_string)

# ---------------------------------------------------------------------------------------- #
# ERA5 verif target
filename_ERA5 = sorted(glob(conf['ERA5_ours']['save_loc']))

# pick years
year_range = conf['ERA5_ours']['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_ERA5 = [fn for fn in filename_ERA5 if any(year in fn for year in years_pick)]

# merge yearly ERA5 as one
ds_ERA5 = [vu.get_forward_data(fn) for fn in filename_ERA5]
ds_ERA5_merge = xr.concat(ds_ERA5, dim='time')
    
# Select the specified variables and their levels
variables_levels = conf['ERA5_ours']['verif_variables']

# subset merged ERA5 and unify coord names
ds_ERA5_merge = vu.ds_subset_everything(ds_ERA5_merge, variables_levels)
ds_ERA5_merge = ds_ERA5_merge.rename({'latitude':'lat','longitude':'lon'})

# ---------------------------------------------------------------------------------------- #
# forecast
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))

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

# some of the forecast files have lat/lon as masked arrays
# this may result-in a mismatch between the weatherbench clim (lat/lon arrays) and the fcst (masked arrays)
# it only happens to some CREDIT rollouts but will be applied to IFS as well
OURS_dataset = xr.open_dataset(conf['geo']['geo_file_nc'])
x_OURS = np.array(OURS_dataset['longitude'])
y_OURS = np.array(OURS_dataset['latitude'])

acc_results = []

for fn_ours in filename_OURS:
    # --------------------------------------------------------------- #
    # import and subset forecast
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = vu.ds_subset_everything(ds_ours, variables_levels)
    ds_ours = ds_ours.isel(time=ind_lead)
    dayofyear = ds_ours['time.dayofyear']
    hourofday = ds_ours['time'].dt.hour

    # ============================== #
    # resolve the masked array issue
    ds_ours['lon'] = x_OURS
    ds_ours['lat'] = y_OURS
    # ============================== #
    ds_ours = ds_ours.compute()
    
    # --------------------------------------------------------------- #
    # get ERA5 verification target
    ds_target = ds_ERA5_merge.sel(time=ds_ours['time']).compute()
    
    # --------------------------------------------------------------- #
    # get ERA5 climatology
    ds_clim_target = ds_ERA5_clim.sel(dayofyear=dayofyear, hour=hourofday).compute()
    
    # ========================================== #
    # ERA5 anomaly
    ds_anomaly_ERA5 = ds_target - ds_clim_target

    # fcst anomaly
    ds_anomaly_OURS = ds_ours - ds_clim_target
    
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






