import os
import argparse
from glob import glob
from datetime import datetime, timedelta

import dask
import numpy as np
import xarray as xr

from tqdm import tqdm
from dask import delayed
from distributed import Client
from dask_jobqueue import PBSCluster
from concurrent.futures import ProcessPoolExecutor

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_ind_start', help='verif_ind_start')
parser.add_argument('verif_ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

verif_ind_start = int(args['verif_ind_start'])
verif_ind_end = int(args['verif_ind_end'])
print('select verif inds {}:{}'.format(verif_ind_start, verif_ind_end))
leads_do = np.array([5, 23, 47, 71, 95, 119, 143, 167, 191, 215, 239])


path_verif = '/glade/derecho/scratch/ksha/CREDIT/verif/wxformer_acc_{}_{}.nc'.format(verif_ind_start, verif_ind_end)

def get_forward_data(filename) -> xr.DataArray:
    """Lazily opens the Zarr store on gladefilesystem.
    """
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset

def select_variables(ds, variables_levels):
    selected_vars = {}
    for var, levels in variables_levels.items():
        selected_vars[var] = ds[var] if levels is None else ds[var].isel(level=levels)
    return xr.Dataset(selected_vars)

def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
  return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  bounds = _latitude_cell_bounds(points)
  bounds[0] = bounds[0]*-1
  bounds[-1] = bounds[-1]*-1
  # _assert_increasing(bounds)
  upper = bounds[1:]
  lower = bounds[:-1]
  # normalized cell area: integral from lower to upper of cos(latitude)
  return np.sin(lower) - np.sin(upper)


def get_lat_weights(ds: xr.Dataset) -> xr.DataArray:
  """Computes latitude/area weights from latitude coordinate of dataset."""
  weights = _cell_area_from_latitude(np.deg2rad(ds.lat.data))
  weights /= np.mean(weights)
  weights = ds.lat.copy(data=weights)
  return weights


def _assert_increasing(x: np.ndarray):
  if not (np.diff(x) > 0).all():
    raise ValueError(f"array is not increasing: {x}")

def sp_avg(DS, wlat):
    return DS.weighted(wlat).mean(['lat', 'lon'], skipna=True)

# Adjust your file paths and chunk sizes as necessary
filenames_ERA5 = sorted(glob('/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/TOTAL_*'))
filenames_ERA5 = filenames_ERA5[-6:-1]
ds_ERA5 = [get_forward_data(fn) for fn in filenames_ERA5]
ds_ERA5_merge = xr.concat(ds_ERA5, dim='time')
    
# Select the specified variables and their levels
variables_levels = {
    'V500': None, 
    'U500': None, 
    'T500': None, 
    'Q500': None, 
    'Z500': None,
    'SP': None, 
    't2m': None,}

ds_ERA5_merge = select_variables(ds_ERA5_merge, variables_levels)
ds_ERA5_merge = ds_ERA5_merge.rename({'latitude':'lat','longitude':'lon'})

# forecast
filenames_OURS = sorted(glob('/glade/campaign/cisl/aiml/gathered/*.nc'))
filenames_OURS = [fn for fn in filenames_OURS if '2018' in fn or '2019' in fn or '2020' in fn]
filenames_OURS = filenames_OURS[verif_ind_start:verif_ind_end]

# latitude weighting
lat = xr.open_dataset(filenames_OURS[0])["lat"]
w_lat = np.cos(np.deg2rad(lat))
w_lat = w_lat / w_lat.mean()

#all ERA obs
filename_ERA5_clim = sorted(
    glob('/glade/work/wchapman/miles_branchs/pull_Jun27_2024/notebooks/Forecast_Verification/ERA5_DOY*.nc'))

#all for climo
filename_OURS_clim = sorted(glob('/glade/campaign/cisl/aiml/gathered/forecast_climo/medium_boy_DOY*.nc'))

acc_results = []


for fn_ours in tqdm(filenames_OURS, desc="Processing files"):
    flag_run = True
    
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = select_variables(ds_ours, variables_levels)
    ds_ours = ds_ours.isel(time=leads_do)
    
    
    #observation anomaly:
    ds_target = ds_ERA5_merge.sel(time=ds_ours['time']).compute()
    
    dayofyear_values = ds_target['time.dayofyear'].values

    string_ERA5_clim = '/glade/work/wchapman/miles_branchs/pull_Jun27_2024/notebooks/Forecast_Verification/ERA5_DOY{:05d}.nc'
    required_ERA5_clim = [
        string_ERA5_clim.format(day) for day in dayofyear_values]

    for fn_required in required_ERA5_clim:
        if os.path.exists(fn_required) is False:
            print('Missing: {}'.format(fn_required))
            flag_run = False

    ds_ERA5_clim = [xr.open_dataset(fn) for fn in required_ERA5_clim]
    ds_clim_merge = xr.concat(ds_ERA5_clim, dim='time')
    ds_clim_merge = ds_clim_merge.rename({'latitude':'lat','longitude':'lon'})
    
    ds_clim_merge['time'] = ds_target['time']
    
    ds_anomaly_ERA5 = ds_target - ds_clim_merge

    #forecast anomaly:
    string_OURS_clim = '/glade/campaign/cisl/aiml/gathered/forecast_climo/medium_boy_DOY{:03d}_LEAD{:03d}.nc'
    required_OURS_clim = [string_OURS_clim.format(day, (leads_do[ee])+1) for ee, day in enumerate(ds_ours['time.dayofyear'])]

    for fn_required in required_OURS_clim:
        if os.path.exists(fn_required) is False:
            print('Missing: {}'.format(fn_required))
            flag_run = False

    # --------------------------------------------------------------- #
    # ACC section
    if flag_run:
        datasets_f = [xr.open_dataset(fn) for fn in required_OURS_clim]
        
        for_climo = xr.concat(datasets_f, dim='time')
        for_climo = for_climo.drop_vars('level')
        for_climo['time'] = ds_ours['time']
        ds_anomaly_OURS = ds_ours - for_climo
        
        top = sp_avg(ds_anomaly_OURS*ds_anomaly_ERA5, w_lat)
        bottom = np.sqrt(
            sp_avg(ds_anomaly_OURS**2, w_lat) * sp_avg(ds_anomaly_ERA5**2, w_lat))
                
        acc_results.append((top/bottom).drop_vars('time'))

# Combine RMSE results if needed
combined_acc = xr.concat(acc_results, dim='DOY')


# Save the combined dataset
print('Save to {}'.format(path_verif))
combined_acc.to_netcdf(path_verif)




