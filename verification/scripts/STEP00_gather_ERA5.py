import os
import sys
import yaml
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd

# ------------------------------------------------- #
# interpolation utils
# from scipy.interpolate import griddata
import scipy.interpolate as spint
from scipy.spatial import Delaunay
import itertools

def interp_weights(xy, uv, d=2):
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

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
# ======================= #
model_name = 'ERA5_weatherbench'
filename_prefix = 'ERA5_%Y-%m-%dT%HZ.nc'
save_loc = conf[model_name]['save_loc'] + filename_prefix
# interpolation weights were computed for 90N -> 90S
# If the data has 90S -> 90N, it should be flipped
flip_lat = False
# ======================= #

# import the original ERA5 from WeatherBench GS
ds_ERA5 = xr.open_zarr(
    'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')

# --------------------------------------------------------- #
# subset and organize their xr.Dataset

variables_levels = conf[model_name]['verif_variables']
ds_ERA5 = vu.ds_subset_everything(ds_ERA5, variables_levels)

# unify variable and coord names
ds_ERA5 = ds_ERA5.rename({'latitude':'lat','longitude':'lon'})
ds_ERA5 = ds_ERA5.rename(conf[model_name]['rename_variables'])
ds_ERA5 = ds_ERA5.squeeze('level')

# --------------------------------------------------------- #
# preparing for the regriding and separated *.nc save 

# ERA5 lat/lons
x_ERA5 = np.array(ds_ERA5['lon'])
y_ERA5 = np.array(ds_ERA5['lat'])

if flip_lat:
    y_ERA5 = np.flipud(y_ERA5)
    
lon_ERA5, lat_ERA5 = np.meshgrid(x_ERA5, y_ERA5)

# OUR lat/lons
OURS_dataset = xr.open_dataset(conf['geo']['geo_file_nc'])
x_OURS = np.array(OURS_dataset['longitude'])
y_OURS = np.array(OURS_dataset['latitude'])
lon_OURS, lat_OURS = np.meshgrid(x_OURS, y_OURS)
shape_OURS = lon_OURS.shape

# pick the years we need
year_range = conf[model_name]['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1)

# get initialization time
init_time = pd.to_datetime(ds_ERA5['time'])
# get variables
list_var_names = list(ds_ERA5.keys())

# interp weights
temp_data = np.load(conf['geo']['regrid_weights_numpy'], allow_pickle=True)[()]
vtx = temp_data['vtx']
wts = temp_data['wts']

for i_dt, dt_index in enumerate(init_time[verif_ind_start:verif_ind_end]):

    # indexing could start from nonzero
    i_dt = i_dt + verif_ind_start
    
    # init year is within selection 
    if dt_index.year in years_pick:

        # get file name
        save_name = datetime.strftime(dt_index, save_loc)

        # save and skip exists
        if os.path.exists(save_name) is False:
            
            print('Processing {}'.format(os.path.basename(save_name)))
            
            # allocate regrided dataset
            ds_ERA5_regrid = xr.Dataset()
            ds_ERA5_regrid = ds_ERA5_regrid.assign_coords({'lon': x_OURS, 'lat': y_OURS})
            
            # subset on initialization time
            ds_ERA5_slice = ds_ERA5.isel(time=slice(i_dt, i_dt+1))
    
            # -------------------------------------------------------------------------- #
            # interpolation section
    
            # assign time coord info to the allocated xr.Dataset
            ds_ERA5_regrid['time'] = ds_ERA5_slice['time']
            
            # loop through variables
            for var_name in list_var_names:
                
                print('Interpolate {}'.format(var_name))
                
                # select the variable on the current time
                ERA5_var = ds_ERA5_slice[var_name].isel(time=0)
    
                # ========================================================================== #
                if flip_lat:
                    ERA5_var = np.flipud(ERA5_var)
                # scipy.interpolate.griddata(method='linear') with manually inputted weights #
                ERA5_var_regrid = interpolate(ERA5_var, vtx, wts)
                ERA5_var_regrid = np.reshape(ERA5_var_regrid, shape_OURS)
                # ========================================================================== #
                
                # np.array --> xr.DataArray
                ERA5_var_regrid_da = xr.DataArray(
                    ERA5_var_regrid[None, ...], 
                    coords={
                        'time': ds_ERA5_slice['time'],
                        'lat': y_OURS, 
                        'lon': x_OURS,},
                    dims=['time', 'lat', 'lon'])
    
                # add xr.DataArray to the allocated xr.Dataset
                ds_ERA5_regrid[var_name] = ERA5_var_regrid_da
    
            ds_ERA5_regrid = ds_ERA5_regrid.drop_vars('level')
            
            # Save to netCDF4
            ds_ERA5_regrid.to_netcdf(save_name)
            print('Save to {}'.format(save_name))
        