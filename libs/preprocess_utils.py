'''
base_trainer.py
-------------------------------------------------------
Content:
    - get_forward_data
    - zscore_var
    - residual_zscore_var

Yingkai Sha
ksha@ucar.edu
'''

import os
import yaml
import numpy as np
import xarray as xr
from glob import glob

def get_forward_data(filename) -> xr.DataArray:
    '''
    Check nc vs. zarr files
    open file as xr.Dataset
    '''
    if filename[-3:] == '.nc' or filename[-4:] == '.nc4':
        dataset = xr.open_dataset(filename)
    else:
        dataset = xr.open_zarr(filename, consolidated=True)
    return dataset

def zscore_var(conf, varname, ind_level=None):
    '''
    Compute mean and variance (can be converted to std) from yearly zarr or nc files
    It combines two yearly files iteratively using the pooling equations:
    https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    The function relies on one of the credit-data/data_preprocessing/data_config.yml

    This function works for all variables
    Input coordinates should be (time, level, longitude, latitude)
    '''
    # ------------------------------------------------------------------------------------ #
    # lists yearly files and open as xr.Dataset
    filenames = sorted(glob(conf['zscore'][varname]))
    
    year_range = conf['zscore']['years_range']
    train_years = [str(year) for year in range(year_range[0], year_range[1])]
    train_files = [file for file in filenames if any(year in file for year in train_years)]
    
    list_ds_train = []
    
    for fn in train_files:
        list_ds_train.append(get_forward_data(fn))
        
    # ------------------------------------------------------------------------------------ #
    # determine if the var has levels
    ds_example = list_ds_train[0][varname]
    var_shape = ds_example.shape
    N_grids = var_shape[-1] * var_shape[-2]
    mean_std_save = np.empty((2,))
    mean_std_save.fill(np.nan)

    # ------------------------------------------------------------------------------------ #
    # loop thorugh files and compute mean and std 
    for i_fn, ds in enumerate(list_ds_train):
        
        # get the xr.Dataset per var per level
        if ind_level is not None:
            ds_subset = ds[varname].isel(level=ind_level)
        else:
            ds_subset = ds[varname]
        
        # get mean and var for the current year
        mean_current_yr = float(ds_subset.mean())
        var_current_yr = float(ds_subset.var())
        L = len(ds_subset) * N_grids
        
        print('{} - {}'.format(mean_current_yr, var_current_yr))
            
        if i_fn == 0:
            # if it is the first year, pass current year to the combined 
            mean_std_save[0] = mean_current_yr
            mean_std_save[1] = var_current_yr
            N_samples = L
            
        else:
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            mean_new = (L * mean_current_yr + N_samples * mean_std_save[0]) / (L + N_samples)
            var_new = ((L - 1) * var_current_yr + (N_samples - 1) * mean_std_save[1]) / (L + N_samples - 1)
            var_new_adjust = (L * N_samples * (mean_current_yr - mean_std_save[0])**2) / (L + N_samples) / (L + N_samples -1)
            
            mean_std_save[0] = mean_new
            mean_std_save[1] = var_new + var_new_adjust
            N_samples = N_samples + L
            
            print('{} - {}'.format(mean_std_save[0], mean_std_save[1]))
            
    if ind_level is not None:
        save_name = conf['zscore']['save_loc'] + '{}_level{}_mean_std_{}.npy'.format(conf['zscore']['prefix'], ind_level+1, varname)
    else:
        save_name = conf['zscore']['save_loc'] + '{}_mean_std_{}.npy'.format(conf['zscore']['prefix'], varname)
        
    print('Save to {}'.format(save_name))
    np.save(save_name, mean_std_save)

def residual_zscore_var(conf, varname, ind_level=None):
    '''
    Given yearly zarr or nc files, compute the zscore of a variable, apply 
    np.diff on its 'time' coordinate, and compute the mean and std the resulting np.diff outputs.

    This function works for all variables
    Input coordinates should be (time, level, longitude, latitude)
    '''

    filenames = sorted(glob(conf['residual'][varname]))
    
    year_range = conf['residual']['years_range']
    train_years = [str(year) for year in range(year_range[0], year_range[1])]
    train_files = [file for file in filenames if any(year in file for year in train_years)]
    
    list_ds_train = []
    
    for fn in train_files:
        list_ds_train.append(get_forward_data(fn))
        
    # ------------------------------------------------------------------------------------ #
    ds_example = list_ds_train[0][varname]
    var_shape = ds_example.shape
    
    N_grids = var_shape[-1] * var_shape[-2]
    mean_std_save = np.empty((2,))
    mean_std_save.fill(np.nan)

    # ------------------------------------------------------------------------------------ #
    # mean, std
    ds_mean = xr.open_dataset(conf['residual']['mean_loc'])
    ds_std = xr.open_dataset(conf['residual']['std_loc'])
    
    if ind_level is not None:
        ds_mean = ds_mean.isel(level=ind_level)
        ds_std = ds_std.isel(level=ind_level)

    for i_fn, ds in enumerate(list_ds_train):
        
        if ind_level is not None:
            ds = ds.isel(level=ind_level)
            
        # ===================================================================== #
        # apply np.diff
        var_diff = xr.apply_ufunc(
            np.diff,
            (ds[varname] - ds_mean[varname]) / ds_std[varname],
            input_core_dims=[['time']],
            output_core_dims=[['time_diff']],  # Change this to a new dimension name
            vectorize=True,
            dask='allowed',
            output_dtypes=[ds[varname].dtype]
        )
        
        ds_out = var_diff.to_dataset(name='{}_diff'.format(varname))
        
        ds_out = ds_out.assign_coords(
            time_diff=ds_out['time_diff'])
        
        ds_out = ds_out.transpose("time_diff", "latitude", "longitude")
        
        # ===================================================================== #
        # compute the mean and std from the np.diff result
        
        ds_subset = ds_out['{}_diff'.format(varname)]
        
        # get mean and var for the current year
        mean_current_yr = float(ds_subset.mean())
        var_current_yr = float(ds_subset.var())
        L = len(ds_subset) * N_grids
        
        print('{} - {}'.format(mean_current_yr, var_current_yr))
            
        if i_fn == 0:
            # if it is the first year, pass current year to the combined 
            mean_std_save[0] = mean_current_yr
            mean_std_save[1] = var_current_yr
            N_samples = L
            
        else:
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            mean_new = (L * mean_current_yr + N_samples * mean_std_save[0]) / (L + N_samples)
            var_new = ((L - 1) * var_current_yr + (N_samples - 1) * mean_std_save[1]) / (L + N_samples - 1)
            var_new_adjust = (L * N_samples * (mean_current_yr - mean_std_save[0])**2) / (L + N_samples) / (L + N_samples -1)
            
            mean_std_save[0] = mean_new
            mean_std_save[1] = var_new + var_new_adjust
            N_samples = N_samples + L
            
            print('{} - {}'.format(mean_std_save[0], mean_std_save[1]))
    
    if ind_level is not None:
        save_name = conf['residual']['save_loc'] + '{}_level{}_mean_std_{}.npy'.format(conf['residual']['prefix'], ind_level+1, varname)
    else:
        save_name = conf['residual']['save_loc'] + '{}_mean_std_{}.npy'.format(conf['residual']['prefix'], varname)
        
    print('Save to {}'.format(save_name))
    np.save(save_name, mean_std_save)


