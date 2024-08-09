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

def zscore_by_var(conf, varname):
    '''
    Compute mean and variance (can be converted to std) from yearly zarr or nc files
    It combines two yearly files iteratively using the pooling equations:
    https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    The function relies on one of the credit-data/data_preprocessing/data_config.yml
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
    
    if len(var_shape) == 4:
        flag_has_level = True
    elif len(var_shape) == 3:
        flag_has_level = False
    else:
        print('data shape {} is not accepted, need to have at least (time, lat, lon)'.format(var_shape))
        raise
    
    N_grids = var_shape[-1] * var_shape[-2]
    
    if flag_has_level:
        N_levels = var_shape[1]
        mean_std_save = np.empty((2, N_levels))
        mean_std_save.fill(np.nan)
        N_samples = np.empty((N_levels,))
        N_samples.fill(np.nan)
    
    else:
        mean_std_save = np.empty((2,))
        mean_std_save.fill(np.nan)

    # ------------------------------------------------------------------------------------ #
    # if has level:
    if flag_has_level:
        
        # loop thorugh files and compute mean and std 
        for i_fn, ds in enumerate(list_ds_train):
            
            # loop through levels
            for i_level in range(N_levels):
                
                # get the xr.Dataset per var per level
                ds_subset = ds[varname].isel(level=i_level)
                
                # get mean and var for the current year
                mean_current_yr = float(ds_subset.mean())
                var_current_yr = float(ds_subset.var())
                L = len(ds_subset) * N_grids
                
                print('level {} current {} - {}'.format(i_level, mean_current_yr, var_current_yr))
                    
                if i_fn == 0:
                    # if it is the first year, pass current year to the combined 
                    mean_std_save[0, i_level] = mean_current_yr
                    mean_std_save[1, i_level] = var_current_yr
                    N_samples[i_level] = L
                    
                else:
                    # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
                    mean_new = (L * mean_current_yr + N_samples[i_level] * mean_std_save[0, i_level]) / (L + N_samples[i_level])
                    var_new = ((L - 1) * var_current_yr + (N_samples[i_level] - 1) * mean_std_save[1, i_level]) / (L + N_samples[i_level] - 1)
                    var_new_adjust = (L * N_samples[i_level] * (mean_current_yr - mean_std_save[0, i_level])**2) / (L + N_samples[i_level]) / (L + N_samples[i_level] - 1)
                    
                    mean_std_save[0, i_level] = mean_new
                    mean_std_save[1, i_level] = var_new + var_new_adjust
                    N_samples[i_level] = N_samples[i_level] + L
                    
                    print('level {} combine {} - {}'.format(i_level, mean_std_save[0, i_level], mean_std_save[1, i_level]))
                        
    # ------------------------------------------------------------------------------------ #
    # if no level
    else:
        # loop thorugh files and compute mean and std 
        for i_fn, ds in enumerate(list_ds_train):
            
            # get the xr.Dataset per var per level
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

    save_name = conf['zscore']['save_loc'] + '{}_mean_std_{}.npy'.format(conf['zscore']['prefix'], varname)
    print('Save to {}'.format(save_name))
    np.save(save_name, mean_std_save)
