'''
A collection of functions used for the verification of CREDIT project
-----------------------------------------------------------------------
Content:

    Combining individual netCDF files
        - create_dir
        - get_nc_files
        - ds_subset_everything
        - process_file_group
        
    Calculate forecast and ERA5 climatology
        - get_doy_range
        - select_doy_range
        - get_filename_prefix_by_radius
        - dataset_time_slice
        - open_datasets_with_preprocessing
        - gaussian_weights_for_temporal_sum
        - weighted_temporal_sum

    Calculate verification scores
        
'''

import os
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

def create_dir(path):
    """
    Create dir if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_forward_data(filename) -> xr.DataArray:
    '''
    Lazily opens the Zarr store on gladefilesystem.
    '''
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset
    

def get_nc_files(base_dir, folder_prefix='%Y-%m-%dT%HZ'):
    """
    Get a list of lists containing paths to NetCDF files in each subdirectory of the base directory,
    sorted by date.

    output = [
        [lead_time1, lead_time2, ...], # initialization time 1
        [lead_time1, lead_time2, ...], # initialization time 2
        [lead_time1, lead_time2, ...], # initialization time 3
        ]
    
    args:
        base_dir: the storage place of individual forecasts
        folder_prefix: the prefix of sub-folders in terms of their datetime info
    
    """
    
    all_files_list = []
    
    # Collect directories and sort them by date
    for parent_dir, sub_dirs, files in os.walk(base_dir):
        
        # Sort directories by date extracted from their names
        sorted_sub_dirs = sorted(sub_dirs, key=lambda x: datetime.strptime(x, folder_prefix))

        # loop through initialization times
        for dir_ini_time in sorted_sub_dirs:

            # get nc file path and glob
            dir_path = os.path.join(parent_dir, dir_ini_time)
            nc_files = sorted(glob(os.path.join(dir_path, '*.nc')))

            # send glob results to a list
            if nc_files:
                all_files_list.append(nc_files)
            else:
                print('folder {} does not have nc files'.format(dir_path))
                raise
    
    return all_files_list

def ds_subset_everything(ds, variables_levels, time_intervals=None):
    """
    Subset a given xarray.Dataset, preserve specific variable/level/time

    args:
        ds: xarray.Dataset
        variables_levels: a dictionary that looks like this
            variables_levels = {
                                'forecast_hour': None,
                                'V500': None,  # Keep all levels
                                'SP': None,
                                't2m': None,
                                'U': [14, 10, 5],  
                                'V': [14, 10, 5], 
                            }
            Leave level as None if (1) keeping all levels or (2) the variable does not have level dim
        time_intervals: a time slice that applies to each variable (optional) 
    """
    # allocate the output xarray.Dataset
    ds_selected = xr.Dataset()

    # loop through the subset info
    for var, levels in variables_levels.items():
        if var in ds:
            if levels is None:
                # keep all level
                ds_selected[var] = ds[var]
            else:
                # subset levels
                ds_selected[var] = ds[var].sel(level=levels)
        else:
            print('variable {} does not exist in the given xarray.Dataset'.format(var))

    # optional time subset
    if time_intervals is not None:
        ds_selected = ds_selected.isel(time=time_intervals)
        
    return ds_selected

def process_file_group(file_list, output_dir, variables_levels, time_intervals=None):
    """
    Process a group of NetCDF files, combining them into a single NetCDF file.

    args:
        file_list: a list of nc filenames
        output_dir: save combined nc to this place
        variables_levels, time_intervals: see `ds_subset_everything`
    """

    # get the folder name of the original, inidividual forecasts,
    subdir_name = file_list[0].split(os.sep)[-2]
    print("Processing subdirectory: {}".format(subdir_name))
    
    # use folder name as output file name
    output_file = os.path.join(output_dir, f'{subdir_name}.nc')
    print('Output name: {}'.format(output_file))
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Skipping {subdir_name} as {output_file} already exists.")
        return

    # Open multiple NetCDF files as a single dataset and subset to specified variables/levels/time
    ds = xr.open_mfdataset(file_list, 
                           combine='by_coords', 
                           preprocess=lambda ds: ds_subset_everything(ds, variables_levels, time_intervals), 
                           parallel=True)
    
    # make sure time coord is 'time'
    ds = ds.rename({'datetime': 'time'})

    # Save the dataset
    ds.to_netcdf(output_file)
    ds.close()

def get_doy_range(doy, days_before, days_after):
    '''
    Get a range of days based on the centered day and before / after days

    args:
        doy: the center day of year
        days_before: number of days before the centered day
        days_after: number of days after the centered day
    '''
    if not (1 <= doy <= 366):
        raise ValueError("DOY must be between 1 and 366")

    # get start and end day-of-year
    start_doy = doy - days_before
    end_doy = doy + days_after

    # generate a list of days
    doys = [(doy+offset) % 366 if (doy+offset) % 366 != 0 else 366 for offset in range(start_doy-doy, end_doy-doy+1)]
    return sorted(doys)

def select_doy_range(ds, doy_range):
    '''
    Slice day-of-year from a xarray.Dataset

    args:
        ds: an xarray.Dataset with 'time' coords
        doy_range: a range of day-of-year int numbers
    '''
    time_doy = ds['time'].dt.dayofyear
    return ds.sel(time=time_doy.isin(doy_range))


def get_filename_prefix_by_radius(day_of_year, day_minus, day_plus, filename_prefix, day_interval=0.5):
    '''
    Create a list of filename prefix based on the datetime information given
    
    args:
        day_of_year: day-of-year
        day_minus: number of days prior to the given day-of-year
        day_plus: number of days after the given day-of-year
        filename_prefix: the string format of generated prefix
        day_interval: intervals between neighbouring days, only 0.5 (12-hour) and 1 (24-hour) are tested 
    '''
    
    assert day_of_year >= 1, 'day of year must >= 1'
    assert day_minus < 0 < day_plus, 'must have "day_minus < 0 < day_plus"'
    
    # datetime reference
    year_ref = 2024
    datetime_obj = datetime(year_ref, 1, 1) + timedelta(days=day_of_year-1)
    
    # day ranges
    day_radius = [day_minus, day_plus + 2*(day_interval)]
    
    prefix_list = []
    
    for delta in np.arange(day_radius[0], day_radius[1], day_interval):

        # get the datetime within radius
        date = datetime_obj + timedelta(days=delta)

        # convert datetime to prefix format
        date_prefix = date.strftime(filename_prefix)
        prefix_list.append(date_prefix)
        
    return prefix_list

def dataset_time_slice(ds, ind_time, file_name):
    '''
    Check and slice the 'time' dimension of a given xarray.Dataset

    args:
        ds: an xarray.Dataset with 'time' coord
        ind_time: an int number that points to the time dim
        file_name: the file_name of ds, used for reporting errors only
    '''
    
    if 'time' in ds.dims:
        ds = ds.isel(time=slice(ind_time, ind_time+1))
        return ds
    else:
        raise ValueError("The dimension 'time' does not exist in the dataset from file {}. Likely caused by an empty file".format(file_name))

def open_datasets_with_preprocessing(list_filename, preprocess_func, index):
    '''
    Open xarray.Dataet with given pre-processing operations

    args:
        list_filename: a list of file names that can be opened by xarray
        preprocess_func: a pre-defined function that can operate on xarray.Dataset
        index: "other inputs" needed by preprocess_func
    '''
    
    datasets = []
    for file in list_filename:
        ds = xr.open_dataset(file, chunks={}, engine='netcdf4')
        ds = preprocess_func(ds, index, file)
        datasets.append(ds)
        
    return xr.concat(datasets, dim='time', combine_attrs='override')

def gaussian_weights_for_temporal_sum(times, center_doy, center_hour, width, max_day=366):
    '''
    Apply Gaussian weighting on a set of timeseries values centered on a given day-of-year and hour-of-day

    args:
        times: xarray time coords that supports times.dt.dayofyear operation
        center_doy: the centering day-of-year
        center_hour: the centering hour-of-day
        width: an int that scales the weights. Tested with 15 day ranges with width=10
    '''

    # compute day differences for all times.values relative to the center
    diff_days = np.abs((times.dt.dayofyear - center_doy).values)
    diff_days = np.minimum(diff_days, (max_day - diff_days))
    diff_hours = np.abs((times.dt.hour - center_hour).values)
    diff_combined = np.sqrt(diff_days**2 + (diff_hours / 24.0)**2)

    # compute weights
    weights = np.exp(-0.5 * (diff_combined / width)**2)
    
    return xr.DataArray(weights, coords=[times], dims=['time'])

def weighted_temporal_sum(ds, center_doy, center_hour, width, var_names):
    '''
    Computed weighted mean on quantities changing over time

    args:
        ds: xarray.Dataset with 'time' coords
        center_doy: the centering day-of-year
        center_hour: the centering hour-of-day
        width: an int that scales the weights. Tested with 15 day ranges with width=10
        var_names: variables exist in the given ds
    '''
    
    results = []
    for var_name in var_names:
        weights = gaussian_weights_for_temporal_sum(ds['time'], center_doy, center_hour, width)
        weights /= weights.sum()
        weighted_data = ds[var_name] * weights
        weighted_mean = weighted_data.sum('time')
        weighted_mean.name = var_name
        results.append(weighted_mean)

    return xr.merge(results)


