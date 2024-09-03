import os
import sys
import yaml
import numpy as np
import xarray as xr
from glob import glob

sys.path.insert(0, os.path.realpath('../../libs/'))
import preprocess_utils as pu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='varname')
args = vars(parser.parse_args())

year = int(args['year'])

config_name = os.path.realpath('../data_config_6h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# Get zscore values
zscore_mean = xr.open_dataset(conf['cache']['mean_loc'])
zscore_std = xr.open_dataset(conf['cache']['std_loc'])

# Get variable names
varnames = list(conf['cache'].keys())
varnames = varnames[:-5] # remove save_loc and others
varname_surf = list(set(varnames) - set(['U', 'V', 'T', 'Q']))

# ---------------------------------------------------------------------------------- #
# chunking and compression
compress = dict(zlib=True, complevel=1)

chunk_size_3d = dict(chunksizes=(conf['zarr_opt']['chunk_size_3d']['time'],
                                 conf['zarr_opt']['chunk_size_3d']['latitude'],
                                 conf['zarr_opt']['chunk_size_3d']['longitude']))

chunk_size_4d = dict(chunksizes=(conf['zarr_opt']['chunk_size_4d']['time'],
                                 conf['zarr_opt']['chunk_size_4d']['level'],
                                 conf['zarr_opt']['chunk_size_4d']['latitude'],
                                 conf['zarr_opt']['chunk_size_4d']['longitude']))

dict_encoding = {}

for i_var, var in enumerate(varnames):
    if var in varname_surf:
        dict_encoding[var] = {**compress, **chunk_size_3d}
    else:
        dict_encoding[var] = {**compress, **chunk_size_4d}
# ---------------------------------------------------------------------------------- #

for i_var, var in enumerate(varnames):
    
    filenames = glob(conf['cache'][var])
    fn = [fn for fn in filenames if str(year) in fn][0]

    ds_original = pu.get_forward_data(fn)
    ds_var = ds_original[var]

    ds_zscore_var = (ds_var - zscore_mean[var])/zscore_std[var]
    ds_zscore_var = ds_zscore_var.to_dataset()
    
    if i_var == 0:
        ds_base = ds_zscore_var
    else:
        ds_base = ds_base.merge(ds_zscore_var)

save_name = conf['cache']['save_loc'] + conf['cache']['prefix'] + '_{}.nc'.format(year)

print('Save to {}'.format(save_name))

# ds_base.to_zarr(save_name, mode="w", consolidated=True, compute=True)
ds_base.to_netcdf(save_name, format='NETCDF4', encoding=dict_encoding)

