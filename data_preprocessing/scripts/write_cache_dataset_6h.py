import os
import sys
import yaml
import zarr
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
    
# Get variable names
varnames = list(conf['cache'].keys())
varnames = varnames[:-5] # remove save_loc and others
varname_surf = list(set(varnames) - set(['U', 'V', 'T', 'Q']))

# Get zscore values
zscore_mean = xr.open_dataset(conf['cache']['mean_loc'])
zscore_std = xr.open_dataset(conf['cache']['std_loc'])

# -------------------------------------------------------------------------------- #
# chunking and compression settings

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

chunk_size_3d = dict(chunks=(conf['zarr_opt']['chunk_size_3d']['time'],
                                 conf['zarr_opt']['chunk_size_3d']['latitude'],
                                 conf['zarr_opt']['chunk_size_3d']['longitude']))

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d']['time'],
                                 conf['zarr_opt']['chunk_size_4d']['level'],
                                 conf['zarr_opt']['chunk_size_4d']['latitude'],
                                 conf['zarr_opt']['chunk_size_4d']['longitude']))

dict_encoding = {}

for i_var, var in enumerate(varnames):
    if var in varname_surf:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
# -------------------------------------------------------------------------------- #

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

    # chunk data
    if var in varname_surf:
        ds_base[var] = ds_base[var].chunk(conf['zarr_opt']['chunk_size_3d'])
    else:
        ds_base[var] = ds_base[var].chunk(conf['zarr_opt']['chunk_size_4d'])

save_name = conf['cache']['save_loc'] + conf['cache']['prefix'] + '_{}.zarr'.format(year)
print('Save to {}'.format(save_name))

ds_base.to_zarr(save_name, mode="w", consolidated=True, compute=True, encoding=dict_encoding)

# save to zarr using dask client
# with Client(n_workers=100, threads_per_worker=2) as client:
#     ds_base.to_zarr(save_name,
#                     mode="w",
#                     consolidated=True,
#                     compute=True)

