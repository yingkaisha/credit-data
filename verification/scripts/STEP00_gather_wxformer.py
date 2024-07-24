import os
import sys
import yaml
from glob import glob
from datetime import datetime

import numpy as np
import xarray as xr

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')

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

# ==================== #
model_name = 'wxformer'
# ==================== #

def process_files_concurrently(base_dir, all_files_list, output_dir, variables_levels, time_intervals=None, max_workers=10):
    """
    Process files concurrently using ThreadPoolExecutor.
    """
    # create dir if it does not exist
    vu.create_dir(output_dir)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(vu.process_file_group, file_list, output_dir, 
                                   variables_levels, time_intervals) for file_list in all_files_list]
        for future in futures:
            future.result()  # Wait for all futures to complete

variables_levels = conf[model_name]['verif_variables']

base_dir = conf[model_name]['save_loc_rollout']
output_dir = conf[model_name]['save_loc_gather']
time_intervals = None

# Get list of NetCDF files
all_files_list = vu.get_nc_files(base_dir)

for i in range(verif_ind_start, verif_ind_end):
    i_correct = i
    process_files_concurrently(base_dir, [all_files_list[i_correct]], output_dir, variables_levels, time_intervals)



