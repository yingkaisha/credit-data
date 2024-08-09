import os
import sys
import yaml
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import preprocess_utils as pu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('varname', help='varname')
args = vars(parser.parse_args())

varname = args['varname']

config_name = os.path.realpath('../data_config_6h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

pu.zscore_by_var(conf, varname)

