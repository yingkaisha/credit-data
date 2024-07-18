import os
import sys
import yaml
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
verif_hour = conf[model_name]['verif_hour']
longest_hour = conf[model_name]['longest_hour']
leads_do = np.arange(verif_hour, longest_hour+verif_hour, verif_hour)
leads_do = leads_do - 1 # -1 for Python indexing
print('Verifying lead times: {}'.format(leads_do))
# ====================== #

# radius of days to compute climatology
day_minus = -15; day_plus = 15

# Define the date format
filename_prefix = "-%m-%dT%HZ"

# path and file info
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))
path_campaign = conf[model_name]['save_loc_clim']
                     
# Variable names
var_names = ['U500', 'Z500', 'Q500', 'T500', 'V500', 'SP', 't2m']

# manual input bad files in '/glade/campaign/cisl/aiml/gathered/'
# provide replacements in '/glade/campaign/cisl/aiml/ksha/CREDIT/gathered/'
# correct file info and rerun climo days/leads that touchs the bad files
filename_OURS_backup = sorted(glob('/glade/campaign/cisl/aiml/ksha/CREDIT/gathered/*.nc'))

ind_bad = [206, 209, 211, 215, 360, 390, 400]
filename_bad = []

for i, i_bad in enumerate(ind_bad):
    file_old = filename_OURS[i_bad]
    file_new = filename_OURS_backup[i]

    if os.path.basename(file_old) == os.path.basename(file_new):
        filename_bad.append(file_new)
        filename_OURS[i_bad] = filename_OURS_backup[i]
    else:
        print('Replacement of bad file {} not found'.format(file_old))
        raise

# Batch processing by day of year and lead time
for day_of_year in range(verif_ind_start, verif_ind_end):

    prefix_pick = vu.get_filename_prefix_by_radius(day_of_year, day_minus, day_plus, filename_prefix, 0.5)
    
    filtered_files = [fn for fn in filename_OURS if any(date in fn for date in prefix_pick)]

    flag_rerun = False
    for name_bad in filename_bad:
        if name_bad in filtered_files:
            flag_rerun = True
    
    #center_doy_gw = day_of_year
    
    for lead_time in leads_do:
        
        # Adjusting center hour for lead time
        lead_hour_of_day = (lead_time) % 24  
        day_add_lead = day_of_year + int(lead_time/24)
        
        output_path = '{}medium_boy_DOY{:03}_LEAD{:03}.nc'.format(path_campaign, day_add_lead, lead_time+1)
        
        flag_exist = os.path.exists(output_path)

        if flag_exist is False:
            #if (flag_exist is False) or (flag_rerun):
            #if flag_rerun:
            
            print('Missing: {}'.format(output_path))
            print('Processing day {}, lead time {}'.format(day_add_lead, lead_time+1))
            
            ds = vu.open_datasets_with_preprocessing(filtered_files, vu.dataset_time_slice, lead_time)
            
            width = 10
            weighted_mean = vu.weighted_temporal_sum(ds, day_add_lead, lead_hour_of_day, width, var_names)
            weighted_mean = weighted_mean.compute()
            
            weighted_mean.to_netcdf(output_path)
            print('Save to {}'.format(output_path))
            print('Finished processing day {}, lead time {}'.format(day_add_lead, lead_time))
        # else:
        #     print('Skip {}'.format(output_path))

