'''
This file will take in json files and process the data across different runs to store the summary
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from src.utils import analysis_utils
from src.utils.file_handling import get_files_recursively

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/process_data.py <list of json files")
    exit()

json_files = sys.argv[1:] # all the json files
json_files = get_files_recursively(json_files)

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def process_runs(runs):
    # get mean and std

    runs = runs.squeeze()
    if (len(runs.shape) >= 2):
        mean = np.mean(runs, axis = 0)

        stderr = np.std(runs , axis = 0) / np.sqrt(runs.shape[0])
    else:
        print('Only one run. Mean is just that squeezed run')
        mean = runs
        stderr = np.zeros(mean.shape)

    return mean , stderr

# currentl doesnt not handle frames
def process_data_interface(json_handles):
    for js in json_handles:
        runs = []
        iterables = get_param_iterable_runs(js)
        
        for i in iterables:
            folder, file = create_file_name(i, 'processed', False)
            create_folder(folder) # make the folder before saving the file
            filename = folder + file + '.pcsd'
            # check if file exists
            print(filename)
            if os.path.exists(filename):
                print("Processed")

            else:
                return_data, max_returns = analysis_utils.load_different_runs(i)
                mean_return_data, stderr_return_data = process_runs(return_data)
                return_data = {
                    'mean' : mean_return_data,
                    'stderr' : stderr_return_data
                    
                }
                # save the things
                analysis_utils.pkl_saver({
                        'return_data' : return_data,
                        'max_returns': max_returns
                    }, filename)

if __name__ == '__main__':
    process_data_interface(json_handles)