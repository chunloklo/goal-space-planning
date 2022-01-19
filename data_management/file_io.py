from copy import copy
import pickle as pkl
from typing import Dict
import os
import json
import hashlib # need to start hasing filenames to shorten them

# Simple wrapper for loading and saving pkl objects
def load_pkl(filename):
    with open(filename, 'rb') as fil:
        data = pkl.load(fil)
    return data

def save_pkl(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as fil:
        pkl.dump(obj, fil)

def hash_string(name):
    return hashlib.sha256(name.encode()).hexdigest()

def get_folder_and_config_hash(config: dict, sub_folder = 'results', folder_keys=['experiment_name']):
    """Returns the folder and hash of the config for saving data

    Args:
        config (dict): Configuration of the data that you want to save
        sub_folder (str, optional): Subfolder you want to save your results in. Defaults to 'results'.
        folder_keys (list, optional): Additional subfolders that you want to subdivide your results by. Note that this will also remove the key from the dict when hashing. Defaults to ['experiment_name'].

    Returns:
        [type]: [description]
    """
    folder = get_folder_name(config, sub_folder, folder_keys)
    
    config_clean = copy.copy(config)
    for folder_key in folder_keys:
        config_clean.pop(folder_key)

    file_str = json.dumps(config, sort_keys=True, default=str)
    return folder, hash_string(file_str)

def get_folder_name(config, sub_folder = 'results', folder_keys=['experiment_name']):
    folder = '/'.join([config[folder_key] for folder_key in folder_keys])
    folder = f"{os.getcwd()}/{sub_folder}/{folder}/"
    return folder

def save_data(experiment: Dict, data: Dict):
        folder, filename = get_folder_and_config_hash(experiment)
        output_file_name = folder + filename
        save_pkl(data, output_file_name + '.pkl')

def load_data(experiment: Dict):
    folder, file = get_folder_and_config_hash(experiment)
    filename = folder + file + '.pkl'
    data = load_pkl(filename)
    return data
