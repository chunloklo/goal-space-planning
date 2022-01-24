from copy import copy
from typing import List
import os
import json
import hashlib
from data_io.io.zodb_io import save_config_and_data_zodb, load_data_config_from_id, open_db, close_db, DB_FOLDER, DB_CONFIGS_KEY, DB_DATA_KEY

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
    config_hash = get_config_hash(config, folder_keys)
    return folder, config_hash

def get_config_hash(config: dict, folder_keys=['experiment_name']):
    """Returns the hash of the config for saving data

    Args:
        config (dict): Configuration of the data that you want to save
        folder_keys (list, optional): Additional subfolders that you want to subdivide your results by. Note that this will also remove the key from the dict when hashing. Defaults to ['experiment_name'].

    Returns:
        [str]: hash for the config
    """
    config_clean = copy(config)
    for folder_key in folder_keys:
        config_clean.pop(folder_key)

    file_str = json.dumps(config, sort_keys=True, default=str)
    return hash_string(file_str)

def get_folder_name(config, sub_folder = 'results', folder_keys=['experiment_name']):
    folder = '/'.join([config[folder_key] for folder_key in folder_keys])
    folder = f"{os.getcwd()}/{sub_folder}/{folder}/"
    return folder

# Methods for saving and loading config data with zodb
def save_data_zodb(config, data):
    folder, config_hash = get_folder_and_config_hash(config, sub_folder=DB_FOLDER)
    save_config_and_data_zodb(config_hash, config, data, folder)

def load_data_from_config_zodb(config):
    folder, config_hash = get_folder_and_config_hash(config, sub_folder=DB_FOLDER)
    _, data = load_data_config_from_id(config_hash, folder)
    return data

def load_data_from_config_id_zodb(db_folder: str, config_id: str):
    """Loads data from config_id. This frees us from having to rehash the configs (in case the hashing function changed), and saving performance when loading from a dict.

    Args:
        db_folder (str): folder at which the database is at 
        config_id (str): configuration id

    Returns:
        Any: data
    """
    _, data = load_data_config_from_id(config_id, db_folder)
    return data

class IdLinkedConfig(dict):
    """Behaves like a normal dict, but has an additional method that lets you link a config id with the config
    """
    def __init__(self, config, id):
        super().__init__(config)
        self.id = id

    def get_id(self):
        return self.id

def load_all_db_configs_and_keys(db_folder: str) -> List[IdLinkedConfig]:
    """Returns a list of configs and keys in a given zodb. It returns two lists so there is an explicit mapping between the configs, and config_ids

    Args:
        db_folder (str): [description]

    Returns:
        [List[IdLinkedConfig]]: List of configs with their linked ids
    """
    db, connection, root, lock = open_db(db_folder, create_if_not_exist=False)
    items = list(root[DB_CONFIGS_KEY].items())

    close_db(db, connection, lock)
    return [IdLinkedConfig(item[1], item[0]) for item in items]