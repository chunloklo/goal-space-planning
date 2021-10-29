import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
import ZEO
import os
import time
import ZODB
import ZEO
import transaction
import pathlib
from src.utils import analysis_utils, formatting
import persistent.dict

ZEO_PORT = 8200

def start_zeo_server():
    db_folder = f'{os.getcwd()}/results_db/'

    if not os.path.exists(db_folder):
        time.sleep(2)
        try:
            os.makedirs(db_folder)
        except:
            pass

    address, stop = ZEO.server(db_folder + 'zeo_experiments.fs', port=ZEO_PORT)

    return address, stop

def open_zeo_client():
    client = ZEO.client(ZEO_PORT, cache_size = 0)
    db = ZODB.DB(client)
    connection = db.open()
    root = connection.root()
    return root

def zodb_check_exists(experiment: ExperimentDescription):
    db_key = get_db_key(experiment)
    root = open_zeo_client()
    if root.get('experiments', None) is None:
        return False

    if root['experiments'].get(db_key, None) is None:
        return False

    return True

MAX_ATTEMPTS = 10
def zodb_remove(file_name: str):
    root = open_zeo_client()

    max_attempts = MAX_ATTEMPTS
    attempts = 0
    while True:
        try:
            with transaction.manager:
                transaction.begin()
                if root.get('experiments', None) is None:
                    root['experiments'] = persistent.dict.PersistentDict()
                
                if file_name in root['experiments']:
                    del root['experiments'][file_name]
                transaction.commit()
        except transaction.interfaces.TransientError:
            print('transient error')
            attempts += 1
            if attempts == max_attempts:
                raise
        else:
            break

def zodb_saver(obj, file_name):
    root = open_zeo_client()

    max_attempts = MAX_ATTEMPTS
    attempts = 0
    while True:
        try:
            with transaction.manager:
                transaction.begin()
                if root.get('experiments', None) is None:
                    root['experiments'] = persistent.dict.PersistentDict()
                root['experiments'][file_name] = obj
                transaction.commit()
        except transaction.interfaces.TransientError:
            print('transient error')
            attempts += 1
            if attempts == max_attempts:
                raise
        else:
            break
            
def zodb_loader(file_name):
    root = open_zeo_client()
    return root['experiments'][file_name]


def get_db_key(experiment):
    '''
    We will make folder names with agent and problem and then appends the rest fo the config
    return the folder and filename
    '''
    folder = f"{experiment['agent']}/{experiment['problem']}"
    keys = list(experiment.keys())
    keys.remove("agent")
    keys.remove("problem")
    # make filename
    keys = sorted(keys)
    file_name = ''
    for k in keys:
        if isinstance(experiment[k], float):
            file_name += f"{k}_{formatting.float_to_string(experiment[k])}_"
        elif isinstance(experiment[k], dict):
            file_name += f"{k}_{formatting.deseriazlie_dict_to_name(experiment[k])}_"
        else:
            if isinstance(experiment[k], list):
                continue
            file_name += f"{k}_{experiment[k]}_"
    file_name = file_name[:-1] # give only the name

    # return folder, file_name
    return f'{folder}/{formatting.hash_name(file_name)}'

def use_zodb():
    try:
        return os.environ["USE_ZODB"] == 'TRUE'
    except:
        return False