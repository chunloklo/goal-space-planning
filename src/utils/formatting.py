'''
This file will contain code pertaining to teh naming issues with difefrent variant
'''
import os
import time
import hashlib # need to start hasing filenames to shorten them
import copy

def hash_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def float_to_string(number, precision=10):
    return '{0:.{prec}f}'.format(
        number, prec=precision,
    ).rstrip('0').rstrip('.') or '0'



def deserialize_list_to_name(l):
    '''
    For now asusu,ing list is only of int, character and floats
    '''
    str_str = ''
    for ele in l:
        if isinstance(ele, float):
            str_str += f"{float_to_string(ele)}_"
        else:
            str_str += f"{ele}_"
    return str_str[:-1] # remove the last _


def deseriazlie_dict_to_name(d):
    ''' takes input  as dictionaruy and returns a name for the same (for now assumes a specificaiton)
    # FIXME : ignores lists inside dictionaries for now'''
    keys = list(d.keys())
    keys = sorted(keys)
    file_name = ''
    for k in keys:
        if isinstance(d[k], float):
            file_name += f"{k}_{float_to_string(d[k])}_"
        else:
            if isinstance(d[k], list):
                str_ = deserialize_list_to_name(d[k])
                file_name += f"{k}_{str_}_"
            else:
                file_name += f"{k}_{d[k]}_"
    return file_name[:-1]

# This method "pushes up" the metaParameters from
# within a key to the high level dict. This is so 
# we could transform a params dict between the new format
# with metaParamaters as a key to the old format which 
# didn't have the metaParameters key.
def pushup_metaParameters(experiment):
    experiment = copy.deepcopy(experiment)
    metaParams = experiment['metaParameters']
    del experiment['metaParameters']
    for k in metaParams.keys():
        experiment[k] = metaParams[k]
    return experiment

# [chunlok 2022-01-17] There's additional work here needed to make sure that this hashing gives us reasonable results
# for non-string/int/float keys. This seems to be working fine though, so we'll just tackle that when the time comes.
def create_file_name(experiment: dict, sub_folder = 'results'):
    # Storing data in folder with experiment_name
    folder = f"{experiment['experiment_name']}"
    keys = list(experiment.keys())
    keys.remove('experiment_name')

    # make filename
    keys = sorted(keys)
    file_name = ''
    for k in keys:
        if isinstance(experiment[k], float):
            file_name += f"{k}_{float_to_string(experiment[k])}_"
        elif isinstance(experiment[k], dict):
            file_name += f"{k}_{deseriazlie_dict_to_name(experiment[k])}_"
        else:
            if isinstance(experiment[k], list):
                raise NotImplementedError('List is not supported when hashing dictionary names')
                continue
            file_name += f"{k}_{experiment[k]}_"
    
    file_name = file_name[:-1] # give only the name
    folder = f'{os.getcwd()}/{sub_folder}/{folder}/'

    # return folder, file_name
    return folder, hash_name(file_name)

def get_folder_name(experiments, sub_folder = 'results'):
    folder = f"{experiments['experiment_name']}/{experiments['agent']}/{experiments['problem']}"
    folder = f"{os.getcwd()}/{sub_folder}/{folder}/"
    return folder

def pretty_print_experiment(experiment):
    for k in experiment.keys():
        print(f"{k} : {experiment[k]}")


def create_folder(folder):
    os.makedirs(folder, exist_ok=True)