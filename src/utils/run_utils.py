import os, sys
from typing import Dict
from src.utils.formatting import create_file_name , get_folder_name, pushup_metaParameters
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from PyExpUtils.utils.dict import DictPath, flatKeys, get
import traceback
from src.utils import analysis_utils
from src.data_management import zeo_common

class InvalidRunException(Exception):
    pass

def cleanup_files(experiment: Dict):
    if zeo_common.use_zodb():
        data_key = zeo_common.get_db_key(experiment)
        error_key = zeo_common.get_db_key(experiment) + '_ERROR'
        zeo_common.zodb_remove(error_key)
        zeo_common.zodb_remove(data_key)
    else:
        folder, filename = create_file_name(experiment)
        output_file_name = folder + filename

        if os.path.exists(output_file_name + '.pkl'):
            os.remove(output_file_name + '.pkl')
        if os.path.exists(output_file_name + '.err'):
            os.remove(output_file_name + '.err')
        pass

def save_error(experiment: Dict, exception: Exception):

    err_text = str(exception) + '\n'
    err_text += "".join(traceback.TracebackException.from_exception(exception).format()) + '\n'
        
    if zeo_common.use_zodb():
        error_key = zeo_common.get_db_key(experiment) + '_ERROR'
        zeo_common.zodb_saver(err_text, error_key)
    else:
        folder, filename = create_file_name(experiment)
        output_file_name = folder + filename

        file_name = output_file_name + '.err'

        os.makedirs(os.path.dirname(folder), exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(err_text)

def save_data(experiment: Dict, data: Dict):
    if zeo_common.use_zodb():
        zeo_common.zodb_saver(data, zeo_common.get_db_key(experiment))
    else:
        folder, filename = create_file_name(experiment)
        output_file_name = folder + filename
        analysis_utils.pkl_saver(data, output_file_name + '.pkl')
    pass

def load_data(experiment: Dict):
    folder, file = create_file_name(experiment)
    filename = folder + file + '.pkl'
    data = analysis_utils.pkl_loader(filename)
    return data

def experiment_completed(experiment, include_errored=True):
    '''
    Returns True if experiment is yet to be done
    '''
    experiment = pushup_metaParameters(experiment)

    if zeo_common.use_zodb():
        exists = zeo_common.zodb_check_exists(experiment)
        return exists

    folder, filename = create_file_name(experiment)
    output_file_name = folder + filename
    # Cut the run if already done
    if os.path.exists(output_file_name + '.pkl'):
        return True
    elif include_errored and os.path.exists(output_file_name + '.err'):
        return True
    else:
        folder, filename = create_file_name(experiment)
        output_file_name = folder + filename
        exists = os.path.exists(output_file_name + '.pkl')
    return exists

def get_list_pending_experiments(expDescription: ExperimentDescription, exclude_errored=True):
    '''
    Inputs : ExperimentModel
    Returns : Index of pending experiments
    '''
    # given a list of expeiments
    pending_experiments = []
    experiment_no = expDescription.numPermutations()
    print(experiment_no)

    for idx in range(experiment_no):
        exp = expDescription.getPermutation(idx)
        print(f'Checking [{idx}/{experiment_no}]\r' , end = "")
        if not experiment_completed(exp, exclude_errored):
            pending_experiments.append(idx)
    print('')
    print(f'Num experiments left: {len(pending_experiments)}/{experiment_no}')
    return pending_experiments

