import os, sys
from src.utils.formatting import create_file_name , get_folder_name, pushup_metaParameters
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from PyExpUtils.utils.dict import DictPath, flatKeys, get
from src.data_management import zeo_common

def experiment_completed(experiment):
    '''
    Returns True if experiment is yet to be done
    '''
    # Cut the run if already done
    if zeo_common.use_zodb():
        exists = zeo_common.zodb_check_exists(zeo_common.get_db_key(experiment))
    else:
        folder, filename = create_file_name(experiment)
        output_file_name = folder + filename
        exists = os.path.exists(output_file_name + '.pkl')
    return exists

def get_list_pending_experiments(expDescription: ExperimentDescription):
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
        if not experiment_completed(pushup_metaParameters(exp)):
            pending_experiments.append(idx)
    print('')
    return pending_experiments

