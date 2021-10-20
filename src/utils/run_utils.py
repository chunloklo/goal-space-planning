import os, sys
from src.utils.formatting import create_file_name , get_folder_name, pushup_metaParameters
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from PyExpUtils.utils.dict import DictPath, flatKeys, get

def experiment_completed(experiment):
    '''
    Returns True if experiment is yet to be done
    '''
    experiment = pushup_metaParameters(experiment)

    folder, filename = create_file_name(experiment)
    output_file_name = folder + filename
    # Cut the run if already done
    if os.path.exists(output_file_name + '.pkl'):
        return True
    else:
        return False

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
        if not experiment_completed(exp):
            pending_experiments.append(idx)
    print('')
    print(f'Num experiments left: {len(pending_experiments)}/{experiment_no}')
    return pending_experiments

