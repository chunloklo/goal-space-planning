'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from src.utils.json_handling import get_sorted_dict
from src.utils import analysis_utils 
from src.utils.formatting import create_folder
from src.utils.file_handling import get_files_recursively
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parallizable experiment run file')
parser.add_argument('--json', '-j', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
args = parser.parse_args()


experiment_list = args.json

json_files = get_files_recursively(experiment_list)
json_handles = [get_sorted_dict(j) for j in json_files]

sens_key = 'kappa'

for handle in json_handles:
    # print(handle)
    key_params = handle['metaParameters'][sens_key]
    key_params = list(set(key_params))
    key_params.sort()
    # print(key_params)

    perf_list = []

    for key_param in key_params:
        specific_handle = copy.copy(handle)
        specific_handle['metaParameters'][sens_key] = [key_param]
        run, param, data, max_returns = analysis_utils.find_best(specific_handle, data = 'return')
        if (param == None):
            print(f'{sens_key}: {key_param}')
        else:
            print(f'{sens_key}: {key_param}, alpha: {param["alpha"]} alg: {param["behaviour_alg"]}')
        # print(np.mean(run['mean']))


        perf_list.append(np.mean(run.get('mean', float('-inf'))))
        # perf_list.
        # print()
        # print(param)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(key_params, perf_list)
    foldername = './plots'
    get_experiment_name = 'test_sens'
    ax.set_xscale('log')
    plt.savefig(f'{foldername}/sens_plot{get_experiment_name}.png', dpi = 300)

    sdfsdf

    analysis_utils.get_param_iterable_runs
    # runs, params, keys, best_data = analysis_utils.find_best_key(js, key= d_keys, data = 'return_data')
asdasd

fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)
for en, js in enumerate(json_handles):
    run, key_param , data, max_returns = analysis_utils.find_best(js, data = 'return')
    label_str = f'{key_param["agent"]} + {key_param.get("behaviour_alg", "")}'
    print(key_param)
    plot(axs, data = data[key_to_plot], label = f"{label_str}")
    if en == 0:
        axs.plot(max_returns[0,0,:], label='max return')
    #print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])


axs.set_ylim([-300, 110])
axs.spines['top'].set_visible(False)
if show_legend:
    axs.set_title(f'{key_to_plot} accuracy')
    axs.legend()

axs.spines['right'].set_visible(False)
axs.tick_params(axis='both', which='major', labelsize=8)
axs.tick_params(axis='both', which='minor', labelsize=8)
axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
get_experiment_name = input("Give the input for experiment name: ")
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}.pdf', dpi = 300)
plt.savefig(f'{foldername}/learning_curve_{get_experiment_name}.png', dpi = 300)


