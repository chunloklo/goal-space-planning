'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from src.utils.json_handling import get_param_iterable_runs, get_sorted_dict
from src.utils import analysis_utils 
from src.utils.formatting import create_folder
from src.utils.file_handling import get_files_recursively
import argparse
import numpy as np
from src.analysis.plot_utils import load_experiment_data, get_json_handle, get_x_range

parser = argparse.ArgumentParser(description='Parallizable experiment run file')
parser.add_argument('-j', '--json-path', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('-nl', '--no-legend', action='store_true', help='Flag to not show legend')
parser.add_argument('-c', '--cumulative', action='store_true', help='Whether to show the cumulative return' )

args = parser.parse_args()
show_legend = not args.no_legend
json_files = args.json_path

json_files = get_files_recursively(json_files)
print(json_files)
json_handles = [get_sorted_dict(j) for j in json_files]
print(json_handles)

# key_to_plot = 'return_data' # the key to plot the data

# fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)

# plt.figure()
fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)

def smoothingAverage(arr, p=0.5):
    m = 0
    for i, a in enumerate(arr):
        if i == 0:
            m = a
            yield m
        else:
            m = p * m + (1 - p) * a
            yield m

for en, js in enumerate(json_handles):
    data = load_experiment_data(js)
    # print(data.keys())
    data = data["reward"]
    data = data[0, 0, :]

    print(data.shape)

    # print(data)
    x_range = get_x_range(0, data.shape[0], 1)
    # print(list(x_range))
    for i in range(len(data)):
        if i > 0:
            data[i] = data[i] + data[i - 1]
    # data = list(smoothingAverage(data))
    # print(js.keys())
    label_str = ''
    if 'skip_prob' in js['metaParameters']:
        label_str = f'skip_prob: {js["metaParameters"]["skip_prob"]}'
    axs.plot(x_range, data, label=label_str)


    # plot(axs, data = data)

    # run, param , data, max_returns = analysis_utils.find_best(js, data = 'return')
    # label_str = f'{param["agent"]} + {param.get("skip_alg", "")}'
    # plot(axs, data = data[key_to_plot], label = f"{label_str}")
    # if en == 0:
    #     if args.cumulative:
    #         cumulative(max_returns[0,0,:])
    #         # axs.plot(max_returns[0,0,:], label='max return')
    #     else:
    #         axs.plot(max_returns[0,0,:], label='max return')
    # #print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])
# asdasd

# axs.set_ylim([-600, 110])
# axs.set_xlim([3250, 3750])
# axs.set_ylim([-100, 100])
axs.legend()
axs.spines['top'].set_visible(False)
# if show_legend:
#     axs.set_title(f'{key_to_plot} accuracy')
#     axs.legend()

axs.spines['right'].set_visible(False)
axs.tick_params(axis='both', which='major', labelsize=8)
axs.tick_params(axis='both', which='minor', labelsize=8)
axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
get_experiment_name = input("Give the input for experiment name: ")
plt.savefig(f'{foldername}/reward_{get_experiment_name}.png', dpi = 300)