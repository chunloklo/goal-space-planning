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
import argparse

parser = argparse.ArgumentParser(description='Parallizable experiment run file')
parser.add_argument('-j', '--json-path', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('-nl', '--no-legend', action='store_true', help='Flag to not show legend')

args = parser.parse_args()
show_legend = not args.no_legend
json_files = args.json_path

json_files = get_files_recursively(json_files)

json_handles = [get_sorted_dict(j) for j in json_files]

agent_colors={
    "Q_Tabular": 'red',
    "Dyna_Tab": 'yellow',
    "Dynaesp_Tab": 'red',
    "Dynaqp_Tab": 'green',
    "Q_Linear": 'red',
    "Option_Q_Tab": 'cyan',
    "Q_OptionIntra_Tab": 'blue',
    "Dyna_Optionqp_Tab": 'purple',
    "Option_Given_Q_Tab": 'magenta',
    "Dyna_Option_Givenqp_Tab": 'orange',
    "DynaQP_OptionIntra_Tab": 'purple',
    "DynaESP_OptionIntra_Tab": 'blue',
    "OptionPlanning_Tab": '#66CCEE',
    "DynaOptions_Tab": '#4287f5',
}


def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def  plot(ax , data, label = None , color = None):
    mean = data['mean']

    # mean = smoothen_runs(mean)
    stderr = data['stderr']
    if color is not None:
        base, = ax.plot(mean, label = label, linewidth = 2, color = color)
    else:
        base, = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.4  )

key_to_plot = 'return_data' # the key to plot the data

fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)
for en, js in enumerate(json_handles):
    run, param , data, max_returns = analysis_utils.find_best(js, data = 'return')
    label_str = f'{param["agent"]} + {param.get("behaviour_alg", "")}'
    print(param)
    plot(axs, data = data[key_to_plot], label = f"{label_str}")
    if en == 0:
        axs.plot(max_returns[0,0,:], label='max return')
    #print(key_to_plot, data[key_to_plot]['mean'][-5:], data[key_to_plot]['stderr'][-5:])


# axs.set_ylim([-300, 110])
axs.set_ylim([-100, 200])
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


