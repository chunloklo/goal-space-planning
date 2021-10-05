import os
import sys
sys.path.append(os.getcwd())

from genericpath import isdir
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from matplotlib.animation import FuncAnimation


from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first
from tqdm import tqdm

# Mass taking imports from process_data for now
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from src.utils import analysis_utils
from action_values import _plot_init, scale_value, get_json_handle, load_experiment_data, get_experiment_name

def generatePlot(json_handle):

    # data = load_experiment_data(exp_path, file_name)

    experiment_name = get_experiment_name() + '_action_selected'

    save_path = "./plots/"
    save_folder = os.path.splitext(save_path)[0]

    if (not os.path.isdir(save_folder)):
        os.makedirs(save_folder)


    data = load_experiment_data(json_handle)

    # print(return_data)
    # Processing data here so the dimensions are correct
    data = data["action_selected"]


    print(data.shape)
    data = data[:, 0, :]
    option_percentages = []
    option_std = []

    num_actions = 4

    runs = data.shape[0]
    episodes = data.shape[1]
    for episode in range(episodes):
        percentages = np.zeros(runs)
        for run in range(runs):
            actions = np.array(data[run, episode])
            total = actions.shape[0]
            action_total = np.where(actions < num_actions)[0].shape[0]
            option_total = np.where(actions >= num_actions)[0].shape[0]
            percentages[run] = option_total / total
        option_percentages.append(np.mean(percentages))
        option_std.append(np.std(percentages))

    option_percentages = np.array(option_percentages)
    option_std = np.array(option_std)
    
    plt.figure()
    plt.plot(option_percentages)
    plt.fill_between(range(len(option_percentages)), option_percentages + option_std, option_percentages - option_std, alpha=0.1)
    plt.ylabel('% option selected')

    plt.savefig(save_folder + f'/{experiment_name}.pdf', dpi = 300)
    plt.savefig(save_folder + f'/{experiment_name}.png', dpi = 300)



if __name__ == "__main__":

    # read the arguments etc
    if len(sys.argv) < 2:
        print("usage : python analysis/process_data.py <list of json files")
        exit()


    json_handle = get_json_handle()

    # Only use the first handle for now?
    generatePlot(json_handle)

    exit()