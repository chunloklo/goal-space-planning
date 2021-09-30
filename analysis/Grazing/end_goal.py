import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from matplotlib.animation import FuncAnimation
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first
from src.utils import analysis_utils
from action_values import _plot_init, scale_value, get_json_handle, load_experiment_data, get_experiment_name

def generatePlot(json_handle):
    all_data = load_experiment_data(json_handle)

    
    experiment_name = get_experiment_name() + '_end_goal'

    save_path = "./plots/"
    save_folder = os.path.splitext(save_path)[0]

    _, [ax0, ax1] = plt.subplots(2, 1)

    # First run only
    data = all_data["end_goal"]

    num_goals = 4

    # Formatting data
    # removing inner dimension for per episode
    data = data[:, 0, :]
    percentages = []
    for i in range(data.shape[1]):
        # for each episode
        percentages.append([0] * num_goals)
        unique_val, unique_count = np.unique(data[:,i], return_counts=True)
        for val, count in zip(unique_val, unique_count):
            percentages[i][val] = count / sum(unique_count)

    percentages = np.array(percentages)

    ax0.set_ylabel('% goal visits')
    for goal in range (1, 4):
        goal_percentage = percentages[:, goal]

        ax0.plot(analysis_utils.smoothen_runs(goal_percentage, factor=0.5), label=f'{goal}')
    # asdasd
    # for goal in range(1, 4):
        # plt.scatter(np.where(data == goal)[0], [60] * np.where(data == goal)[0].shape[0], label=f'{goal}', s=0.2)


    # plt.legend()

    # This does not handle plotting multiple runs yet. This is to be done

    ax1.set_ylabel('goal schedule')

    # Taking the first run, since all the rewards are the same anyways
    data = all_data['goal_rewards'][0][0]
    for goal in range(3):
        ax1.plot(data[:, goal], label=f'goal {goal + 1}')

    plt.legend()

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