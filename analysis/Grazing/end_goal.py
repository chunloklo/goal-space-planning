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

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        results = loadResults(exp, 'end_goal.npy')

        # This does not handle plotting multiple runs yet. This is to be done
        data = None
        for r in results:
            data = r.load()

        print(data.shape)

        fig = plt.figure()

        for goal in range(1, 4):
            print(np.where(data == goal))
            plt.scatter(np.where(data == goal)[0], [60] * np.where(data == goal)[0].shape[0], label=f'{goal}')
        # plt.legend()


        results = loadResults(exp, 'goal_rewards.npy')

        # This does not handle plotting multiple runs yet. This is to be done
        data = None
        for r in results:
            data = r.load()

        print(data.shape)

        # fig = plt.figure()

        for goal in range(3):
            plt.plot(data[:, goal], label=f'{goal + 1}')

        plt.legend()


        plt.show()

if __name__ == "__main__":
    # f, axes = plt.subplots(1)
    axes = None
    bounds = []

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths, bounds)

    # plt.show()
    exit()

    save_path = 'experiments/exp/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/learning-curve.png', dpi=100)
