from genericpath import isdir
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

from src.analysis.learning_curve import lineplot, plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first
from tqdm import tqdm

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first

def getBest(results):

    best = first(results)
    for r in results:
        a = r.load()[0]
        b = best.load()[0]
        am = np.mean(a)
        bm = np.mean(b)
        if am > bm:
            best = r

    return best


def get_experiment(exp):
    max_returns_results = loadResults(exp, 'max_return.npy')
    return_results = loadResults(exp, 'return.csv')

    # Find some way of getting the best result instead when plotting experiments
    print("This script only plots the first configuration in the experiment")
    first_returns = first(return_results)
    best_returns = getBest(return_results)
    max_returns = first(max_returns_results)

    return best_returns, max_returns

def plot_mean_std(ax, data, label, color, dashed):
    means = np.mean(data, axis=0)
    ste = np.std(data, axis=0) / np.sqrt(data.shape[0])
    lineplot(ax, means, stderr=ste, label=label, color=color, dashed=dashed)


def generatePlot(exp_paths):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        best, max_returns = get_experiment(exp)

        print('best parameters:', exp_path)
        print(best.params)

        fig, ax = plt.subplots()

        alg = exp.agent
        data = best.load()
        #plot_mean_std(ax, data, label=alg, color='red', dashed=False)

        plotBest(best, ax, label=alg, color='red', dashed=False)

        ax.plot(max_returns.load(), label='max returns')

        ax.legend()
        plt.show()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    generatePlot(exp_paths)

    exit()