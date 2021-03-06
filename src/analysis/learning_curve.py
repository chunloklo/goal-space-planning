import numpy as np
from PyExpUtils.utils.generator import windowAverage

def smoothingAverage(arr, p=0.5):
    m = 0
    for i, a in enumerate(arr):
        if i == 0:
            m = a
            yield m
        else:
            m = p * m + (1 - p) * a
            yield m

def confidenceInterval(mean, stderr):
    return (mean - 2.0 * stderr, mean + 2.0 * stderr)


def plotBest(best, ax, window=1, smoothing=0, color=None, label=None, alpha=0.4, alphaMain=1, labelParams=None, dashed=False):
    label = label if label is not None else best.exp.agent

    params = ''
    if labelParams is not None:
        l = [f'{key}-{best.params[key]}' for key in labelParams]
        params = ' ' + ' '.join(l)


    mean = best.mean()
    try:
        ste = best.stderr()
    except:
        ste = [0]
        pass

    if len(mean.shape) == 1:
        mean = np.reshape(mean, (-1, 1))
        ste = np.reshape(ste, (-1, 1))
    if type(label) != list:
        label = [label] * mean.shape[1]

    if type(dashed) != list:
        dashed = [dashed] * mean.shape[1]


    for i in range(mean.shape[1]):
        lineplot(ax, mean[:, i], stderr=ste[:, i], smoothing=smoothing, window=window, color=colors[label[i]], label=label[i] + params, alpha=alpha, alphaMain=alphaMain, dashed=dashed[i])

def plotIndividualAndMean(best, ax, window=1, smoothing=0, color=None, label=None, alpha=0.4, alphaMain=1, labelParams=None, dashed=False):
    label = label if label is not None else best.exp.agent

    params = ''
    if labelParams is not None:
        l = [f'{key}-{best.params[key]}' for key in labelParams]
        params = ' ' + ' '.join(l)

    mean = best.mean()
    data = best.load()

    for i in range(data.shape[0]):
        lineplot(ax, data[i, :], smoothing=smoothing, window=window, color=colors[label], alphaMain=0.1, dashed=dashed)

    lineplot(ax, mean, smoothing=smoothing, window=window, color=colors[label], label=label + params, alpha=alpha, alphaMain=alphaMain, dashed=dashed)

def lineplot(ax, mean, window=1, smoothing=0, stderr=None, color=None, label=None, alpha=0.4, alphaMain=1, dashed=None):
    if dashed:
        dashes = ':'
    else:
        dashes = None

    if window > 1:
        mean = windowAverage(mean, window)

    if window > 1 and stderr is not None:
        stderr = windowAverage(stderr, window)

    mean = np.array(list(smoothingAverage(mean, smoothing)))



    ax.plot(mean, linestyle=dashes, label=label, color=color, alpha=alphaMain, linewidth=2)
    if stderr is not None:
        stderr = np.array(list(smoothingAverage(stderr, smoothing)))
        (low_ci, high_ci) = confidenceInterval(mean, stderr)
        ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=color, alpha=alpha * alphaMain)

    if (label != None):
        ax.legend()

colors={
    "Q_Tabular": 'red',
    "Dyna_Tab_Dist": 'blue',
    "Dyna_Tab": 'yellow',
    "Dynaqp_Tab": 'green',
    "Q_Linear": 'red',
    "Option_Q_Tab": 'cyan',
    "Dyna_Optionqp_Tab": 'purple',
    "Option_Given_Q_Tab": 'magenta',
    "Dyna_Option_Givenqp_Tab": 'orange',
    "DynaQP_OptionIntra_Tab": 'purple',
    "OptionPlanning_Tab": '#66CCEE'
}