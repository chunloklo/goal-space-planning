import numpy as np
from typing import Any, Dict, Optional
from PyExpUtils.results.backends.backend import ResultList, DuckResult
from PyExpUtils.results.results import getBest

def confidenceInterval(mean, stderr, mult):
    return (mean - mult * stderr, mean + mult * stderr)

def buildOptions(options: Optional[Dict[str, Any]]):
    options = options if options is not None else {}
    out_options = {}

    if options.get('dashed'):
        out_options['style'] = '--'
    elif options.get('dotted'):
        out_options['style'] = ':'
    else:
        out_options['style'] = None

    out_options['alpha_main'] = options.get('alpha_main', 1.0)
    out_options['alpha'] = options.get('alpha', 0.4) * out_options['alpha_main']
    out_options['color'] = options.get('color')
    out_options['ci_mult'] = options.get('ci_mult', 1.0)
    out_options['label'] = options.get('label')
    out_options['width'] = options.get('width', 0.75)
    out_options['legend'] = options.get('legend', True)
    out_options['x_label'] = options.get('x_label')
    out_options['y_label'] = options.get('y_label')

    return out_options

def lineplot(ax, mean, stderr: Optional[np.ndarray] = None, options: Optional[Dict[str, Any]] = None):
    o = buildOptions(options)

    p, = ax.plot(mean, label=o['label'], linestyle=o['style'], color=o['color'], alpha=o['alpha_main'], linewidth=o['width'])

    color = p.get_color()
    if stderr is not None:
        (lo, hi) = confidenceInterval(mean, stderr, o['ci_mult'])
        ax.fill_between(range(len(mean)), lo, hi, color=color, alpha=o['alpha'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if o['legend']:
        ax.legend(frameon=False)

    if o['x_label']:
        ax.set_xlabel(o['x_label'])

    if o['y_label']:
        ax.set_ylabel(o['y_label'])

def plot(result: DuckResult, ax, options: Optional[Dict[str, Any]] = None):
    if options is None: options = {}

    label = options.get('label')

    # if we don't have a label and that's because it wasn't specified
    # then default to the agent's name
    if not label and 'label' not in options:
        label = result.exp.agent
        options['label'] = label

    # call through to the result's mean/stderr functions
    mean = result.mean()
    stderr = result.stderr()

    return lineplot(ax, mean, stderr, options)

def plotBest(results: ResultList, ax, options: Optional[Dict[str, Any]] = None):
    if options is None: options = {}

    prefer = options.get('prefer', 'big')
    best = getBest(results, prefer=prefer)

    return plot(best, ax, options)