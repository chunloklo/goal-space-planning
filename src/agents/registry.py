from agents.SARSA import SARSA
from agents.Q_Tabular import Q_Tabular
from agents.DynaQ_Tabular import DynaQ_Tabular
from agents.PrioritizedSweep import PrioritizedSweep

def getAgent(name):
    if name == 'SARSA':
        return SARSA
    elif name == 'Q_Tabular':
        return Q_Tabular
    elif name == 'DynaQ_Tabular':
        return DynaQ_Tabular
    elif name == 'PrioritizedSweep':
        return PrioritizedSweep
    else:
        raise NotImplementedError()
