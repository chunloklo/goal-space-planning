from agents.SARSA import SARSA
from agents.Q_Tabular import Q_Tabular
def getAgent(name):
    if name == 'SARSA':
        return SARSA
    elif name == 'Q_Tabular':
        return Q_Tabular
    else:
        raise NotImplementedError()
