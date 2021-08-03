from agents.SARSA import SARSA
from agents.Q_Tabular import Q_Tabular
from agents.DynaQ_Tab_Sample import DynaQ_Tab_Sample
from agents.PrioritizedSweep import PrioritizedSweep
from agents.DynaQ_Tab_Dist import DynaQ_Tab_Dist
from agents.DynaQ_Tab_Exp import DynaQ_Tab_Exp

def getAgent(name):
    if name == 'SARSA':
        return SARSA
    elif name == 'Q_Tabular':
        return Q_Tabular
    elif name == 'DynaQ_Tab_Sample':
        return DynaQ_Tab_Sample
    elif name == 'PrioritizedSweep':
        return PrioritizedSweep
    elif name == 'DynaQ_Tab_Dist':
        return DynaQ_Tab_Dist
    elif name == 'DynaQ_Tab_Exp':
        return DynaQ_Tab_Exp
        
    else:
        raise NotImplementedError()
