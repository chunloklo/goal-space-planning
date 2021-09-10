from agents.SARSA import SARSA
# Tabular Agents with Primitive Actions
from agents.Q_Tabular import Q_Tabular
from agents.Dyna_Tab import Dyna_Tab
from agents.Dyna_Tab_Dist import Dyna_Tab_Dist
from agents.Dynaqp_Tab import Dynaqp_Tab
from agents.PrioritizedSweep import PrioritizedSweep

# Tabular Agents with Options
from agents.Option_Q_Tab import Option_Q_Tab
from agents.Option_Given_Q_Tab import Option_Given_Q_Tab

from agents.Dyna_Optionqp_Tab import Dyna_Optionqp_Tab
from agents.Dyna_Option_Givenqp_Tab import Dyna_Option_Givenqp_Tab

from agents.DynaQP_OptionIntra_Tab import DynaQP_OptionIntra_Tab

# Linear Agents with Primitive Actions
from agents.Q_Linear import Q_Linear
from agents.Dyna_Linear_Dist import Dyna_Linear_Dist

# Option Planning
from agents.OptionPlanning_Tab import OptionPlanning_Tab


def getAgent(name):
    if name == 'SARSA':
        return SARSA
    elif name == 'Q_Tabular':
        return Q_Tabular
    elif name == 'Dyna_Tab':
        return Dyna_Tab
    elif name == 'Dyna_Tab_Dist':
        return Dyna_Tab_Dist
    elif name == 'Dynaqp_Tab':
        return Dynaqp_Tab
    elif name == 'PrioritizedSweep':
        return PrioritizedSweep
    elif name == 'Q_Linear':
        return Q_Linear
    elif name == 'Dyna_Linear_Dist':
        return Dyna_Linear_Dist  
    elif name == 'Option_Q_Tab':
        return Option_Q_Tab
    elif name == 'Option_Given_Q_Tab':
        return Option_Given_Q_Tab
    elif name == 'Dyna_Optionqp_Tab':
        return Dyna_Optionqp_Tab
    elif name == 'Dyna_Option_Givenqp_Tab':
        return Dyna_Option_Givenqp_Tab
    elif name == 'DynaQP_OptionIntra_Tab':
        return DynaQP_OptionIntra_Tab
    elif name == 'OptionPlanning_Tab':
        return OptionPlanning_Tab
    else:
        raise NotImplementedError()