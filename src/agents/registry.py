# Tabular Agents

from agents.Q_Tabular import Q_Tabular

# Tabular Agents with Options
from agents.Option_Given_Q_Tab import Option_Given_Q_Tab
from agents.Dyna_Option_Givenqp_Tab import Dyna_Option_Givenqp_Tab
from agents.DynaOptions_Tab import DynaOptions_Tab
from agents.Dyna_Tab import Dyna_Tab

# Option Planning
from agents.OptionPlanning_Tab import OptionPlanning_Tab

def getAgent(name):
    if name == 'Q_Tabular':
        return Q_Tabular
    elif name == 'DynaOptions_Tab':
        return DynaOptions_Tab
    elif name == 'Option_Given_Q_Tab':
        return Option_Given_Q_Tab
    elif name == 'Dyna_Option_Givenqp_Tab':
        return Dyna_Option_Givenqp_Tab
    elif name == 'OptionPlanning_Tab':
        return OptionPlanning_Tab
    elif name == 'Dyna_Tab':
        return Dyna_Tab
    else:
        raise NotImplementedError()