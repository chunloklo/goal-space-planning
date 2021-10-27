# Tabular Agents

from agents.Q_Tabular import Q_Tabular
from agents.Dynaqp_Tab import Dynaqp_Tab

# Tabular Agents with Options
from agents.Option_Given_Q_Tab import Option_Given_Q_Tab
from agents.Dyna_Option_Givenqp_Tab import Dyna_Option_Givenqp_Tab
from agents.DynaQP_OptionIntra_Tab import DynaQP_OptionIntra_Tab
from agents.DynaESP_OptionIntra_Tab import DynaESP_OptionIntra_Tab
from agents.Dynaesp_Tab import Dynaesp_Tab

# Option Planning
from agents.OptionPlanning_Tab import OptionPlanning_Tab


def getAgent(name):
    if name == 'Q_Tabular':
        return Q_Tabular
    elif name == 'Dynaqp_Tab':
        return Dynaqp_Tab
    elif name == 'Dynaesp_Tab':
        return Dynaesp_Tab
    elif name == 'Option_Given_Q_Tab':
        return Option_Given_Q_Tab
    elif name == 'Dyna_Option_Givenqp_Tab':
        return Dyna_Option_Givenqp_Tab
    elif name == 'DynaQP_OptionIntra_Tab':
        return DynaQP_OptionIntra_Tab
    elif name == 'DynaESP_OptionIntra_Tab':
        return DynaESP_OptionIntra_Tab
    elif name == 'OptionPlanning_Tab':
        return OptionPlanning_Tab
    else:
        raise NotImplementedError()