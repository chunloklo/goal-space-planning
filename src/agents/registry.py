# Tabular Agents with Options
from agents.DynaOptions_Tab import DynaOptions_Tab
from agents.Dyna_Tab import Dyna_Tab
from agents.ActorCritic_Tab import ActorCritic_Tab
from agents.Direct_Tab import Direct_Tab
from agents.OptionPlanning_Tab import OptionPlanning_Tab

def getAgent(name):
    if name == 'DynaOptions_Tab':
        return DynaOptions_Tab
    elif name == 'OptionPlanning_Tab':
        return OptionPlanning_Tab
    elif name == 'Dyna_Tab':
        return Dyna_Tab
    elif name =='ActorCritic_Tab':
        return ActorCritic_Tab
    elif name == 'Direct_Tab':
        return Direct_Tab
    else:
        raise NotImplementedError()