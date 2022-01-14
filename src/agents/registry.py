# Tabular Agents with Options
from src.agents.DynaOptions_Tab import DynaOptions_Tab
from src.agents.Dyna_Tab import Dyna_Tab
from src.agents.ActorCritic_Tab import ActorCritic_Tab
from src.agents.Direct_Tab import Direct_Tab
from src.agents.OptionPlanning_Tab import OptionPlanning_Tab
from src.agents.DynaOptions_NN import DynaOptions_NN

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
    elif name == 'DynaOptions_NN':
        return DynaOptions_NN
    else:
        raise NotImplementedError()