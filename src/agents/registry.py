# Tabular Agents with Options
from src.agents.GSP_NN import GSP_NN
from src.agents.DynaOptions_Tab import DynaOptions_Tab
from src.agents.Dyna_Tab import Dyna_Tab
from src.agents.ActorCritic_Tab import ActorCritic_Tab
from src.agents.Direct_Tab import Direct_Tab
from src.agents.OptionPlanning_Tab import OptionPlanning_Tab
from src.agents.DynaOptions_NN import DynaOptions_NN
from src.agents.GSP_Tab import GSP_Tab
from src.agents.Dyna_NN import Dyna_NN
from src.agents.Dreamer import Dreamer
from src.agents.DynO_FromGSP_NN import DynO_FromGSP_NN

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
    elif name == 'GSP_Tab':
        return GSP_Tab
    elif name == 'Dyna_NN':
        return Dyna_NN
    elif name == 'GSP_NN':
        return GSP_NN
    elif name == 'Dreamer':
        return Dreamer
    elif name == 'DynO_FromGSP_NN':
        return DynO_FromGSP_NN
    else:
        raise NotImplementedError()