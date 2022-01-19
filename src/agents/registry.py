# Tabular Agents with Options
<<<<<<< HEAD
from agents.DynaOptions_Tab import DynaOptions_Tab
from agents.Dyna_Tab import Dyna_Tab
from agents.ActorCritic_Tab import ActorCritic_Tab
from agents.Direct_Tab import Direct_Tab
from agents.OptionPlanning_Tab import OptionPlanning_Tab
from agents.DynaOptions_NN import DynaOptions_NN
from agents.DynaOptionsLtd_Tab import DynaOptionsLtd_Tab
from agents.DynaOptionsGPI_Tab import DynaOptionsGPI_Tab
=======
from src.agents.DynaOptions_Tab import DynaOptions_Tab
from src.agents.Dyna_Tab import Dyna_Tab
from src.agents.ActorCritic_Tab import ActorCritic_Tab
from src.agents.Direct_Tab import Direct_Tab
from src.agents.OptionPlanning_Tab import OptionPlanning_Tab
from src.agents.DynaOptions_NN import DynaOptions_NN
>>>>>>> 9d2c90e88615c7109933b30709b317bef489d473

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
    elif name == 'DynaOptionsLtd_Tab':
        return DynaOptionsLtd_Tab
    elif name == 'DynaOptionsGPI_Tab':
        return DynaOptionsGPI_Tab
    else:
        raise NotImplementedError()