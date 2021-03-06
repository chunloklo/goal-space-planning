from src.environments.GrazingWorldSimple import GrazingWorldSimple
from src.problems.MountainCar import MountainCar
from src.problems.GrazingWorld import GrazingWorld, GrazingWorldWithMiddleOption, GrazingWorldSimple, GrazingWorldAdam
from src.problems.TMaze import TMaze
from src.problems.HMazeProblem import HMazeProblem
from src.problems.PinballProblem import PinballOracleProblem, PinballProblem, PinballTermProblem

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar
    elif name == 'GrazingWorld':
        return GrazingWorld
    elif name == 'GrazingWorldSimple':
        return GrazingWorldSimple
    elif name == 'GrazingWorldWithMiddleOption':
        return GrazingWorldWithMiddleOption
    elif name == 'GrazingWorldAdam':
        return GrazingWorldAdam
    elif name == 'TMaze':
        return TMaze
    elif name == 'HMaze':
        return HMazeProblem
    elif name == 'PinballProblem':
        return PinballProblem
    elif name == 'PinballTermProblem':
        return PinballTermProblem
    elif name == 'PinballOracleProblem':
        return PinballOracleProblem
    else: 
        raise NotImplementedError()