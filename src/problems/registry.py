from environments.GrazingWorldSimple import GrazingWorldSimple
from src.problems.MountainCar import MountainCar
from src.problems.GrazingWorld import GrazingWorld, GrazingWorldWithMiddleOption, GrazingWorldSimple, GrazingWorldAdam, GrazingWorldAdamNested
from src.problems.TMaze import TMaze
from src.problems.HMazeProblem import HMazeProblem

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
    elif name == 'GrazingWorldAdamNested':
        return GrazingWorldAdamNested
    elif name == 'TMaze':
        return TMaze
    elif name == 'HMaze':
        return HMazeProblem
    else: 
        raise NotImplementedError()