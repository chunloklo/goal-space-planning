from environments.GrazingWorldSimple import GrazingWorldSimple
from src.problems.MountainCar import MountainCar
from src.problems.GrazingWorld import GrazingWorld, GrazingWorldWithMiddleOption, GrazingWorldSimple, GrazingWorldAdam

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
    else: 
        raise NotImplementedError()