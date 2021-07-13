from src.problems.MountainCar import MountainCar
from src.problems.GrazingWorld import GrazingWorld

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    elif name == 'GrazingWorld':
        return GrazingWorld

    else: 
        raise NotImplementedError()