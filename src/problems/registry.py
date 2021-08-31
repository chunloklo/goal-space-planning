from src.problems.MountainCar import MountainCar
from src.problems.GrazingWorld import GrazingWorld
from src.problems.MazeWorld import MazeWorld

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    elif name == 'GrazingWorld':
        return GrazingWorld
    elif name == 'MazeWorld':
        return MazeWorld
    else: 
        raise NotImplementedError()