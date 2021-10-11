from environments.GrazingWorldSimple import GrazingWorldSimple
from src.problems.MountainCar import MountainCar
from src.problems.GrazingWorld import GrazingWorld, GrazingWorldWithMiddleOption, GrazingWorldSimpleProblem, GrazingWorldSimpleProblemDirectOptions
from src.problems.MazeWorld import MazeWorld

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    elif name == 'GrazingWorld':
        return GrazingWorld
    elif name == 'GrazingWorldSimple':
        return GrazingWorldSimpleProblem
    elif name == 'GrazingWorldSimpleProblemDirectOptions':
        return GrazingWorldSimpleProblemDirectOptions
    elif name == 'GrazingWorldWithMiddleOption':
        return GrazingWorldWithMiddleOption
    elif name == 'MazeWorld':
        return MazeWorld
    else: 
        raise NotImplementedError()