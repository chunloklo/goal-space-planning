from src.problems.BaseProblem import BaseProblem
from src.environments.MazeWorld import MazeWorld as MWEnv
from src.environments.GrazingWorld import GrazingWorld as GWEnv
from src.environments.GrazingWorldSimple import GrazingWorldSimple as GWSimpleEnv
from src.environments.GrazingWorldAdam import GrazingWorldAdam as GWEnvAdam
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option
from src.utils.create_options import get_options
from src.utils import globals

class MazeWorld(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = MWEnv(self.seed)
        self.actions = 4
        self.options = get_options("Maze")[0]
        self.rep = Tabular(self.env.shape, self.actions)

        self.features = self.rep.features()
        self.gamma = self.params['gamma']

        globals.blackboard['terminal_state'] = 297