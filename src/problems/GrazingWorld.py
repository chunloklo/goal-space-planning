from src.problems.BaseProblem import BaseProblem
from src.environments.GrazingWorld import GrazingWorld as GWEnv
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular

class GrazingWorld(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = GWEnv(self.seed)
        self.actions = 4

        self.rep = Tabular(self.env.shape, self.actions)

        self.features = self.rep.features()
        self.gamma = 1.0
