from src.problems.BaseProblem import BaseProblem
from src.environments.GrazingWorld import GrazingWorld as GWEnv
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option

class GrazingWorld(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = GWEnv(self.seed, reward_sequence_length=500)
        self.actions = 4
        self.options = [load_option('GrazingO1'), load_option('GrazingO2'),load_option('GrazingO3')]
        #self.options = None
        self.rep = Tabular(self.env.shape, self.actions)

        self.features = self.rep.features()
        self.gamma = 0.95
