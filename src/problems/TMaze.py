from src.problems.BaseProblem import BaseProblem
from src.environments.TMaze import TMaze as TMEnv
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option
from src.utils.create_options import get_options
from src.utils import globals

class TMaze(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = TMEnv(self.seed, reward_sequence_length=self.params['reward_sequence_length'], initial_learning=self.params['exploration_phase'])
        self.actions = 4
        self.options = None
        self.rep = Tabular(self.env.shape, self.actions)

        self.features = self.rep.features()
        self.gamma = self.params['gamma']
        globals.blackboard['terminal_state'] = 49