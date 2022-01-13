from src.problems.BaseProblem import BaseProblem
from src.environments.HMaze import HMaze
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option
from src.utils.create_options import get_options
from src.utils import globals, param_utils

class HMazeProblem(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = HMaze(self.seed, reward_sequence_length=self.params['reward_sequence_length'], initial_learning=self.params['exploration_phase'])
        self.actions = 4
        self.options = self.env.get_options()
        self.gamma = self.params['gamma']

    def get_representation(self, rep_type: str):
        rep_type = param_utils.check_valid(rep_type, lambda x: x in ['Tabular', 'Image'])
        if rep_type == 'Tabular':
            return self.env.get_tabular_feature()
        elif rep_type == 'Image':
            return self.env.get_image_feature()