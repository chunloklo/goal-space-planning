from src.problems.BaseProblem import BaseProblem
from src.environments.HMaze import HMaze
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option
from src.utils.create_options import get_options
from src.utils import globals, param_utils
import pickle

class HMazeProblem(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = HMaze(self.seed, reward_sequence_length=self.params['reward_sequence_length'], initial_learning=self.params['exploration_phase'])
        self.actions = 4
        self.options = self.env.get_options()
        self.gamma = self.params['gamma']
        self.goals, self.goal_policies = self.env.get_goals_and_policies()

    def get_representation(self, rep_type: str):
        rep_type = param_utils.check_valid(rep_type, lambda x: x in ['Tabular', 'Image'])
        if rep_type == 'Tabular':
            return self.env.get_tabular_feature()
        elif rep_type == 'Image':
            return self.env.get_image_feature()
    
    def get_goals(self):
        return self.goals
    
    def get_learned_goal_policies(self):
        return self.goal_policies

    def get_pretrained_GSP_models(self):
        state_estimate_learner = pickle.load(open('src/environments/data/HMaze/state_estimate_learner.pkl', 'rb'))
        goal_estimate_learner = pickle.load(open('src/environments/data/HMaze/goal_estimate_learner.pkl', 'rb'))
        goal_value_learner = pickle.load(open('src/environments/data/HMaze/goal_value_learner.pkl', 'rb'))
        return state_estimate_learner, goal_estimate_learner, goal_value_learner