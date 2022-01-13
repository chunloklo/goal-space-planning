import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException
from src.environments.GrazingWorld import GrazingWorld
from src.agents.components.models import OptionActionModel_Sutton_Tabular, OptionModel_Sutton_Tabular
from src.utils.create_options import get_options
from typing import Tuple

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GrazingWorldAdam(GrazingWorld):
    """
    The board is a nxn matrix, with (using NumPy matrix indexing):
        [n-1, 0] as the start at bottom-left
        [2, 2] goal, occasional reward: 50, otherwise 0
        [2, n-3] goal, occasional reward: 40, otherwise 0
        [n-3,n-3] goal, always gives reward of 1

    Each time step incurs -0.1 reward. An episode terminates when the agent reaches the goal.
    """
    def __init__(self, seed:int, reward_sequence_length=10, initial_learning=0):
        super().__init__(seed, 12, reward_sequence_length, initial_learning)
        self.shape = (8, 12)
        self.nS = np.prod(self.shape)

        self.goals[0]['position'] = (1, 1)
        self.goals[1]['position'] = (2, 7)
        self.goals[2]['position'] = (6, 9)

        self.step_to_goals = [
            10,
            11,
            7
        ]

        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (self.shape[0]-2, 2)
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[i]["position"] for i in range(1,4)]
        self.terminal_states = [state[0]*self.shape[1] + state[1] for state in self.terminal_state_positions]

        globals.blackboard["grid_nS"] = self.shape[-1]*self.shape[-2]

    def _limit_coordinates(self, s, a):
        """
        Prevent the agent from falling out of the grid world
        """
        coord = s+a
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        """
        deal with potential wall bump
        calculate scalar index of each special goal state, and check that the agent didn't end up moving there
        """
        wall_grids = []
        for i in [0, 1]:
            goal_number = np.ravel_multi_index(np.array(self.goals[i]["position"]), self.shape)
            wall_grids.append(goal_number-1)
            wall_grids.append(goal_number+1)
            wall_grids.append(goal_number+self.shape[1] - 1)
            wall_grids.append(goal_number+self.shape[1])   
            wall_grids.append(goal_number+self.shape[1] + 1)
    
        if np.ravel_multi_index(coord, self.shape) in wall_grids:
            coord = s
        return coord

class GrazingWorldAdamImageFeature(BaseRepresentation):
    def __init__(self):
        self.base_token_viz = np.array([    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                            ['W', 'G', 'W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                            ['W', 'W', 'W', ' ', ' ', ' ', 'W', 'G', 'W', ' ', ' ', ' '],
                                            [' ', ' ', ' ', ' ', ' ', ' ', 'W', 'W', 'W', ' ', ' ', ' '],
                                            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'G', ' ', ' '],
                                            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']])
        self._preload_mapping()

    def _preload_mapping(self):
        self.representation_map = {}

        for s0 in range(8):
            for s1 in range(12):
                        
                image = np.zeros((8, 12, 1))
                for r in range(8):
                    for c in range(12):
                        image[r, c] = self._translate(self.base_token_viz[r, c])

                image[s0, s1] = self._translate('A')

                self.representation_map[(s0, s1)] = image

    def _translate(self, token: str):
        if token == ' ':
            return 1.0
        elif token == 'G':
            return 1.0
        elif token == 'A':
            return 0.0
        elif token == 'W':
            return 0.99
        pass

    def features(self):
        return (8, 12, 1)

    def encode(self, s):
        return self.representation_map[tuple(s)]

def state_index_to_coord(x):
    s = (x // 12, x % 12)
    return s

def get_pretrained_option_model() -> Tuple[OptionModel_Sutton_Tabular, OptionActionModel_Sutton_Tabular]:
    model_r = np.load('src/environments/data/GrazingWorldAdam_OptionModel_r.npy', False)
    model_discount = np.load('src/environments/data/GrazingWorldAdam_OptionModel_discount.npy', False)
    model_transition = np.load('src/environments/data/GrazingWorldAdam_OptionModel_transition.npy', False)

    num_states = 8 * 12 
    num_actions = 4
    num_options = 3
    options = get_options('GrazingAdam')
    
    option_model = OptionModel_Sutton_Tabular(num_states + 1, num_actions, num_options, options)
    option_model.reward_model = model_r
    option_model.discount_model = model_discount
    option_model.transition_model = model_transition


    action_model_r = np.load('src/environments/data/GrazingWorldAdam_ActionOptionModel_r.npy', False)
    action_model_discount = np.load('src/environments/data/GrazingWorldAdam_ActionOptionModel_discount.npy', False)
    action_model_transition = np.load('src/environments/data/GrazingWorldAdam_ActionOptionModel_transition.npy', False)
    
    action_option_model = OptionActionModel_Sutton_Tabular(num_states + 1, num_actions, num_options, options)
    action_option_model.reward_model = action_model_r
    action_option_model.discount_model = action_model_discount
    action_option_model.transition_model = action_model_transition
    return option_model, action_option_model

def get_all_transitions():
    base_token_viz = np.array([ [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                ['W', 'G', 'W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                ['W', 'W', 'W', ' ', ' ', ' ', 'W', 'G', 'W', ' ', ' ', ' '],
                                [' ', ' ', ' ', ' ', ' ', ' ', 'W', 'W', 'W', ' ', ' ', ' '],
                                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'G', ' ', ' '],
                                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']])

    env = GrazingWorldAdam(0)
    env.start()
    transitions = []
    for r in range(8):
        for c in range(12):
            for a in range(4):
                if base_token_viz[r, c] == 'W':
                    continue
                s = (r, c)
                env.current_state = s
                reward, sp, t = env.step(a)
                if (t):
                    gamma = 0
                    sp = (0, 0)
                else:
                    gamma = 1
                transitions.append((tuple(s), a, tuple(sp), reward, gamma))
    return transitions