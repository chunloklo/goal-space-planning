import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException
from src.environments.GrazingWorld import GrazingWorld

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

        self.goals[1]['position'] = (1, 1)
        self.goals[2]['position'] = (2, 7)
        self.goals[3]['position'] = (6, 9)

        self.step_to_goals = {
            1: 10,
            2: 11,
            3: 7
        }

        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (self.shape[0]-2, 2)
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[i]["position"] for i in range(1,4)]
        self.terminal_states = [state[0]*self.shape[1] + state[1] for state in self.terminal_state_positions]

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
        for i in range(1,3):
            goal_number = np.ravel_multi_index(np.array(self.goals[i]["position"]), self.shape)
            wall_grids.append(goal_number-1)
            wall_grids.append(goal_number+1)
            wall_grids.append(goal_number+self.shape[1] - 1)
            wall_grids.append(goal_number+self.shape[1])   
            wall_grids.append(goal_number+self.shape[1] + 1)
    
        if np.ravel_multi_index(coord, self.shape) in wall_grids:
            coord = s
        return coord

