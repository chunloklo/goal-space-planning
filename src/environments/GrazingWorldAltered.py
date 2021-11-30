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

class GrazingWorldAltered(GrazingWorld):
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
        self.shape = (12, 18)
        self.nS = np.prod(self.shape)

        self.goals[1]['position'] = (2, 15)
        self.goals[2]['position'] = (9, 1)
        self.goals[3]['position'] = (0, 0)

        self.step_to_goals = {
            1: 15,
            2: 15,
            3: 15
        }

        """
        deal with potential wall bump
        calculate scalar index of each special goal state, and check that the agent didn't end up moving there
        """
        self.wall_grids = []
        for i in range(1,3):
            goal_number = np.ravel_multi_index(np.array(self.goals[i]["position"]), self.shape)
            self.wall_grids.append(goal_number-1)
            self.wall_grids.append(goal_number+1)
            self.wall_grids.append(goal_number+self.shape[1] - 1)
            self.wall_grids.append(goal_number+self.shape[1])   
            self.wall_grids.append(goal_number+self.shape[1] + 1)

        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (int(self.shape[0]/2), int(self.shape[1]/2))
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[i]["position"] for i in range(1,4)]
        self.terminal_states = [state[0]*self.shape[1] + state[1] for state in self.terminal_state_positions]

        self.selectable_states = list(range(self.shape[0]*self.shape[1]))
        for i, wall_grid in enumerate(self.wall_grids):
            self.selectable_states.remove(wall_grid)
        for i, terminal_state in enumerate(self.terminal_states):
            self.selectable_states.remove(terminal_state)

    def _limit_coordinates(self, s, a):
        """
        Prevent the agent from falling out of the grid world
        """
        coord = s+a
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)

    
        if np.ravel_multi_index(coord, self.shape) in self.wall_grids:
            coord = s
        return coord

