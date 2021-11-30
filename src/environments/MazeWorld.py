import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class MazeWorld(BaseEnvironment):
    """
    The board is a nxn matrix, with (using NumPy matrix indexing):
        [n-1, 0] as the start at bottom-left
        [2, 2] goal, occasional reward: 50, otherwise 0
        [2, n-3] goal, occasional reward: 40, otherwise 0
        [n-3,n-3] goal, always gives reward of 1

    Each time step incurs -0.1 reward. An episode terminates when the agent reaches the goal.
    """
    def __init__(self, seed:int):
        #super().__init__(seed, 12, reward_sequence_length, initial_learning)
        self.shape = (27, 11)
        self.nS = np.prod(self.shape)
        self.error_max_steps = 50000
        self.goals = {
            1:{
                "position" : (0,10),
                "reward" : 200,
                "current_reward":200,
                "iterator" : 0
            }
        }

        self.goals[1]['position'] = (0, 10)

        self.step_to_goals = {
            1: 97,
        }

        """
        deal with potential wall bump
        calculate scalar index of each special goal state, and check that the agent didn't end up moving there
        """
        self.wall_grids = []
        flag=False
        for i in range(3,self.shape[0],4):
            if flag:
                for j in range(self.shape[1]-1):
                    self.wall_grids.append(self.state_encoding((i,j)))   
            else:
                for j in range(1,self.shape[1]):
                    self.wall_grids.append(self.state_encoding((i,j)))
            flag = not flag


        self.action_encoding = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }

        self.step_penalty = -1
        self.nS = np.prod(self.shape)
        self.nA = 4
       
        self.start_state = (self.shape[0]-1, 0)
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[1]["position"]]
        self.terminal_states = [self.state_encoding(state) for state in self.terminal_state_positions]

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

        
    def start(self):
        self.num_steps = 0
        return self.current_state

    # give all actions for a given state
    def actions(self, s):
        return [UP, RIGHT, DOWN, LEFT]

    # give the rewards associated with a given state, action, next state tuple
    def rewards(self, s, terminal):
        if terminal:
            globals.collector.collect('max_return', self.goals[1]["current_reward"] + self.step_to_goals[1] * self.step_penalty)  
            return self.goals[1]["current_reward"]
        else:
            return self.step_penalty
            

    # get the next state and termination status
    def next_state(self, s, a):
        # list of terminal state positions (top left, right, and bottom right)    
        is_done = tuple(s) in self.terminal_state_positions      
        self.current_state = self._limit_coordinates(np.array(s), np.array(a)).astype(int)
        self.current_state = self.start_state if is_done else self.current_state

        return self.current_state, is_done

    def step(self, a):
        self.num_steps += 1
        if (self.num_steps >= self.error_max_steps):
            raise InvalidRunException(f'There have been {self.error_max_steps} steps in this episode, over the maximum allowed number of steps. This means the agent is likely stuck and thus exiting.')
        s = self.current_state
        sp, t = self.next_state(s, self.action_encoding[a])
        r = self.rewards(s, t)
        return (r, sp, t)

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

    def state_encoding(self,state):
        return state[0]*self.shape[1]+state[1]

