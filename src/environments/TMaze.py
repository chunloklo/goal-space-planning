import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException
from src.environments.RewardSchedules import ConstantRewardSchedule, CyclingRewardSchedule

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class TMaze(BaseEnvironment):
    """
    The environment is a 7x7 grid world, with (using NumPy matrix indexing):
        [6, 3] as the start at bottom center
        [0, 3] junction
        [0, 0] left goal
        [0, 6] right goal

    Each time step incurs -0.1 reward. An episode terminates when the agent reaches the goal.
    """
    def __init__(self, seed:int, reward_sequence_length=10, initial_learning=0):
        random.seed(1)
        self.size = 15
        self.shape = (self.size, self.size)
        self.reward_sequence_length = reward_sequence_length
        self.initial_learning = initial_learning
        self.num_steps = None
        self.error_max_steps = 100000000000
        """
        dictionary to keep track of goals
            position: position on the grid
            reward: when it's not zero (for the 3rd goal it is always 1)
            reward_sequence: generate reward sequences of both top goals to start with 0, lenght according to poisson dist.
            iterators: to keep track where we are in the reward sequence
        """
        self.goals = [
            {
                "position" : (self.size - 1,0),
                "schedule" : CyclingRewardSchedule([0, 100], self.reward_sequence_length, cycle_type='step')
            },
            {
                "position" : (self.size - 1, self.size - 1),
                "schedule" : CyclingRewardSchedule([100, 0], self.reward_sequence_length, cycle_type='step')
            },
        ]

        self.step_to_goals = [
            self.size - 1 + ((self.size - 1) / 2) + self.size - 1,
            self.size - 1 + ((self.size - 1) / 2) + self.size - 1,
        ]

        self.action_encoding = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }

        # Fixed world for 7x7. Moved to variable size
        # self.world = [
        #     [ '0', '0', '0', '0', '0', '0', '0' ],
        #     [ '0', '1', '1', '0', '1', '1', '0' ],
        #     [ '0', '1', '1', '0', '1', '1', '0' ],
        #     [ '0', '1', '1', '0', '1', '1', '0' ],
        #     [ '0', '1', '1', '0', '1', '1', '0' ],
        #     [ '0', '1', '1', '0', '1', '1', '0' ],
        #     ['G1', '1', '1', '0', '1', '1', 'G2'],
        # ]

        # self.world = [['0'] * self.size] * self.size

        assert self.size % 2 == 1

        self.valid_grids = []
        for r in range(self.size):
            for c in range(self.size):
                if r == 0 or c == 0 or c == self.size - 1 or c == self.size // 2:
                    self.valid_grids.append((r, c))

        self.step_penalty = -1
        self.nS = np.prod(self.shape)
        self.nA = 4
        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (self.size - 1,  self.size // 2)
        self.current_state = self.start_state
        self.terminal_state_positions = [goal["position"] for goal in self.goals]
        
    def start(self):
        self.num_steps = 0
        return self.current_state

    # give all actions for a given state
    def actions(self, s):
        return [UP, RIGHT, DOWN, LEFT]

    # give the rewards associated with a given state, action, next state tuple
    def rewards(self, s, terminal):
        if 'num_steps_passed' in globals.blackboard and 'step_logging_interval' in globals.blackboard:
            if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
                goal_reward_rates = np.zeros(len(self.goals))
                for i in range(len(self.goals)):
                    num_steps = self.step_to_goals[i]
                    goal_reward_rates[i] = (num_steps * self.step_penalty + self.goals[i]["schedule"]()) / (num_steps + 1)
                max_reward_rate = np.max(goal_reward_rates)
                globals.collector.collect('max_reward_rate', max_reward_rate) 
        if terminal:  
            for i in range(len(self.goals)):
                if self.goals[i]["position"]==tuple(s):
                    return self.goals[i]["schedule"]()
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
        coord[0] = min(coord[0], self.size - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.size - 1)
        coord[1] = max(coord[1], 0)
        """
        deal with potential wall bump
        calculate scalar index of each special goal state, and check that the agent didn't end up moving there
        """
        # print(coord in self.wall_grids)
        if tuple(coord) not in self.valid_grids:
            coord = s
            # print('hit a wall!')
        return coord

