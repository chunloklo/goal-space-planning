import numpy as np
from RlGlue import BaseEnvironment
from src.environments.RewardSchedules import ConstantRewardSchedule, CyclingRewardSchedule
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GrazingWorld(BaseEnvironment):
    """
    The board is a nxn matrix, with (using NumPy matrix indexing):
        [n-1, 0] as the start at bottom-left
        [2, 2] goal, occasional reward: 50, otherwise 0
        [2, n-3] goal, occasional reward: 40, otherwise 0
        [n-3,n-3] goal, always gives reward of 1

    Each time step incurs -0.1 reward. An episode terminates when the agent reaches the goal.
    """
    def __init__(self, seed:int,  size=10, reward_sequence_length=10, initial_learning=0):
        random.seed(1)
        self.size = size
        self.shape = (size, size)
        self.reward_sequence_length = reward_sequence_length
        self.initial_learning = initial_learning
        self.il_counter = -1
        self.special_goal_nums = [1,2]
        self.num_steps = None
        self.error_max_steps = 10000000
        """
        dictionary to keep track of goals
            position: position on the grid
            reward: when it's not zero (for the 3rd goal it is always 1)
            reward_sequence: generate reward sequences of both top goals to start with 0, lenght according to poisson dist.
            iterators: to keep track where we are in the reward sequence
        """
        self.reward_schedule = globals.param['reward_schedule']
        if self.reward_schedule == 'cyclic':
            self.goals = [
                {
                    "position" : (2,2),
                    "schedule": CyclingRewardSchedule([0, 3], self.reward_sequence_length, cycle_offset=0, cycle_type='step'),
                },
                {
                    "position" : (2,size-3),
                    "schedule": CyclingRewardSchedule([0, 2], self.reward_sequence_length, cycle_offset=self.reward_sequence_length // 2, cycle_type='step'),
                },
                {
                    "position" : (size-3,size-3),
                    "schedule": ConstantRewardSchedule(1),
                } 
            ]
        elif self.reward_schedule == 'goal2_switch':
            self.goals = [
                {
                    "position" : (2,2),
                    "schedule": ConstantRewardSchedule(0),
                },
                {
                    "position" : (2,size-3),
                    "schedule": CyclingRewardSchedule([0, 0.5], self.reward_sequence_length, cycle_type='step', repeat=False),
                },
                {
                    "position" : (size-3,size-3),
                    "schedule": ConstantRewardSchedule(0),
                } 
            ]
        elif self.reward_schedule == 'zero_debug':
            self.goals = [
                #For saving options
                {
                    "position" : (2,2),
                    "schedule": ConstantRewardSchedule(0),
                },
                {
                    "position" : (2,size-3),
                    "schedule": ConstantRewardSchedule(0),
                },
                {
                    "position" : (size-3,size-3),
                    "schedule": ConstantRewardSchedule(0),
                } 
            ]
        else:
            raise NotImplementedError(f'"reward_schedule" {self.reward_schedule} not implemented in GrazingWorld')

        self.step_to_goals = [
            13,
            18,
            9
        ]

        self.action_encoding = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }
        # standard rewards
        # self.step_penalty = -1
        # scaled rewards (for NN)
        self.step_penalty = -0.1
        self.nS = np.prod(self.shape)
        self.nA = 4
        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (self.shape[0]-1, 0)
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[i]["position"] for i in range(len(self.goals))]
        
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
            # rewards = [self.goals[i]["schedule"]() for i in range(1,4)]
            # globals.collector.collect('goal_rewards', rewards)  
            # best_goal_num = np.argmax(rewards) + 1
            # globals.collector.collect('max_return', self.goals[best_goal_num]["schedule"]() + self.step_to_goals[best_goal_num] * self.step_penalty)  
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
            wall_grids.append(goal_number-self.shape[0] - 1)
            wall_grids.append(goal_number-self.shape[0] + 1)
            wall_grids.append(goal_number-1)
            wall_grids.append(goal_number+1)
            wall_grids.append(goal_number+self.shape[0] - 1)
            wall_grids.append(goal_number+self.shape[0])   
            wall_grids.append(goal_number+self.shape[0] + 1)
    
        if np.ravel_multi_index(coord, self.shape) in wall_grids:
            coord = s
        return coord

