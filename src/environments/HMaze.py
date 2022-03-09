from typing import List
import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException
from src.environments.RewardSchedules import ConstantRewardSchedule, CyclingRewardSchedule
from src.utils.Option import QOption
from PyFixedReps.BaseRepresentation import BaseRepresentation
from PyFixedReps.Tabular import Tabular

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class HMaze(BaseEnvironment):
    """
    The environment is a NxN grid world. There are 4 goals, one at each corner.
    Each time step incurs -0.1 reward. An episode terminates when the agent reaches the goal.
    """
    def __init__(self, seed:int, reward_sequence_length=10, initial_learning=0):
        random.seed(1)
        self.size = 15
        self.shape = (self.size, self.size)
        self.reward_sequence_length = reward_sequence_length
        self.num_steps = None

        self.goals = [
            {
                "position" : (0,0), # Top left
                "schedule": CyclingRewardSchedule([1.0, -1.0, -1.0, -1.0], self.reward_sequence_length, cycle_offset=initial_learning, cycle_type='step'),
            },
            {
                "position" : (self.size - 1, 0), # Bottom left
                "schedule": CyclingRewardSchedule([-1.0, -1.0, -1.0, 1.0], self.reward_sequence_length, cycle_offset=initial_learning, cycle_type='step'),
            },
            {
                "position" : (0, self.size - 1), # Top right
                "schedule": CyclingRewardSchedule([-1.0, -1.0, 1.0, -1.0], self.reward_sequence_length, cycle_offset=initial_learning, cycle_type='step'),
            },
            {
                "position" : (self.size - 1, self.size - 1), # Bottom right
                "schedule": CyclingRewardSchedule([-1.0, 1.0, -1.0, -1.0], self.reward_sequence_length, cycle_offset=initial_learning, cycle_type='step'),
            } 
        ]


        # self.goals = [
        #     {
        #         "position" : (0,0), # Top left
        #         "schedule": ConstantRewardSchedule(-1),
        #     },
        #     {
        #         "position" : (self.size - 1, 0), # Bottom left
        #         "schedule": ConstantRewardSchedule(-1),
        #     },
        #     {
        #         "position" : (0, self.size - 1), # Top right
        #         "schedule": ConstantRewardSchedule(1),
        #     },
        #     {
        #         "position" : (self.size - 1, self.size - 1), # Bottom right
        #         "schedule": ConstantRewardSchedule(-1),
        #     } 
        # ]


        # Takes self.size to get to goal
        self.step_to_goals = [self.size] * len(self.goals)
        # Accounting for the last step giving 0 reward.

        self.action_encoding = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }

        assert self.size % 2 == 1

        self.valid_grids = []
        for r in range(self.size):
            for c in range(self.size):
                # left col, right col, and then the middle row
                if c == 0 or c == self.size - 1 or r == self.size // 2:
                    self.valid_grids.append((r, c))

        self.step_penalty = -0.1
        # self.step_penalty = 0
        self.nS = len(self.valid_grids)
        self.nA = 4
        # Start in the middle square
        self.start_state = (self.size // 2,  self.size // 2)
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
                for i, goal in enumerate(self.goals):
                    num_steps = self.step_to_goals[i]
                    goal_reward_rates[i] = ((num_steps - 1) * self.step_penalty + goal["schedule"]()) / (num_steps + 1)
                max_reward_rate = np.max(goal_reward_rates)
                globals.collector.collect('max_reward_rate', max_reward_rate) 

        for goal in self.goals:
            if goal["position"]==tuple(s):
                return goal["schedule"]()
        
        if terminal:
            return 0
            
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
        s = self.current_state
        sp, t = self.next_state(s, self.action_encoding[a])
        r = self.rewards(sp, t)
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

    def get_goals_and_policies(self):
        tab_feature = self.get_tabular_feature()

        goals = [tab_feature.encode(s) for s in [(0, 0), 
            (0, self.size - 1), 
            (self.size // 2, 0), 
            (self.size - 1, 0), 
            (self.size - 1, self.size - 1), 
            (self.size // 2, self.size - 1)]]
        goal_0_policy = {}
        goal_1_policy = {}
        goal_2_policy = {}
        goal_3_policy = {}
        goal_4_policy = {}
        goal_5_policy = {}

        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self.valid_grids:
                    continue
                # GOAL 0: (0, 0)
                # print(self.size // 2)
                # asda
                if r <= self.size // 2 and c == 0:
                    goal_0_policy[tab_feature.encode((r, c))] = UP
                else:
                    goal_0_policy[tab_feature.encode((r, c))] = None

                # GOAL 1: (0, self.size - 1)
                if r <= self.size // 2 and c == self.size - 1:
                    goal_1_policy[tab_feature.encode((r, c))] = UP
                else:
                    goal_1_policy[tab_feature.encode((r, c))] = None

                # GOAL 2: (self.size // 2, 0)
                if c == 0 and r <= self.size // 2:
                    goal_2_policy[tab_feature.encode((r, c))] = DOWN
                elif c == 0 and r > self.size // 2:
                    goal_2_policy[tab_feature.encode((r, c))] = UP
                elif r == self.size // 2:
                    goal_2_policy[tab_feature.encode((r, c))] = LEFT
                else:
                    goal_2_policy[tab_feature.encode((r, c))] = None

                # GOAL 3: (self.size - 1, 0)
                if r >= self.size // 2 and c == 0:
                    goal_3_policy[tab_feature.encode((r, c))] = DOWN
                else:
                    goal_3_policy[tab_feature.encode((r, c))] = None
                
                # GOAL 4: (self.size - 1, self.size - 1)
                if r >= self.size // 2 and c == self.size - 1:
                    goal_4_policy[tab_feature.encode((r, c))] = DOWN
                else:
                    goal_4_policy[tab_feature.encode((r, c))] = None

                # GOAL 5: (self.size // 2, self.size - 1)
                if c == self.size - 1 and r <= self.size // 2:
                    goal_5_policy[tab_feature.encode((r, c))] = DOWN
                elif c == self.size - 1 and r > self.size // 2:
                    goal_5_policy[tab_feature.encode((r, c))] = UP
                elif r == self.size // 2:
                    goal_5_policy[tab_feature.encode((r, c))] = RIGHT
                else:
                    goal_5_policy[tab_feature.encode((r, c))] = None

        return goals, [goal_0_policy, goal_1_policy, goal_2_policy, goal_3_policy, goal_4_policy, goal_5_policy]

    def get_options(self):

        # Defining common policy selection and termination functions
        def policy_selection(policy, state):
            return policy[state]

        def termination_condition(termination_set, state):
            if state in termination_set:
                return True
            return False

        num_actions = 4

        tab_feature = self.get_tabular_feature()

        option_1_policy = {}
        for r in range(self.size):
            for c in range(self.size):
                if (r == self.size // 2):
                    option_1_policy[tab_feature.encode((r, c))] = LEFT
                
                if c == 0 or c == self.size -1:
                    option_1_policy[tab_feature.encode((r, c))] = UP

        term_set = [tab_feature.encode(s) for s in [(0, 0), (0, self.size - 1), (self.size // 2, 0), (self.size - 1, 0), (self.size - 1, self.size - 1), (self.size // 2, self.size - 1)]]

        # the initiation set doesn't actually matter
        option1 = QOption(None , option_1_policy,
                        term_set, policy_selection,
                        termination_condition, num_actions)
        
        option_2_policy = {}
        for r in range(self.size):
            for c in range(self.size):
                if (r == self.size // 2):
                    option_2_policy[tab_feature.encode((r, c))] = RIGHT
                
                if c == 0 or c == self.size -1:
                    option_2_policy[tab_feature.encode((r, c))] = DOWN

        # option_2_term_set = [tab_feature.encode(s) for s in [(self.size - 1, 0), (self.size - 1, self.size - 1), (self.size // 2, self.size - 1)]]

        # the initiation set doesn't actually matter
        option2 = QOption(None , option_2_policy,
                        term_set, policy_selection,
                        termination_condition, num_actions)

        return [option1, option2]

    def get_tabular_feature(self):
        return HMazeTabularFeature(self.valid_grids)

    def get_image_feature(self):
        return HMazeImageFeature(self.size, 16)

class HMazeTabularFeature(Tabular):
    def __init__(self, valid_grids: List):
        self.encode_map = {}
        self.decode_map = {}
        for i, s in enumerate(valid_grids):
            self.encode_map[s] = i
            self.decode_map[i] = s

        self.num_features = len(valid_grids)

    def encode(self, s):
        return self.encode_map[tuple(s)]

    def decode(self, x):
        return self.decode_map[x]

    def features(self):
        return self.num_features


class HMazeImageFeature(BaseRepresentation):
    def __init__(self, size: int, resolution: int):
        self.size = size
        self.resolution = resolution

    def features(self):
        return (self.resolution, self.resolution, 1)

    def encode(self, s):
        # converting from state space to resolution space
        def _image_space_convert(s):
            return s / self.size * (self.resolution - 1)
        img_coord = (_image_space_convert(s[0]), _image_space_convert(s[1]))
        
        weights = np.zeros((2, 2))
        r_floor = np.floor(img_coord[0])
        c_floor = np.floor(img_coord[1])
        r_extend = img_coord[0] - r_floor
        c_extend = img_coord[1] - c_floor

        weights[0, 0] = (1 - r_extend) * (1 - c_extend)
        weights[1, 0] = r_extend * (1 - c_extend)
        weights[0, 1] = (1 - r_extend) * c_extend
        weights[1, 1] = r_extend * c_extend

        image = np.zeros((self.resolution, self.resolution, 1))
        image[r_floor:r_floor+1, c_floor:c_floor+1] = weights
        return image

