import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random

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
    def __init__(self, seed:int,  size=8, reward_sequence_length=10, initial_learning=0):
        random.seed(1)
        self.size = size
        self.shape = (size, int(size*1.5))
        self.reward_sequence_length = reward_sequence_length
        self.initial_learning = initial_learning
        self.il_counter = -1
        self.special_goal_nums = [1,2]
        """
        dictionary to keep track of goals
            position: position on the grid
            reward: when it's not zero (for the 3rd goal it is always 1)
            reward_sequence: generate reward sequences of both top goals to start with 0, lenght according to poisson dist.
            iterators: to keep track where we are in the reward sequence
        """
        self.goals = {
            1:{
                "position" : (1,1),
                "reward" : 100,
                "current_reward":0,
                "reward_sequence_length": self.reward_sequence_length,
                "iterator" : 0
            },
            2:{
                "position" : (2,size-5),
                "reward" : 50,
                "current_reward":0,
                "reward_sequence_length": self.reward_sequence_length,
                "iterator" : int(self.reward_sequence_length/2)
            },
            3:{
                "position" : (size-2,size-3),
                "current_reward":1,
            }
        }

        self.step_to_goals = {
            1: 10,
            2: 10,
            3: 7
        }

        self.action_encoding = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }

        self.step_penalty = -1
        self.nS = np.prod(self.shape)
        self.nA = 4
        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (self.shape[0]-3, 2)
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[i]["position"] for i in range(1,4)]
        
    def start(self):
        return self.current_state

    # give all actions for a given state
    def actions(self, s):
        return [UP, RIGHT, DOWN, LEFT]

    def state_encoding(self,position):
        return self.size*position[0]+position[1]

    # give the rewards associated with a given state, action, next state tuple
    def rewards(self, s, terminal):
        if terminal:
            rewards = [self.goals[i]["current_reward"] for i in range(1,4)]
            globals.collector.collect('goal_rewards', rewards)  
            best_goal_num = np.argmax(rewards) + 1
            globals.collector.collect('max_return', self.goals[best_goal_num]["current_reward"] + self.step_to_goals[best_goal_num] * self.step_penalty)  
            for i in range(1,4):
                if self.goals[i]["position"]==tuple(s):
                    globals.collector.collect('end_goal', i)  
                    return self.goals[i]["current_reward"]
        else:
            return self.step_penalty
            
    # if iterator reached the end of sequence, generate new sequence and flip reward amount for both goals with not fixed rewards
    def update_goals(self):
        if self.il_counter < self.initial_learning:
            self.il_counter +=1
            increment_iter = False
        else:
            increment_iter = True
        for i in range(1,3):
            if self.goals[i]["iterator"] == self.goals[i]["reward_sequence_length"]:
                self.gen_reward_sequence(i,self.goals[i]["current_reward"] )
                self.goals[i]["iterator"] = 0
            else:
                if increment_iter  and i in self.special_goal_nums:
                    self.goals[i]["iterator"] += 1


    def gen_reward_sequence(self,terminal_state, previous_terminal_reward):
        #self.goals[terminal_state]["reward_sequence_length"] =  np.random.poisson(lam=self.reward_sequence_length)
        if previous_terminal_reward == 0:
            self.goals[terminal_state]["current_reward"] = self.goals[terminal_state]["reward"]
        else:
            self.goals[terminal_state]["current_reward"] = 0

    # get the next state and termination status
    def next_state(self, s, a):
        # list of terminal state positions (top left, right, and bottom right)    
        is_done = tuple(s) in self.terminal_state_positions      
        self.current_state = self._limit_coordinates(np.array(s), np.array(a)).astype(int)
        self.current_state = self.start_state if is_done else self.current_state

        return self.current_state, is_done

    def step(self, a):
        s = self.current_state
        sp, t = self.next_state(s, self.action_encoding[a])
        r = self.rewards(s, t)
        if t:
            self.update_goals()
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
        for i in range(1,3):
            goal_number = np.ravel_multi_index(np.array(self.goals[i]["position"]), self.shape)
            # wall_grids.append(goal_number-self.shape[0] - 1)
            # wall_grids.append(goal_number-self.shape[0] + 1)
            wall_grids.append(goal_number-1)
            wall_grids.append(goal_number+1)
            wall_grids.append(goal_number+self.shape[0] - 1)
            wall_grids.append(goal_number+self.shape[0])   
            wall_grids.append(goal_number+self.shape[0] + 1)
    
        if np.ravel_multi_index(coord, self.shape) in wall_grids:
            coord = s
        return coord

