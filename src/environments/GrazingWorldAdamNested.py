import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.utils.run_utils import InvalidRunException
from src.environments.GrazingWorld import GrazingWorld
import math


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GrazingWorldAdamNested(GrazingWorld):
    """
    Currently only works up to 1 nest, goal3's and last nest goal1 and 2's are terminal states

    The board is a nxn matrix, with (using NumPy matrix indexing):
        [n-1, 0] as the start at bottom-left
        [2, 2] goal, occasional reward: 50, otherwise 0
        [2, n-3] goal, occasional reward: 40, otherwise 0
        [n-3,n-3] goal, always gives reward of 1

    Each time step incurs -0.1 reward. An episode terminates when the agent reaches the goal.
    """
    def __init__(self, seed:int, reward_sequence_length=10, initial_learning=0, num_nests = 2):
        super().__init__(seed, 12, reward_sequence_length, initial_learning)
        
        
        self.num_nests = num_nests
        self.nest_counter = 0
        
        # first nest has 1 env, then 2* as many in each subsequent nest
        
        if self.num_nests<2:
            raise Exception("Number of nests must be greater than 1.")
        self.shape = (self.num_nests**2-1,8, 12)
        self.nS = np.prod(self.shape)

        globals.blackboard["grid_nS"] = self.shape[-1]*self.shape[-2]

        self.non_nested_goal_positions = [ (1,1), (2, 7) ]  

        self.third_goal_pos = (6, 9)
        self.third_goal_pos_full = (0,6,9)
        self.current_grid_no = 0
        self.transition_goal_positions = []
        for i in range(self.shape[0]):
            for goal in self.non_nested_goal_positions:
                self.transition_goal_positions.append( tuple(  (i,)  ) + goal) 

        # for 2 layers we have 2**2 goal states, and 5 phases, since we want to have phase for 0 rewards too (phase is a period of timesteps during which one reward is "on")
        # mostly just works 2 layers with 600 reward sequence length
        self.phases = (2**self.num_nests+1)
        self.full_cycle  = self.phases* self.reward_sequence_length # so far just done for 1 layer of depth
        self.goals = dict()
        for grid_no in range(math.floor(self.shape[0]/2), self.shape[0]):
            it_start = grid_no*(self.phases) -1 

            for goal_no in range(1,3):
                self.goals[(grid_no, goal_no)] = dict()
                self.goals[(grid_no, goal_no)]["position"] = tuple( (grid_no,) ) + self.non_nested_goal_positions[goal_no-1]
                self.goals[(grid_no, goal_no)]["reward_sequence_length"] = self.reward_sequence_length
                self.goals[(grid_no, goal_no)]["current_reward"] = 0

                if goal_no ==1:
                    self.goals[(grid_no, goal_no)]["reward"] =100
                    self.goals[(grid_no, goal_no)]["iterator"] = self.full_cycle - (it_start+1)*self.full_cycle/(self.phases*2)
                elif goal_no == 2:
                    self.goals[(grid_no, goal_no)]["reward"] = 50
                    self.goals[(grid_no, goal_no)]["iterator"] = self.full_cycle - it_start*self.full_cycle/(self.phases*2)

        for grid_no in range(self.shape[0]):
            self.goals[(grid_no, 3)] = {}
            self.goals[(grid_no, 3)]["position"] = tuple( (grid_no,) )  + self.third_goal_pos
            self.goals[(grid_no, 3)]["current_reward"] = 1

        self.goal_pos_to_goal_num = {}
        for k,v in self.goals.items():
            self.goal_pos_to_goal_num[v["position"]] = k
        

        # hard coded for now, has to be made general later (10 steps from start to goal 1, 11 steps from start to goal 2)
        self.step_to_goals = {
            (0,3): 7,
            (1,1): 20,
            (1,2): 21,
            (2,1): 21,
            (2,2): 22
        } 
        
        """
        deal with potential wall bump
        calculate scalar index of each special goal state, and check that the agent didn't end up moving there
        """
        self.wall_grids = []
        for grid_no in range(self.shape[0]):
            #only do goal 1 and 2
            for goal_no in range(2):
                goal_pos = self.non_nested_goal_positions[goal_no]
                self.wall_grids.append( tuple( (grid_no,) ) + ( goal_pos[0], goal_pos[1]-1  ) )
                self.wall_grids.append(  tuple( (grid_no,) ) +  ( goal_pos[0], goal_pos[1]+1  ) )
                self.wall_grids.append(  tuple( (grid_no,) )  +  ( goal_pos[0]+1, goal_pos[1]-1  ) )
                self.wall_grids.append(  tuple( (grid_no,) )  +  ( goal_pos[0]+1, goal_pos[1]  ) )
                self.wall_grids.append(  tuple( (grid_no,) ) +  ( goal_pos[0]+1, goal_pos[1]+1  ) )

        #self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.start_state = (0,self.shape[1]-2, 2)
        self.current_state = self.start_state
        self.terminal_state_positions = [self.goals[k]["position"] for k in self.goals.keys()]
        self.terminal_states = [np.ravel_multi_index(np.array(state), self.shape) for state in self.terminal_state_positions]  # goal 3 is terminal state in all levels

        # self.selectable_states = list(range(self.shape[0]*self.shape[1]))
        # for i, wall_grid in enumerate(self.wall_grids):
        #     self.selectable_states.remove(wall_grid)
        # for i, terminal_state in enumerate(self.terminal_states):
        #     self.selectable_states.remove(terminal_state)

    def transition_to_next_grid(self,s):
        grid_no = s[0]
        goal_pos = s[1:3]
        goal_no = self.non_nested_goal_positions.index(tuple(goal_pos))
        if goal_no == 1:
            next_grid_no = (grid_no+1) *2 -1
        else:
            next_grid_no = (grid_no+1) *2

        return next_grid_no


    def _limit_coordinates(self, s, a):
        """
        Prevent the agent from falling out of the grid world
        """
        if tuple(s) in self.transition_goal_positions:
            next_grid_no = self.transition_to_next_grid(s)
            coord = tuple(  (next_grid_no,) ) + tuple(self.start_state[1:3],)
        else: 
            coord = s[1:3]+a
            coord[0] = min(coord[0], self.shape[1] - 1)
            coord[0] = max(coord[0], 0)
            coord[1] = min(coord[1], self.shape[2] - 1)
            coord[1] = max(coord[1], 0)
            if tuple(coord) in self.wall_grids:
                coord = s
            else:
                coord = tuple( (s[0],)) + tuple(coord)     
        return np.array(coord) 

    def step(self, a):
        self.num_steps += 1
        if (self.num_steps >= self.error_max_steps):
            raise InvalidRunException(f'There have been {self.error_max_steps} steps in this episode, over the maximum allowed number of steps. This means the agent is likely stuck and thus exiting.')
        s = self.current_state
        sp, t = self.next_state(s, self.action_encoding[a])
        r = self.rewards(s, t)
        if t:
            self.update_goals()
        return (r, sp, t)

    # give the rewards associated with a given state, action, next state tuple
    def rewards(self, s, terminal):
        if terminal:            
            # for k in self.goals.keys():
            #     try:
            #         print(self.goals[k]["position"],self.goals[k]["reward"],self.goals[k]["current_reward"],self.goals[k]["reward_sequence_length"],self.goals[k]["iterator"])
            #     except:
            #         print(self.goals[k]["position"],self.goals[k]["current_reward"])

            rewards = [self.goals[k]["current_reward"] for k in self.goals.keys()]
            globals.collector.collect('goal_rewards', rewards)  
            if max(rewards) == 1: # third goal is the only one on:
                best_goal_num = self.goal_pos_to_goal_num[self.third_goal_pos_full]
            else:
                for k in self.goals.keys():
                    if self.goals[k]["current_reward"] == max(rewards):
                        best_goal_num = k
            globals.collector.collect('max_return', self.goals[best_goal_num]["current_reward"] + self.step_to_goals[best_goal_num] * self.step_penalty)  
            for k in self.goals.keys():
                if self.goals[k]["position"]==tuple(s):
                    globals.collector.collect('end_goal', k)  
                    return self.goals[k]["current_reward"]
        else:
            return self.step_penalty
            
    # if iterator reached the end of sequence, generate new sequence and flip reward amount for both goals with not fixed rewards
    def update_goals(self):
        if self.il_counter < self.initial_learning:
            self.il_counter +=1
            increment_iter = False
        else:
            increment_iter = True
        if increment_iter:
            for k in self.goals.keys():
                if "iterator" in self.goals[k]:
                    if self.goals[k]["iterator"] == self.goals[k]["reward_sequence_length"] and self.goals[k]["current_reward"] == self.goals[k]["reward"]:
                        self.gen_reward_sequence(k,self.goals[k]["current_reward"] )
                        self.goals[k]["iterator"] = 0
                    elif self.goals[k]["iterator"] == self.goals[k]["reward_sequence_length"]*4 and self.goals[k]["current_reward"] == 0:
                        self.gen_reward_sequence(k,self.goals[k]["current_reward"] )
                        self.goals[k]["iterator"] = 0
                    else:
                        self.goals[k]["iterator"] += 1

    def gen_reward_sequence(self,terminal_state, previous_terminal_reward):
        #self.goals[terminal_state]["reward_sequence_length"] =  np.random.poisson(lam=self.reward_sequence_length)
        if previous_terminal_reward == 0:
            self.goals[terminal_state]["current_reward"] = self.goals[terminal_state]["reward"]
        else:
            self.goals[terminal_state]["current_reward"] = 0

    # get the next state and termination status
    def next_state(self, s, a):
        # list of terminal state positions (top left, right for leaf nodes and bottom right in all grids)    
        is_done = tuple(s) in self.terminal_state_positions      
        self.current_state = self._limit_coordinates(np.array(s), np.array(a)).astype(int)
        self.current_state = self.start_state if is_done else self.current_state
        self.current_grid_no = self.current_state[0] 
        return self.current_state, is_done


    def state_encoding(self,state):
        return np.ravel_multi_index(np.array(state), self.shape)

