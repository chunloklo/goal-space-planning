from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue

class Dyna_Option_Givenqp_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):

        self.wrapper_class = rlglue.OptionFullExecuteWrapper

        self.env = env
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.num_states = self.env.nS
        self.options = options
        self.num_options = len(self.env.terminal_state_positions)
        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = 1
        self.planning_steps = params['planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']

        self.tau = np.zeros(((int(features/self.num_actions + self.num_options), self.num_actions + self.num_options)))
        self.a = -1
        self.x = -1

        # create initial weights
        self.Q = np.zeros((self.num_states, self.num_actions+self.num_options))
        # initiation set is 1, all options are pertinent everywhere
        self.d = np.ones((self.num_states, self.num_options))
        # termination condition is the environment's discount except in terminal states where it's 0
        self.term_cond = np.ones((self.num_states, self.num_options))*params["gamma"]
        for terminal_state_position in self.env.terminal_state_positions:
            self.Q[self.env.state_encoding(terminal_state_position),:]=0

        self.model = {}
    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Dyna_Optionqp_Tab"

    def is_option(self, o):
        if o>= self.num_actions:
            return True
        else:
            return False

    # Returns the option/action
    def get_action(self, x, o):
        if o>= self.num_actions:
            a, t = self.options[(self.num_actions+self.num_options)-o-1].step(x)
            return a, t
        else:
            return o, False

    def selectAction(self, x) :
        p = self.random.rand()
        if p < self.epsilon:
            o = choice(np.arange(self.num_actions+self.num_options), self.random)
        else:
            o = argmax(self.Q[x,:])

        return o

    def update(self, x,o,xp, r, gamma):
        op = self.selectAction(xp)
        self.tau += 1
        self.tau[x, o] = 0
        # Direct RL
        max_q = 0 if xp == -1 else np.max(self.Q[xp,:])
        self.Q[x, o] = self.Q[x,o] + self.alpha * (r + gamma*max_q - self.Q[x,o]) 

        self.update_model(x,o,xp,r)  
        self.planning_step(gamma)

        return op

    def update_model(self, x, o, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        
        if x not in self.model:
            self.model[x] = {o:(xp,r)}
        else:
            self.model[x][o] = (xp,r)
            

    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        for i in range(self.planning_steps):
            x = choice(np.array(list(self.model.keys())), self.random)
            a = choice(np.array(list(self.model[x].keys())), self.random) 
            xp, r = self.model[x][a]
            r+= self.kappa * np.sqrt(self.tau[x, a])
            if xp ==-1:
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
            
            self.Q[x,a] = self.Q[x,a] + self.alpha * (r + gamma * max_q - self.Q[x, a])
            
            # if self.Q[x,a]>50:
            #     print(self.Q[x,a])
            #     print(x,a)
            

    def agent_end(self, x, o, r, gamma):
        self.update(x, o, -1, r, gamma)
