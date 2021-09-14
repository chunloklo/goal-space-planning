from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice
from src.utils import rlglue

class Option_Given_Q_Tab:
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
        self.epsilon = params['epsilon']

        # create initial weights
        self.Q = np.zeros((self.num_states, self.num_actions+self.num_options))
        # initiation set is 1, all options are pertinent everywhere
        self.d = np.ones((self.num_states, self.num_options))
        # termination condition is the environment's discount except in terminal states where it's 0
        self.term_cond = np.ones((self.num_states, self.num_options))*params["gamma"]
        for terminal_state_position in self.env.terminal_state_positions:
            self.Q[self.env.state_encoding(terminal_state_position),:]=0

        self.option_Qs = [np.zeros((self.num_states, self.num_actions)) for _ in range(self.num_options)]

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Option_Q_Tabular"

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


    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            o = choice(np.arange(self.num_actions+self.num_options), self.random)
        else:
            o = argmax(self.Q[x,:])

        if o>= self.num_actions:
            a, t = self.options[(self.num_actions+self.num_options)-o-1].step(x)
        else:
            a=o
      
        return o

    def update(self, x, o, xp, r, gamma):
        op = self.selectAction(xp)
        #self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*np.max(self.Q[xp,:]) - self.Q[x,a]) 
        self.Q[x, o] = self.Q[x,o] + self.alpha * (r + gamma*np.max(self.Q[xp,:]) - self.Q[x,o]) 
        return op

    def agent_end(self,x,o, r, gamma):
        self.Q[x, o] = self.Q[x,o] + self.alpha * (r  - self.Q[x,o])  

