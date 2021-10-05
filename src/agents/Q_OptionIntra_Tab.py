from typing import Dict, Union
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue
from src.utils import globals

class Q_OptionIntra_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OptionOneStepWrapper

        self.env = env
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.num_states = self.env.nS
        self.options = options
        self.num_options = len(options)
        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']


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

        self.option_r = np.zeros((self.num_states, self.num_options))
        self.option_discount = np.zeros((self.num_states, self.num_options))
        # The final +1 accounts for also tracking the transition probability into the terminal state
        self.option_transition_probability = np.zeros((self.num_states, self.num_options, self.num_states + 1))

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Q_OptionIntra_Tab"

    # Make sure that o is in range of [self.num_actions, self.num_actions + self.num_options)
    def _get_option_info(self, x, o):
        return self.options[o - self.num_actions].step(x)

    # Returns the option/action
    def get_action(self, x, o):
        if o>= self.num_actions:
            a, t = self._get_option_info(x, o)
            return a, t
        else:
            return o, False

    def selectAction(self, x) :
        p = self.random.rand()
        if p < self.epsilon:
            o = choice(np.arange(self.num_actions+self.num_options), self.random)
        else:
            o = argmax(self.Q[x,:])

        if o>= self.num_actions:
            a, t = self._get_option_info(x, o)
        else:
            a=o
    
        return o,a

    def update(self, x, o, a, xp, r, gamma):
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        oa_pair = self.selectAction(xp) if xp != -1 else (-1, -1)

        # Direct RL
        max_q = 0 if xp == -1 else np.max(self.Q[xp,:])

        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*max_q - self.Q[x,a]) 

        # Intra-option value learning
        # This assumes that the option policies are markov
        action_consistent_options = self._get_action_consistent_options(x, a)
        for o in action_consistent_options:
            _, t = self._get_option_info(x, o)
            # not strictly needed because gamma should already cancel out the term on termination
            # but it prevents some unneeded computation that could error out with weird indexing.
            arrival_value = (1 - t) * self.Q[xp,o] + t * max_q if xp != -1 else 0
            self.Q[x, o] = self.Q[x,o] + self.alpha * (r + gamma * arrival_value - self.Q[x,o]) 

        return oa_pair

    def _get_action_consistent_options(self, x: int, a: Union[list, int]):
        action_consistent_options = []
        for o in range(self.num_actions, self.num_actions + self.num_options):
            a_o, t = self._get_option_info(x, o)
            if (t):
                action_consistent_options.append(o)
            else:
                if (isinstance(a, (int, np.integer))):
                    if (a_o == a):
                        action_consistent_options.append(o)
                elif (isinstance(a, list)):
                    if (a_o in a):
                        action_consistent_options.append(o)
                else:
                    raise NotImplementedError(f'_get_action_consistent_option does not yet supports this type {type(a)}. Feel free to add support!')
        return action_consistent_options

    def agent_end(self, x, o, a, r, gamma):
        self.update(x, o, a, -1, r, gamma)
                # Debug logging for each model component
        globals.collector.collect('model_r', np.copy(self.option_r))  
        globals.collector.collect('model_discount', np.copy(self.option_discount))  
        globals.collector.collect('model_transition', np.copy(self.option_transition_probability))  
