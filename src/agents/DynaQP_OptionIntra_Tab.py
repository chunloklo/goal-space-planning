from typing import Dict, Union
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue
from utils import globals

class DynaQP_OptionIntra_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OptionOneStepWrapper

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
        self.planning_steps = params['planning_steps']
        self.model_planning_steps = params['model_planning_steps']
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

        self.visit_history = {}
        self.option_r = np.zeros((self.num_states, self.num_options))
        self.option_discount = np.zeros((self.num_states, self.num_options))
        # The final +1 accounts for also tracking the transition probability into the terminal state
        self.option_transition_probability = np.zeros((self.num_states, self.num_options, self.num_states + 1))

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "DynaQP_OptionIntra_Tab"

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

    def selectAction(self, x) -> tuple[int, int]:
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

        self.tau += 1
        self.tau[x, o] = 0

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

        self.update_model(x,a,xp,r, gamma)  
        self.planning_step(gamma)

        return oa_pair

    def update_model(self, x, a, xp, r, gamma):
        """updates the model 
        
        Returns:
            Nothing
        """
        
        if x not in self.visit_history:
            self.visit_history[x] = {a:(xp,r)}
        else:
            self.visit_history[x][a] = (xp,r)    

        # Update option model
        action_consistent_options = self._get_action_consistent_options(x, a)
        for action_option in action_consistent_options:
            o = action_option - self.num_actions
            _, option_termination = self.options[o].step(x)

            self.option_r[x, o] += self.alpha * (r + gamma * (1 - option_termination) * self.option_r[xp, o] - self.option_r[x, o])
            self.option_discount[x, o] += self.alpha * (option_termination + (1 - option_termination) * gamma * self.option_discount[xp, o] - self.option_discount[x, o])
            # accounting for terminal state
            xp_onehot = np.zeros(self.num_states + 1)
            xp_onehot[xp] = 1
            
            # Note that this update is NOT discounted. Use in conjunction with self.option_discount to form the planning estimate
            self.option_transition_probability[x, o] += self.alpha * ((option_termination * xp_onehot) + (1 - option_termination) * self.option_transition_probability[xp, o] - self.option_transition_probability[x, o]) 

    def _get_action_consistent_options(self, x: int, a: Union[list, int]):
        action_consistent_options = []
        for o in range(self.num_actions, self.num_actions + self.num_options):
            a_o, t = self._get_option_info(x, o)
            if (isinstance(a, int)):
                if (a_o == a):
                    action_consistent_options.append(o)
            if (isinstance(a, list)):
                if (a_o in a):
                    action_consistent_options.append(o)
        return action_consistent_options

    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        for _ in range(self.planning_steps):
            x = choice(np.array(list(self.visit_history.keys())), self.random)
            visited_actions = list(self.visit_history[x].keys())
            action_consistent_options = self._get_action_consistent_options(x, visited_actions)
            available_actions = visited_actions + action_consistent_options

            a = choice(np.array(available_actions), self.random) 
            if (a < self.num_actions):
                xp, r = self.visit_history[x][a]
                discount = gamma

                if (xp == -1):
                    max_q = 0
                else:
                    max_q = np.max(self.Q[xp,:])
            else:
                r = self.option_r[x, a - self.num_actions]
                discount = self.option_discount[x, a - self.num_actions]
                norm = np.linalg.norm(self.option_transition_probability[x, a - self.num_actions])
                if (norm != 0):
                    prob = self.option_transition_probability[x, a - self.num_actions] / norm
                    # +1 here accounts for the terminal state
                    xp = self.random.choice(self.num_states + 1, p=prob)
                    if (xp == self.num_states):
                        max_q = 0
                    else:
                        max_q = np.max(self.Q[xp,:])
                else:
                    # Since the update will likely not be useful anyways, so its better to not assume any transition probability and just do a plain update.
                    max_q = 0
            
            r += self.kappa * np.sqrt(self.tau[x, a])

            self.Q[x,a] = self.Q[x,a] + self.alpha * (r + discount * max_q - self.Q[x, a])

        for _ in range(self.model_planning_steps):
            # Improving option model!
            a = choice(np.array(visited_actions), self.random) 
            xp, r = self.visit_history[x][a]
            self.update_model(x, a, xp, r, gamma)

    def agent_end(self, x, o, a, r, gamma):
        self.update(x, o, a, -1, r, gamma)
                # Debug logging for each model component
        globals.collector.collect('model_r', np.copy(self.option_r))  
        globals.collector.collect('model_discount', np.copy(self.option_discount))  
        globals.collector.collect('model_transition', np.copy(self.option_transition_probability))  
