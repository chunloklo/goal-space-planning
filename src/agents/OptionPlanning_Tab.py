from typing import Dict, Union
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue, globals, options

# Dyna but instead of doing Q Planning we do option planning
class OptionPlanning_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OneStepWrapper

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

        self.tau = np.zeros((self.num_states, self.num_actions))
        self.a = -1
        self.x = -1

        # create initial weights
        self.Q = np.zeros((self.num_states, self.num_actions))

        # initiation set is 1, all options are pertinent everywhere
        self.d = np.ones((self.num_states, self.num_options))
        # termination condition is the environment's discount except in terminal states where it's 0
        self.term_cond = np.ones((self.num_states, self.num_options))*params["gamma"]
        for terminal_state_position in self.env.terminal_state_positions:
            self.Q[self.env.state_encoding(terminal_state_position),:]=0

        self.visit_history = {}
        self.option_r = np.zeros((self.num_states + 1, self.num_options))
        self.option_discount = np.zeros((self.num_states + 1, self.num_options))
        # The final +1 accounts for also tracking the transition probability into the terminal state
        self.termination_state_index = self.num_states
        self.option_transition_probability = np.zeros((self.num_states + 1, self.num_options, self.num_states + 1))

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "OptionPlanning_Tab"

    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            a = choice(np.arange(self.num_actions), self.random)
        else:
            a = argmax(self.Q[x,:])
        return a

    def update(self, x, a, xp, r, gamma):
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        action = self.selectAction(xp) if xp != self.termination_state_index else None

        self.tau += 1
        self.tau[x, a] = 0

        # Direct RL
        max_q = 0 if xp == self.termination_state_index else np.max(self.Q[xp,:])

        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*max_q - self.Q[x,a]) 

        self.update_model(x,a,xp,r, gamma)  
        self.planning_step(gamma)

        return action

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
        action_consistent_options = options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        for action_index in action_consistent_options:
            o_index = options.from_action_to_option_index(action_index, self.num_actions)
            if (xp == self.termination_state_index):
                option_termination = 1
            else:
                _, option_termination = self.options[o_index].step(xp)

            self.option_r[x, o_index] += self.alpha * (r + gamma * (1 - option_termination) * self.option_r[xp, o_index] - self.option_r[x, o_index])
            self.option_discount[x, o_index] += self.alpha * (option_termination + (1 - option_termination) * gamma * self.option_discount[xp, o_index] - self.option_discount[x, o_index])
            # accounting for terminal state
            xp_onehot = np.zeros(self.num_states + 1)
            xp_onehot[xp] = 1
            
            # Note that this update is NOT discounted. Use in conjunction with self.option_discount to form the planning estimate
            self.option_transition_probability[x, o_index] += self.alpha * ((option_termination * xp_onehot) + (1 - option_termination) * self.option_transition_probability[xp, o_index] - self.option_transition_probability[x, o_index]) 

    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """
             
        # Additional model planning steps!
        for _ in range(self.model_planning_steps):
            # resample the states
            x = choice(np.array(list(self.visit_history.keys())), self.random)
            visited_actions = list(self.visit_history[x].keys())

            # Improving option model!
            # We're improving the model a ton here (updating all 4 actions). We could reduce this later but let's keep it high for now?
            for a in visited_actions:
                xp, r = self.visit_history[x][a]
                self.update_model(x, a, xp, r, gamma)

        for _ in range(self.planning_steps):
            x = choice(np.array(list(self.visit_history.keys())), self.random)
            visited_actions = list(self.visit_history[x].keys())
            a = choice(np.array(visited_actions), self.random) 
            action_consistent_options = options.get_action_consistent_options(x, a, self.options)

            # Getting the value of the action
            xp, r = self.visit_history[x][a]
            if (xp == self.termination_state_index):
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])

            option_values = [r + gamma * max_q]

            for a in action_consistent_options:
                r = self.option_r[x, a]
                discount = self.option_discount[x, a]
                norm = np.linalg.norm(self.option_transition_probability[x, a], ord=1)
                if (norm != 0):
                    prob = self.option_transition_probability[x, a] / norm
                    # +1 here accounts for the terminal state
                    xp = self.random.choice(self.num_states + 1, p=prob)
                    if (xp == self.num_states):
                        max_q = 0
                    else:
                        max_q = np.max(self.Q[xp,:])
                else:
                    # Since the update will likely not be useful anyways, so its better to not assume any transition probability and just do a plain update.
                    max_q = 0

                option_value = r + discount * max_q 
                option_values.append(option_value)

            
            exp_bonus = self.kappa * np.sqrt(self.tau[x, a])
                
            if (len(option_values) > 0):
                self.Q[x,a] = self.Q[x,a] + self.alpha * (np.max(option_values) + exp_bonus - self.Q[x, a])

    def agent_end(self, x, a, r, gamma):
        self.update(x, a, self.termination_state_index, r, gamma)
                # Debug logging for each model component
        globals.collector.collect('model_r', np.copy(self.option_r))  
        globals.collector.collect('model_discount', np.copy(self.option_discount))  
        globals.collector.collect('model_transition', np.copy(self.option_transition_probability))  
