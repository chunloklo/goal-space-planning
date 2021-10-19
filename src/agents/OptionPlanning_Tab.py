from typing import Dict, Union
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue, globals, options
from src.agents.components.models import DictModel, TabularOptionModel

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
        self.num_options = len(options)
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

        self.termination_state_index = self.num_states
        self.action_model = DictModel()
        self.option_model = TabularOptionModel(self.num_states + 1, self.num_options, self.alpha)

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

        self.update_model(x,a,xp,r)  
        self.planning_step(gamma)

        return action

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """

        self.action_model.update(x, a, xp, r)

        # Update option model
        action_consistent_options = options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        for action_index in action_consistent_options:
            o_index = options.from_action_to_option_index(action_index, self.num_actions)
            if (xp == self.termination_state_index):
                option_termination = 1
            else:
                _, option_termination = self.options[o_index].step(xp)
                
            self.option_model.update(x, o_index, xp, r, env_gamma = self.gamma, option_gamma = 1 - option_termination)

    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """
             
        # Additional model planning steps!
        for _ in range(self.model_planning_steps):
            # resample the states
            x = choice(self.action_model.visited_states(), self.random)
            visited_actions = self.action_model.visited_actions(x)

            # Improving option model!
            # We're improving the model a ton here (updating all 4 actions). We could reduce this later but let's keep it high for now?
            for a in visited_actions:
                xp, r = self.action_model.predict(x, a)
                self.update_model(x, a, xp, r)

        for _ in range(self.planning_steps):
            x = choice(self.action_model.visited_states(), self.random)
            visited_actions = self.action_model.visited_actions(x)
            a = choice(np.array(visited_actions), self.random) 
            action_consistent_options = options.get_action_consistent_options(x, a, self.options)

            # Getting the value of the action
            xp, r = self.action_model.predict(x, a)
            if (xp == self.termination_state_index):
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])

            option_values = [r + gamma * max_q]

            for a in action_consistent_options:
                r, discount, transition_prob = self.option_model.predict(x, a)
                norm = np.linalg.norm(transition_prob, ord=1)
                if (norm != 0):
                    prob = transition_prob / norm
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
        globals.collector.collect('model_r', np.copy(self.option_model.reward_model.weights.reshape(self.num_states + 1, self.num_options, order='F')))  
        globals.collector.collect('model_discount', np.copy(self.option_model.discount_model.weights.reshape(self.num_states + 1, self.num_options, order='F')))  
        globals.collector.collect('model_transition', np.copy(self.option_model.transition_model.weights.reshape(self.num_states + 1, self.num_options, self.num_states + 1, order='F')))  
