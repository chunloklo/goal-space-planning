from typing import Dict, Union
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import DictModel, TabularModel, TabularOptionModel

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
        self.num_options = len(options)
        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.model_planning_steps = params['model_planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']

        # Whether to plan with all action-consistent options when planning
        self.all_option_planning_update = params['all_option_planning_update']
        
        # Whether to plan with current state.
        self.plan_with_current_state = param_utils.parse_param(params, 'planning_method', ['random', 'current', 'close'])
        
        self.tau = np.zeros((self.num_states, self.num_actions + self.num_options))
        self.a = -1
        self.x = -1

        # create initial weights
        self.Q = np.zeros((self.num_states, self.num_actions+self.num_options))

        self.termination_state_index = self.num_states

        # Creating models for actions and options
        self.action_model = DictModel()
        # The final +1 accounts for also tracking the transition probability into the terminal state
        self.option_model = TabularOptionModel(self.num_states + 1, self.num_options, self.alpha)

        self.distance_from_goal = {}

        # logging temp vars
        self.log_action_selected = []

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "DynaQP_OptionIntra_Tab"

    def selectAction(self, x) :

        p = self.random.rand()
        if p < self.epsilon:
            o = choice(np.arange(self.num_actions+self.num_options), self.random)
        else:
            o = argmax(self.Q[x,:])

        if o>= self.num_actions:
            a, t = options.get_option_info(x, options.from_action_to_option_index(o, self.num_actions), self.options)
        else:
            a=o
    
        return o,a

    def update(self, x, o, a, xp, r, gamma):
        self.log_action_selected.append(o)

        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        oa_pair = self.selectAction(xp) if xp != self.termination_state_index else None

        self.tau += 1
        self.tau[x, o] = 0

        # Direct RL
        max_q = 0 if xp == self.termination_state_index else np.max(self.Q[xp,:])

        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*max_q - self.Q[x,a]) 

        # Intra-option value learning
        # This assumes that the option policies are markov
        action_consistent_options = options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        for o in action_consistent_options:
            _, t = options.get_option_info(x, options.from_action_to_option_index(o, self.num_actions), self.options)
            # not strictly needed because gamma should already cancel out the term on termination
            # but it prevents some unneeded computation that could error out with weird indexing.
            
            arrival_value = (1 - t) * self.Q[xp,o] + t * max_q if xp != self.termination_state_index else 0
            self.Q[x, o] = self.Q[x,o] + self.alpha * (r + gamma * arrival_value - self.Q[x,o]) 

        self.update_model(x,a,xp,r)  


        self.planning_step(gamma, x, self.plan_with_current_state)

        return oa_pair

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        if self.plan_with_current_state == "close":
            if x == self.env.state_encoding(self.env.start_state):
                self.distance_from_goal[x] = 1
            if xp != self.termination_state_index and (xp not in self.distance_from_goal or self.distance_from_goal[xp] > self.distance_from_goal[x] +1):
                self.distance_from_goal[xp] = self.distance_from_goal[x] +1

        self.action_model.update(x, a, xp, r)

        # Update option model
        action_consistent_options = options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        for action_option in action_consistent_options:
            o = options.from_action_to_option_index(action_option, self.num_actions)
            if (xp == self.termination_state_index):
                option_termination = 1
            else:
                # _, termination = self.options[o].step(x)
                _, option_termination = self.options[o].step(xp)
                # option_termination = termination or option_termination
            # print(self.options[o].step(xp))
            self.option_model.update(x, o, xp, r, self.gamma, 1 - option_termination)

    def _planning_update(self, gamma, x, o):
        if (o < self.num_actions):
            xp, r = self.action_model.predict(x, o)
            discount = gamma
            if (xp == self.termination_state_index):
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
        else:
            option_index = options.from_action_to_option_index(o, self.num_actions)
            r, discount, transition_prob = self.option_model.predict(x, option_index)
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
        
        r += self.kappa * np.sqrt(self.tau[x, o])
        self.Q[x,o] = self.Q[x,o] + self.alpha * (r + discount * max_q - self.Q[x, o])

    def planning_step(self,gamma,current_x, plan_current_state):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """
        
        # Additional model planning steps!
        for _ in range(self.model_planning_steps):
            # resample the states
            x = choice(np.array(self.action_model.visited_states()), self.random)
            visited_actions = self.action_model.visited_actions(x)

            # Improving option model!
            # We're improving the model a ton here (updating all 4 actions). We could reduce this later but let's keep it high for now?
            for a in visited_actions:
                xp, r = self.action_model.predict(x, a)
                self.update_model(x, a, xp, r)

        if plan_current_state == "close":
            visited_states, distances = [], []
            for k in self.action_model.visited_states():
                visited_states.append(k)
                distances.append(self.distance_from_goal[k])
            normed_distances = [i/sum(distances) for i in distances]

        for _ in range(self.planning_steps):
            if plan_current_state=="random":
                x = choice(np.array(self.action_model.visited_states()), self.random)
            elif plan_current_state =="current":
                x = current_x
            elif plan_current_state =="close":
                x = self.random.choice(np.array(list(visited_states)), p = normed_distances)
            visited_actions = self.action_model.visited_actions(x)
            
            if self.all_option_planning_update:
                # Pick a random action, then update the action and matching options
                action_to_update = choice(np.array(visited_actions), self.random)
                action_consistent_options = options.get_action_consistent_options(x, action_to_update, self.options, convert_to_actions=True, num_actions=self.num_actions)
                update_options = [action_to_update] + action_consistent_options
            else:
                # Pick a random action/option within all eligable action/options
                action_consistent_options = options.get_action_consistent_options(x, visited_actions, self.options, convert_to_actions=True, num_actions=self.num_actions)
                available_actions = visited_actions + action_consistent_options
                update_options = [choice(np.array(available_actions), self.random)]
            
            for a in update_options: 
                self._planning_update(gamma, x, a)

    def agent_end(self, x, o, a, r, gamma):
        self.update(x, o, a, self.termination_state_index, r, gamma)
        
        self.counter = 0

        # Debug logging for each model component
        globals.collector.collect('model_r', np.copy(self.option_model.reward_model.weights.reshape(self.num_states + 1, self.num_options, order='F')))  
        globals.collector.collect('model_discount', np.copy(self.option_model.discount_model.weights.reshape(self.num_states + 1, self.num_options, order='F')))  
        globals.collector.collect('model_transition', np.copy(self.option_model.transition_model.weights.reshape(self.num_states + 1, self.num_options, self.num_states + 1, order='F')))  

        globals.collector.collect('action_selected', self.log_action_selected.copy())

        self.log_action_selected = []  