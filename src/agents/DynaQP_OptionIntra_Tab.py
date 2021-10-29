from typing import Dict, Union
import numpy as np
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_TB_Tabular, OptionModel_Sutton_Tabular
from src.agents.components.approximators import DictModel

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
        self.lmbda = params['lambda']
        self.model_alg =  param_utils.parse_param(params, 'option_model_alg', lambda p : p in ['sutton'])

        # Whether to plan with all action-consistent options when planning
        self.all_option_planning_update = params['all_option_planning_update']
        
        # Whether to plan with current state.
        self.plan_with_current_state = param_utils.parse_param(params, 'planning_method', lambda p : p in ['random', 'current', 'close'])
        
        self.tau = np.zeros((self.num_states, self.num_actions + self.num_options))
        self.a = -1
        self.x = -1

        # create initial weights
        self.Q = np.zeros((self.num_states, self.num_actions+self.num_options))

        self.termination_state_index = self.num_states

        # Creating models for actions and options
        self.action_model = DictModel()
        # The final +1 accounts for also tracking the transition probability into the terminal state

        # self.option_model = OptionModel_TB_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
        if self.model_alg == 'sutton':
            self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
        else:
            raise NotImplementedError(f'option_model_alg for {self.model_alg} is not implemented')

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

    def get_policy(self, x: int):
        prob = np.zeros(self.num_actions)
        prob += self.epsilon / (self.num_actions + self.num_options)

        # Adding epsilon prob for options
        for o in range(self.num_options):
            a, _ = options.get_option_info(x, o, self.options)
            prob[a] += self.epsilon / (self.num_actions + self.num_options)
        
        o = argmax(self.Q[x, :])
        
        if o >= self.num_actions:
            o, _ = options.get_option_info(x, options.from_action_to_option_index(o, self.num_actions), self.options)

        prob[o] += 1 - self.epsilon
        return prob

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
        self.planning_step(gamma, x, xp, self.plan_with_current_state)

        return oa_pair
    
    def option_model_planning_update(self, x, a, xp ,r):
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, self.gamma, self.alpha)
        else:
            raise NotImplementedError(f'Planning update for {type(self.option_model)} is not implemented')

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
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, self.gamma, self.alpha)
        else:
            raise NotImplementedError(f'Update for {type(self.option_model)} is not implemented')

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
            transition_prob = np.clip(transition_prob, a_min = 0, a_max = None)
            norm = np.linalg.norm(transition_prob, ord=1)
            if (norm != 0):
                prob = transition_prob / norm
                # +1 here accounts for the terminal state
                # print(transition_prob)
                # print(option_index)
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

    def planning_step(self,gamma, x, xp, plan_current_state):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """
        
        # Additional model planning steps if we need them. Usually this is set to 0 though.
        for _ in range(self.model_planning_steps):
            # resample the states
            plan_x = choice(np.array(self.action_model.visited_states()), self.random)
            visited_actions = self.action_model.visited_actions(plan_x)

            # Improving option model!
            # We're improving the model a ton here (updating all 4 actions). We could reduce this later but let's keep it high for now?
            for a in visited_actions:
                xp, r = self.action_model.predict(plan_x, a)
                self.option_model_planning_update(plan_x, a, xp, r)

        if plan_current_state == "close":
            visited_states, distances = [], []
            for k in self.action_model.visited_states():
                visited_states.append(k)
                distances.append(self.distance_from_goal[k])
            normed_distances = [i/sum(distances) for i in distances]

        for _ in range(self.planning_steps):
            if plan_current_state=="random":
                plan_x = choice(np.array(self.action_model.visited_states()), self.random)
            elif plan_current_state =="current":
                visited_states = list(self.action_model.visited_states())
                if (xp in list(visited_states)):
                    plan_x = xp
                else:
                    # Random if you haven't visited the next state yet
                    plan_x = x
            elif plan_current_state =="close":
                plan_x = self.random.choice(np.array(list(visited_states)), p = normed_distances)
            visited_actions = self.action_model.visited_actions(plan_x)
            
            if self.all_option_planning_update:
                # Pick a random action, then update the action and matching options
                action_to_update = choice(np.array(visited_actions), self.random)
                action_consistent_options = options.get_action_consistent_options(plan_x, action_to_update, self.options, convert_to_actions=True, num_actions=self.num_actions)
                update_options = [action_to_update] + action_consistent_options
            else:
                # Pick a random action/option within all eligable action/options
                action_consistent_options = options.get_action_consistent_options(plan_x, visited_actions, self.options, convert_to_actions=True, num_actions=self.num_actions)
                available_actions = visited_actions + action_consistent_options
                update_options = [choice(np.array(available_actions), self.random)]
            
            for a in update_options: 
                self._planning_update(gamma, plan_x, a)

    def agent_end(self, x, o, a, r, gamma):
        self.update(x, o, a, self.termination_state_index, r, gamma)
        self.counter = 0
        
        self.option_model.episode_end()

        globals.collector.collect('Q', np.copy(self.Q))
        globals.collector.collect('action_selected', self.log_action_selected.copy())
        self.log_action_selected = []  