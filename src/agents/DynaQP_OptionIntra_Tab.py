from typing import Dict, Union
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue
from src.utils import globals
from src.utils import options

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
        self.plan_with_current_state = params.get('plan_with_current_state', False)
        
        self.tau = np.zeros((self.num_states, self.num_actions + self.num_options))
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
        self.option_r = np.zeros((self.num_states + 1, self.num_options))
        self.option_discount = np.zeros((self.num_states + 1, self.num_options))
        # The final +1 accounts for also tracking the transition probability into the terminal state
        self.termination_state_index = self.num_states
        self.option_transition_probability = np.zeros((self.num_states + 1, self.num_options, self.num_states + 1))

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

        self.update_model(x,a,xp,r, gamma)  

        if (self.plan_with_current_state):
            self.current_state_planning(gamma, x)
        else:
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
        action_consistent_options = options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        for action_option in action_consistent_options:
            o = options.from_action_to_option_index(action_option, self.num_actions)
            if (xp == self.termination_state_index):
                option_termination = 1
            else:
                # _, termination = self.options[o].step(x)
                _, option_termination = self.options[o].step(xp)
                # option_termination = termination or option_termination

            self.option_r[x, o] += self.alpha * (r + gamma * (1 - option_termination) * self.option_r[xp, o] - self.option_r[x, o])
            self.option_discount[x, o] += self.alpha * (option_termination + (1 - option_termination) * gamma * self.option_discount[xp, o] - self.option_discount[x, o])
            # accounting for terminal state
            xp_onehot = np.zeros(self.num_states + 1)
            xp_onehot[xp] = 1
            
            # Note that this update is NOT discounted. Use in conjunction with self.option_discount to form the planning estimate
            self.option_transition_probability[x, o] += self.alpha * ((option_termination * xp_onehot) + (1 - option_termination) * self.option_transition_probability[xp, o] - self.option_transition_probability[x, o]) 
            # if (self.option_transition_probability[12, 0][12] != 0) :
            #     print("hmm")
            #     print(options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions))
            #     print(f'x: {x}, xp: {xp}, o: {o}, a {a} locations: {np.where(self.option_transition_probability[x, o] != 0)}')
            #     print(np.where(self.option_transition_probability[xp, o] != 0))
            #     exit()

    def _planning_update(self, gamma, x, o):
        if (o < self.num_actions):
            xp, r = self.visit_history[x][o]
            discount = gamma

            if (xp == self.termination_state_index):
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
        else:
            option_index = options.from_action_to_option_index(o, self.num_actions)
            r = self.option_r[x, option_index]
            discount = self.option_discount[x, option_index]
            norm = np.linalg.norm(self.option_transition_probability[x, option_index], ord=1)
            if (norm != 0):
                prob = self.option_transition_probability[x, option_index] / norm
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


    def current_state_planning(self, gamma, x):
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

        visited_actions = list(self.visit_history[x].keys())
        for _ in range(self.planning_steps):
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
                # Debug logging for each model component
        globals.collector.collect('model_r', np.copy(self.option_r))  
        globals.collector.collect('model_discount', np.copy(self.option_discount))  
        globals.collector.collect('model_transition', np.copy(self.option_transition_probability))  
        globals.collector.collect('action_selected', self.log_action_selected.copy())

        self.log_action_selected = []  
