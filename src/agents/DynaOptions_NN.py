
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from agents.components.learners import ESarsaLambda, QLearner, QLearner_ImageNN
from agents.components.search_control import ActionModelSearchControl_Tabular
from environments.GrazingWorldAdam import GrazingWorldAdamImageFeature, get_pretrained_option_model, state_index_to_coord
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_TB_Tabular, OptionModel_Sutton_Tabular
from src.agents.components.approximators import DictModel
from PyFixedReps.BaseRepresentation import BaseRepresentation
import numpy.typing as npt
from PyFixedReps.Tabular import Tabular
from typing import Dict, Optional, Union, Tuple, Any, TYPE_CHECKING
import jax.numpy as jnp
from src.agents.components.buffers import Buffer, DictBuffer
from collections import namedtuple
from src.environments.GrazingWorldAdam import get_all_transitions

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

# [2022-01-04 chunlok] TODO this class right now is fixed to work only for GrazingWorldAdam
# This means that it will likely fail if you try to use it on another environment (ex the image encoding).
# This is just for speedy experimentations. We'll refactor when we need to.
class DynaOptions_NN:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OptionOneStepWrapper
        self.env = problem.getEnvironment()
        self.image_representation: GrazingWorldAdamImageFeature = problem.get_representation('Image')
        self.tabular_representation: Tabular = problem.get_representation('Tabular')

        self.num_actions = problem.actions
        self.params = problem.params
        self.num_states = self.env.shape[0] * self.env.shape[1]
        self.options = problem.options
        self.num_options = len(problem.options)
        self.random = np.random.RandomState(problem.seed)

        # # define parameter contract
        params = self.params
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.polyak_stepsize = params['polyak_stepsize']
        self.planning_steps = param_utils.check_valid(params['planning_steps'], lambda x: isinstance(x, int) and x > 0)
        # Bonus reward for exploration
        self.kappa = params['kappa']
        # The number of steps that the goal state is not visited before the kappa exploration bonus will be applied to that experience.
        self.kappa_interval = params['kappa_interval']

        self.behaviour_learner = QLearner_ImageNN(self.image_representation.features(), self.num_actions, self.alpha)
        self.batch_size = params['batch_size']

        # Replay Buffer:
        # self.buffer_size = 100000
        # self.buffer = Buffer(self.buffer_size, {'s': (2,), 'x': (8, 12, 1), 'a': (), 'xp': (8, 12, 1), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        # self.model_planning_steps = params['model_planning_steps']

        self.priority_alpha = params['priority_alpha']
        self.option_value_alg = param_utils.parse_param(params, 'option_value_alg', lambda p : p in ['None', 'action_selection', 'bootstrap', 'base_target', 'value_shift'])


        self.tau = np.zeros((self.num_options))
        # # Creating models for actions and options
        self.action_model = DictModel()

        self.dict_buffer = DictBuffer()
        transitions = get_all_transitions()
        
        # Prefilling dict buffer so no exploration is needed
        DictBufferData = namedtuple("DictBufferData", "s a sp r gamma")
        for t in transitions:
            self.dict_buffer.update(DictBufferData(t[0], t[1], t[2], t[3], t[4]))
            self.action_model.update(t[0], t[1], t[2], t[3], t[4])

        self.option_model, self.action_option_model = get_pretrained_option_model()

        # For logging state visitation
        self.state_visitations = np.zeros(self.num_states)

        # Logging update distribution
        self.update_distribution = np.zeros(self.num_states)

        self.cumulative_reward = 0

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "DynaOptions_NN"

    def _sample_option(self, s, o, a: Optional[int] = None):
        x = self.tabular_representation.encode(s)
        if a == None:
            r, discount, transition_prob = self.option_model.predict(x, o)
        else:
            r, discount, transition_prob = self.action_option_model.predict(x, a, o)
        

        # norm = np.linalg.norm(transition_prob, ord=1)
        # prob = transition_prob / norm
        # +1 here accounts for the terminal state
        # xp = self.random.choice(self.num_states + 1, p=prob)
        xp = np.argmax(transition_prob)
        if xp >= self.num_states:
            return r, discount, None
        else:
            coord = state_index_to_coord(xp)
            return r, discount, coord

    def _get_batch_option_values(self, batch):
        ss = batch['s']

        rewards = np.zeros((len(ss), self.num_options, self.num_actions))
        discounts = np.zeros((len(ss), self.num_options, self.num_actions))
        xps = np.zeros((len(ss), self.num_options, self.num_actions, *self.image_representation.features()))
        # # 1 is terminal, 0 is not
        terminal = np.zeros((len(ss), self.num_options, self.num_actions))

        values = np.zeros((len(ss), self.num_options, self.num_actions))

        for i, s in enumerate(ss):
            for o in range(self.num_options):
                for a in range(self.num_actions):
                    # print(s, o, a)
                    r, discount, sp = self._sample_option(s, o, a)
                    rewards[i, o, a] = r
                    discounts[i, o, a] = discount
                    

                    if sp is None:
                        terminal[i, o, a] = 1
                    else:
                        # pass
                        xps[i, o, a] = self.image_representation.encode(sp)

        terminal_map = terminal == 1
        # we might need to use the target action values instead?
        action_values = self.behaviour_learner.get_action_values(xps[~terminal_map])

        values[~terminal_map] = rewards[~terminal_map] + discounts[~terminal_map] * np.max(action_values, axis=1)
        # The terminal rewards in the option model is likely incorrect because the reward schedule is different since we are using a pre-loaded model
        # Therefore, assume that the reward is 0 (though it could be higher)
        
        return values

    def _get_option_values(self, s, a:Optional[int]=None):
        # Generating the experience from the option model
        x = self.tabular_representation.encode(s)

        valid_options = []
        rewards = []
        discounts = []
        xps = []

        # For options that ends with the terminal state
        term_valid_options = []
        term_rewards = []

        for o in range(self.num_options):
            r, discount, sp = self._sample_option(s, o, a)
            if sp != None:
                xps.append(self.image_representation.encode(sp))
                valid_options.append(o)
                rewards.append(r)
                discounts.append(discount)
            else:
                term_valid_options.append(o)
                term_rewards.append(r)
                continue
        
        term_option_values = term_rewards

        if (len(xps) == 0):
            return term_valid_options, term_option_values

        xps = np.array(xps)
        rewards = np.array(rewards)
        discounts = np.array(discounts)

        xps_values = self.behaviour_learner.get_target_action_values(xps)
        max_values = np.max(xps_values, axis=1)

        option_values = rewards + discounts * max_values

        valid_options.append(term_valid_options)
        option_values = np.append(option_values, term_option_values)

        return valid_options, option_values

    # public method for rlglue
    def selectAction(self, s: Any) -> Tuple[int, int] :
        if self.random.uniform(0, 1) < self.epsilon:
            a = self.random.choice(4)
            return a, a
        else:
            x = jnp.expand_dims(self.image_representation.encode(s), axis=0)
            q_values = self.behaviour_learner.get_action_values(x)

            if self.option_value_alg == 'action_selection':
                best_action_value = np.max(q_values)
                valid_options, option_values = self._get_option_values(s)
                best_option_value = np.max(option_values)
                best_option = valid_options[np.argmax(option_values)]

                if best_action_value > best_option_value:
                    a = int(np.argmax(q_values))
                else:
                    x = self.tabular_representation.encode(s)
                    a, _ = options.get_option_info(x, best_option, self.options)
            else:
                a = int(np.argmax(q_values))
            
            return a, a
  
    def planning_update(self):
        # Hard coding this for GrazingWorldAdam for now
        tau_states = [13,31,81]

        samples = self.dict_buffer.sample(self.random, self.batch_size, alpha=self.priority_alpha)
        batch_x = jnp.array([self.image_representation.encode(sample.s) for sample in samples])
        batch_a = jnp.array([sample.a for sample in samples])
        batch_xp = jnp.array([self.image_representation.encode(sample.sp) for sample in samples])
        batch_gamma = jnp.array([sample.gamma for sample in samples])
        # batch_r = np.array([sample.r for sample in samples])

        batch_r = np.zeros(len(samples))
        batch_option_values = np.zeros((len(samples), self.num_options))
        for i, sample in enumerate(samples):
            sp, r, gamma = self.action_model.predict(sample.s, sample.a)
            batch_r[i] = r 


        # exploration bonus
        if not globals.blackboard['in_exploration_phase']:
            for i, sample in enumerate(samples):
                tab_x = int(self.tabular_representation.encode(sample.s))
                try:
                    index = tau_states.index(tab_x)
                    if self.tau[index] > self.kappa_interval:
                    # if globals.blackboard['num_steps_passed'] > 15000 and globals.blackboard['num_steps_passed'] < 30000:
                            batch_r[i] += self.kappa
                except ValueError:
                    pass  
            
        # forming batch
        batch = {}
        batch['x'] = batch_x
        batch['a'] = batch_a
        batch['xp'] = batch_xp
        batch['r'] = batch_r
        batch['gamma'] = batch_gamma

        update_type = 'None'

        if self.option_value_alg == 'base_target':
            update_type = 'base_target'

            batch['best_option_q_sa'] = np.zeros((len(samples)))
            for i, sample in enumerate(samples):
                batch['best_option_q_sa'][i] = np.max(self._get_option_values(sample.s, sample.a)[1])
        
        if self.option_value_alg == 'bootstrap':
            update_type = 'bootstrap'

            batch['best_option_v_sp'] = np.zeros((len(samples)))
            for i, sample in enumerate(samples):
                batch['best_option_v_sp'][i] = np.max(self._get_option_values(sample.sp)[1])

        
        _, td_errors = self.behaviour_learner.update(batch, self.polyak_stepsize, update_type=update_type)  
          
        td_errors = np.abs(td_errors)
        # # Updating TD error in the buffer
        for i, sample in enumerate(samples):
            self.dict_buffer.update(sample, td_errors[i])

        if self.option_value_alg == 'value_shift':
            samples = self.dict_buffer.sample(self.random, 8, alpha=0)

            batch = {}
            batch['x'] = jnp.array([self.image_representation.encode(sample.s) for sample in samples])
            batch['s'] = [sample.s for sample in samples]

            best_option_values = np.max(self._get_batch_option_values(batch), axis=1)

            # print(best_option_values)
            batch['best_option_values'] = best_option_values
            self.behaviour_learner.shift_update(batch)  

            # # print(option_values)
            # print(best_option_values.shape)
            # sdsad
            # for i, sample in enumerate(samples):
            #     for a in range(self.num_actions):
            #         _, option_qs = self._get_option_values(sample.s, a)
            #         option_values[i, a] = np.max(option_qs)

            #     # print(np.maximum(action_values, option_values))
                
            # option_action_values = np.maximum(action_values, option_values)
            # batch['values'] = option_action_values
            # self.behaviour_learner.shift_update(batch)  
            # batch_r = np.array([sample.r for sample in samples])

            # Get option values
            # Pass them off to learner to perform things
            pass


    def update(self, s: Any, o, a, sp: Any, r, gamma, terminal: bool = False):
        self.action_model.update(tuple(s), a, sp, r, gamma)

        if sp is None:
            assert gamma == 0
            sp = (0, 0)
        else:
            sp = tuple(sp)
                    
        # # # forming batch
        # batch = {'x': jnp.array([self.image_representation.encode(s)]), 'a': jnp.array([a]), 'xp': jnp.array([self.image_representation.encode(sp)]), 'r': jnp.array([r]), 'gamma': jnp.array([gamma])}
        # # indexing into 0 since it is an array
        # delta = np.abs(self.behaviour_learner.get_deltas(batch))[0]

        # DictBufferData = namedtuple("DictBufferData", "s a sp r gamma")
        # self.dict_buffer.update(DictBufferData(tuple(s), a, sp, r, gamma), delta)

        tab_x = self.tabular_representation.encode(s)

        self.state_visitations[tab_x] += 1


        # Hard coding this for GrazingWorldAdam for now
        tau_states = [13,31,81]

        # Updating tau and exploration bonus
        if not globals.blackboard['in_exploration_phase']:
            self.tau += 1

        try:
            index = tau_states.index(tab_x)
            self.tau[index] = 0
            # if not globals.blackboard['in_exploration_phase']:
                # print(tab_x)
        except ValueError:
            pass    

        x = self.image_representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.image_representation.encode(sp) if not terminal else jnp.zeros(self.image_representation.features())
    
        # buffer_data = {'s': s, 'x': x, 'a': a, 'xp': xp, 'r': r, 'gamma': gamma}
        # self.buffer.update(buffer_data)

        # s = self.action_model.visited_states()
        # print(len(s))

        # adding exploration bonus to batch
        # if not globals.blackboard['in_exploration_phase']:
        for _ in range(self.planning_steps):
            self.planning_update()
    
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        if not terminal:
            oa_pair = self.selectAction(sp)
        else:
            # the oa pair doesn't matter if the agent arrived in the terminal state.
            oa_pair = None

        # Logging
        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            xs = np.zeros((96, 8, 12, 1))
            for r in range(8):
                for c in range(12):
                    s = (r, c)
                    x = self.image_representation.encode(s)
                    xs[r * 12 + c] = x
            q_values = self.behaviour_learner.get_action_values(xs)
            globals.collector.collect('Q', np.copy(q_values)) 

            # Logging state visitation
            globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
            self.state_visitations[:] = 0

            globals.collector.collect('tau', np.copy(self.tau)) 
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            # print(f'reward rate: {np.copy(self.cumulative_reward) / globals.blackboard["step_logging_interval"]}')
            self.cumulative_reward = 0

        return oa_pair

    def agent_end(self, s, o, a, r, gamma):
        self.update(s, o, a, None, r, gamma, terminal=True)
 
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

        globals.collector.collect('update_distribution', np.copy(self.update_distribution))   
        self.update_distribution[:] = 0
