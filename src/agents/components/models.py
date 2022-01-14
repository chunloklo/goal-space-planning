from typing import Any, List, Dict, Tuple
import numpy as np
import numpy.typing as npt

from src.utils import numpy_utils, param_utils
from src.agents.components.traces import Trace
from src.agents.components.approximators import TabularApproximator
from src.utils.feature_utils import stacked_tabular_features, stacked_features
from src.utils import globals
from src.utils import options
from src.utils.Option import Option
import haiku as hk
import jax
import optax
import jax.numpy as jnp
import copy

class RewardModel_ImageNN_funcs():
    def __init__(self, learning_rate: float):
        def r_function(states):
            mlp = hk.Sequential([
                hk.Conv2D(40, [4, 4], stride=1), jax.nn.relu,
                hk.Flatten(),
                hk.Linear(20), jax.nn.relu,
                hk.Linear(1),
            ])
            return mlp(states) 
        self.f_rfunc = hk.without_apply_rng(hk.transform(r_function))
        self.f_opt = optax.adam(learning_rate)

        def get_reward(params: hk.Params, x: Any):
            reward = self.f_rfunc.apply(params, x)
            return reward

        def loss(params: hk.Params, data):
            r = data['r']
            x = data['x']
            r_pred = self.f_rfunc.apply(params, x)
            return  jnp.mean(jnp.square(r - r_pred))

        def update(params: hk.Params, opt_state, data):
            grads = jax.grad(loss)(params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        self.f_get_r = jax.jit(get_reward)
        self.f_update = jax.jit(update)        
        return

class RewardModel_ImageNN():
    def __init__(self, image_size: Tuple[int, int], learning_rate: float):
        # One color channel for B&W
        self.image_size: int = image_size
        self.funcs = RewardModel_ImageNN_funcs(learning_rate)

        dummy_state = jnp.zeros((1, *self.image_size))
        self.params = self.funcs.f_rfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.funcs.f_opt.init(self.params)
    
    def get_reward(self, x: npt.ArrayLike) -> np.ndarray:
        reward = self.funcs.f_get_r(self.params, x)
        return reward

    def update(self, data):
        self.params, self.opt_state = self.funcs.f_update(self.params, self.opt_state, data)

class OptionActionModel_Sutton_Tabular():
    def __init__(self, num_states: int, num_actions: int, num_options: int, options: List[Option]):
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.num_options: int = num_options
        self.options = options
        
        self.reward_model = np.zeros((num_states, num_actions, num_options))
        self.discount_model = np.zeros((num_states, num_actions, num_options))
        self.transition_model = np.zeros((num_states, num_actions, num_options, num_states))
        
    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        """updates the model 
        
        Returns:
            Nothing
        """

        for o in range(self.num_options):
            if env_gamma == 0:
                option_termination = 1
                # If terminating, what the options do at xp doesn't matter, so just set to some number
                ap = 0
            else:
                ap, option_termination = self.options[o].step(xp)

            self.reward_model[x, a, o] += step_size * (r + env_gamma * (1 - option_termination) * self.reward_model[xp, ap, o] - self.reward_model[x, a, o])
            self.discount_model[x, a, o] += step_size * (option_termination + (1 - option_termination) * env_gamma * self.discount_model[xp, ap, o] - self.discount_model[x, a, o])
            
            xp_onehot = numpy_utils.create_onehot(self.num_states, xp)
            # Note that this update is NOT discounted. Use in conjunction with self.option_discount to form the planning estimate
            self.transition_model[x, a, o] += step_size * ((option_termination * xp_onehot) + (1 - option_termination) * self.transition_model[xp, ap, o] - self.transition_model[x, a, o]) 

    def predict(self, x: npt.ArrayLike, a:int, o: int) -> Any:
        reward = self.reward_model[x, a, o]
        discount = self.discount_model[x, a, o]
        next_state_prob = self.transition_model[x, a, o]
        return reward, discount, next_state_prob

    def episode_end(self):
        # Logging
        globals.collector.collect('action_model_r', np.copy(self.reward_model))
        globals.collector.collect('action_model_discount', np.copy(self.discount_model))
        globals.collector.collect('action_model_transition', np.copy(self.transition_model))
        pass

class OptionModel_Sutton_Tabular():
    def __init__(self, num_states: int, num_actions: int, num_options: int, options: List[Option]):
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.num_options: int = num_options
        self.options = options
        
        self.reward_model = np.zeros((num_states, num_options))
        self.discount_model = np.zeros((num_states, num_options))
        self.transition_model = np.zeros((num_states, num_options, num_states))
        
    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float, s:Any=None):
        """updates the model 
        
        Returns:
            Nothing
        """
        if s is None:
            action_consistent_options = options.get_action_consistent_options(x, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        else:
            action_consistent_options = options.get_action_consistent_options(s, a, self.options, convert_to_actions=True, num_actions=self.num_actions)
        for action_option in action_consistent_options:
            o = options.from_action_to_option_index(action_option, self.num_actions)
            if env_gamma == 0:
                option_termination = 1
            else:
                # _, termination = self.options[o].step(x)
                _, option_termination = self.options[o].step(xp)
                # option_termination = termination or option_termination

            self.reward_model[x, o] += step_size * (r + env_gamma * (1 - option_termination) * self.reward_model[xp, o] - self.reward_model[x, o])
            self.discount_model[x, o] += step_size * (option_termination + (1 - option_termination) * env_gamma * self.discount_model[xp, o] - self.discount_model[x, o])
            
            xp_onehot = numpy_utils.create_onehot(self.num_states, xp)
            # Note that this update is NOT discounted. Use in conjunction with self.option_discount to form the planning estimate
            self.transition_model[x, o] += step_size * ((option_termination * xp_onehot) + (1 - option_termination) * self.transition_model[xp, o] - self.transition_model[x, o]) 

    def predict(self, x: npt.ArrayLike, o: int) -> Any:
        reward = self.reward_model[x, o]
        discount = self.discount_model[x, o]
        next_state_prob = self.transition_model[x, o]
        return reward, discount, next_state_prob

    def episode_end(self):
        # Logging
        globals.collector.collect('model_r', np.copy(self.reward_model))
        globals.collector.collect('model_discount', np.copy(self.discount_model))
        globals.collector.collect('model_transition', np.copy(self.transition_model))
        pass

# Differeniating combined models and separate action/option models
class CombinedModel():
    pass

class CombinedModel_ESarsa_Tabular(CombinedModel):
    def __init__(self, num_states: int, num_actions: int, num_options: int, options: List[Option]):
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.num_options: int = num_options
        self.options = options
        
        self.reward_model = np.zeros((num_states, num_actions, 1 + num_options))
        self.discount_model = np.zeros((num_states, num_actions, 1 + num_options))
        self.transition_model = np.zeros((num_states, num_actions, 1 + num_options, num_states))
        
    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        """updates the model 
        
        Returns:
            Nothing
        """
        xp_onehot = numpy_utils.create_onehot(self.num_states, xp)

        # Update action models first
        self.reward_model[x, a, self.num_options] += step_size * (r - self.reward_model[x, a, self.num_options])
        self.discount_model[x, a, self.num_options] += step_size * (env_gamma - self.discount_model[x, a, self.num_options])
        self.transition_model[x, a, self.num_options] += step_size * (xp_onehot - self.transition_model[x, a, self.num_options])

        option_policies = options.get_option_policies_prob(xp, self.options, self.num_actions)
        option_terminations = options.get_option_term(xp, self.options)
        for o in range(self.num_options):
            xp_policy = option_policies[o]
            option_termination = option_terminations[o]

            xp_reward_average = np.average(self.reward_model[xp, :, o], weights=xp_policy)
            self.reward_model[x, a, o] += step_size * (r + env_gamma * (1 - option_termination) * xp_reward_average - self.reward_model[x, a, o])

            xp_discount_average = np.average(self.discount_model[xp, :, o], weights=xp_policy)
            self.discount_model[x, a, o] += step_size * (option_termination + (1 - option_termination) * env_gamma * xp_discount_average - self.discount_model[x, a, o])
            
            xp_transition_average = np.average(self.transition_model[xp, :, o], weights=xp_policy, axis=0)
            self.transition_model[x, a, o] += step_size * ((option_termination * xp_onehot) + (1 - option_termination) * xp_transition_average -  self.transition_model[x, a, o]) 

    def predict(self, x: npt.ArrayLike, o: int) -> Any:
        if (o >= self.num_actions):
            o_index = options.from_action_to_option_index(o, self.num_actions)
            policy = options.get_option_policy_prob(x, self.options[o_index], self.num_actions)

            reward = np.average(self.reward_model[x, :, o_index], weights=policy)
            discount = np.average(self.discount_model[x, :, o_index], weights=policy)
            next_state_prob = np.average(self.transition_model[x, :, o_index], weights=policy, axis=0)
            
        else:
            reward = self.reward_model[x, o, self.num_options]
            discount = self.discount_model[x, o, self.num_options]
            next_state_prob = self.transition_model[x, o, self.num_options]
        return reward, discount, next_state_prob

    def episode_end(self):
        # Logging
        # Processing for logging
        reward = np.zeros((self.num_states, self.num_options))
        discount = np.zeros((self.num_states, self.num_options))
        transition =  np.zeros((self.num_states, self.num_options, self.num_states))
        
        for x in range(0, 100):
            for o in range(self.num_options):
                xo_reward, xo_discount, xo_transition_prob = self.predict(x, options.from_option_to_action_index(o, self.num_actions))
                reward[x, o] = xo_reward
                discount[x, o] = xo_discount
                transition[x, o] = xo_transition_prob

class OptionModel_TB_Tabular():
    def __init__(self, num_state_features: int, num_actions: int, num_options: int, options: List[Option]):
        self.num_state_features: int = num_state_features
        self.num_actions: int = num_actions
        self.num_options: int = num_options
        self.options: List[Option] = options

        # Assigns slices corresponding to target output
        temp_index = 0
        self.reward_indices = slice(temp_index, temp_index := temp_index + self.num_options)
        self.discount_indices = slice(temp_index,  temp_index := temp_index + self.num_options)
        # Assume that the SR vec size is the same as the state encoding size
        self.transition_indices = slice(temp_index, temp_index := temp_index + self.num_options * self.num_state_features)

        self.num_update_features: int = temp_index

        self.trace = Trace(self.num_state_features * self.num_actions, self.num_update_features, 'tabular')

        # Could change based on what option model we want.
        self.approximator = TabularApproximator(self.num_state_features * self.num_actions, self.num_update_features)

    def get_predictions_for_all_actions(self, x: int) -> np.ndarray:
        x_predictions = []
        for a in range(self.num_actions):
            xpap_index = stacked_tabular_features(x, a, self.num_state_features)
            x_predictions.append(self.approximator.predict(xpap_index))
        x_predictions = np.array(x_predictions).transpose()
        return x_predictions

    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, lmbda: float, step_size: float):
        # Output ordering: [Reward | discount | transition]
        # Option ordering within ^ [o0 | o1 | o2 | o3]
        xa_index = stacked_tabular_features(x, a, self.num_state_features)
        x_predictions = self.approximator.predict(xa_index)
        xp_predictions = self.get_predictions_for_all_actions(xp)
  
        option_gamma = 1 - options.get_option_term(xp, self.options)
        option_policies = options.get_option_policies_prob(xp, self.options, self.num_actions)

        transition_gamma = np.repeat(option_gamma, self.num_state_features)
        xp_onehot = numpy_utils.create_onehot(self.num_state_features, xp)
        target_immediate = np.concatenate(([r] * self.num_options, 1 - option_gamma, (1 - transition_gamma) * np.tile(xp_onehot, self.num_options)))

        target_discount = np.concatenate((env_gamma * option_gamma, env_gamma *  option_gamma, transition_gamma))

        # ESarsa target based on policy
        policy_weights = np.concatenate((option_policies, option_policies, np.repeat(option_policies, self.num_state_features, axis=0)))
        xp_average = np.average(xp_predictions, axis=1, weights=policy_weights)

        # Constructing the delta vector
        delta_vector = target_immediate + target_discount * xp_average - x_predictions

        # Getting pi(a|s) for all options
        options_pi = option_policies[:, a]
        options_pi_for_trace = np.concatenate((options_pi, options_pi, np.repeat(options_pi, self.num_state_features)))
        
        self.trace.update(target_discount, lmbda, step_size, xa_index, delta_vector, options_pi_for_trace)
        self.approximator.update(step_size, delta_vector * self.trace.z)
        
    def predict(self, x: int, option_index: int) -> Tuple[float, float, npt.ArrayLike]:
        # This is SPECIFIC to the current Option model where it returns you just the action, not the policy.
        option_policy = np.zeros(self.num_actions)
        action, _ = self.options[option_index].step(x)
        option_policy[action] = 1

        x_predictions = self.get_predictions_for_all_actions(x)

        reward = np.average(x_predictions[self.reward_indices][option_index], weights=option_policy)
        discount = np.average(x_predictions[self.discount_indices][option_index], weights=option_policy)
        transition_prob = np.average(x_predictions[self.transition_indices][option_index * self.num_state_features : (option_index + 1) * self.num_state_features], axis=1, weights=option_policy)

        return reward, discount, transition_prob

    def episode_end(self):
        self.trace.episode_end()

        # Processing for logging
        reward = np.zeros((self.num_state_features, self.num_options))
        discount = np.zeros((self.num_state_features, self.num_options))
        transition =  np.zeros((self.num_state_features, self.num_options, self.num_state_features))
        
        for x in range(0, 100):
            for o in range(self.num_options):
                xo_reward, xo_discount, xo_transition_prob = self.predict(x, o)
                reward[x, o] = xo_reward
                discount[x, o] = xo_discount
                transition[x, o] = xo_transition_prob

        # Logging
        # globals.collector.collect('model_r', reward)
        # globals.collector.collect('model_discount', discount)
        # globals.collector.collect('model_transition', transition) 