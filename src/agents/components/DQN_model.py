import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy.typing as npt
from typing import Any
import numpy as np
from src.utils.log_utils import run_if_should_log
import copy
from src.utils import globals

class GoalLearner_DQN_NN():
    def __init__(self, state_shape, num_actions: int, step_size: float, epsilon: float, polyak_stepsize, adam_eps, arch_flag: str):
        self.num_actions: int = num_actions #
        self.epsilon = epsilon
        self.adam_eps = adam_eps

        assert arch_flag in [
            'pinball_simple',
            'pinball_hard',
        ]

        # Initializing jax functions
        init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
        b_init = hk.initializers.Constant(0.001)
        
        if arch_flag == 'pinball_simple':
            # Initializing jax functions
            def q_function(states):
                q_mlp = hk.Sequential([
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(64), jax.nn.relu,
                    hk.Linear(64), jax.nn.relu,
                    hk.Linear(self.num_actions),
                ])

                v_mlp = hk.Sequential([
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(64), jax.nn.relu,
                    hk.Linear(64), jax.nn.relu,
                    hk.Linear(self.num_actions),
                ])

                return q_mlp(states), v_mlp(states)
        elif arch_flag == 'pinball_hard':
            def q_function(states):
                q_mlp = hk.Sequential([
                    hk.Linear(256, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(256, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(32, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(32, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(self.num_actions),
                ])
                v_mlp = hk.Sequential([
                    hk.Linear(256, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(256, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(32, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(32, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(self.num_actions),
                ])
                return q_mlp(states), v_mlp(states) 
                
        else:
            raise NotImplementedError()

        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))

        self.f_opt = optax.adam(step_size, eps=adam_eps)

        def _take_action_index(data, action):
            return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()
    
        def _loss(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            #o = data['o']
            a = data['a']
            xp = data['xp']
            # gamma = data['gamma']
            goal_policy_cumulant = data['goal_policy_cumulant']
            goal_discount = data['goal_discount']

            policy_pred, reward_pred, gamma_pred = self.f_get_goal_output(params, x)
            # Getting values for specific actions
            policy_pred = _take_action_index(policy_pred, a)
            reward_pred = _take_action_index(reward_pred, a)
            gamma_pred = _take_action_index(gamma_pred, a)
            
            xp_policy_pred, xp_reward_pred, xp_gamma_pred = jax.lax.stop_gradient(self.f_get_goal_output(target_params, xp))
            xp_ap = jnp.argmax(xp_policy_pred, axis=1)

            # ESarsa version rather than Q version
            # policy = jax.nn.one_hot(xp_ap, self.num_actions) * (1 - self.epsilon) + (self.epsilon / self.num_actions)
            # policy_target = goal_policy_cumulant + goal_discount * jnp.average(xp_policy_pred, axis=1, weights=policy)
            # reward_target = r + goal_discount * jnp.average(xp_reward_pred, axis=1, weights=policy)

            policy_target = goal_policy_cumulant + goal_discount * _take_action_index(xp_policy_pred, xp_ap)
            reward_target = r + goal_discount * _take_action_index(xp_reward_pred, xp_ap)

            policy_loss = jnp.mean(jnp.square(policy_target - policy_pred))
            reward_loss = jnp.mean(jnp.square(reward_target - reward_pred))

            # return policy_loss, (jax.lax.stop_gradient(policy_loss), jax.lax.stop_gradient(reward_loss))

            return policy_loss + reward_loss, (jax.lax.stop_gradient(policy_loss), jax.lax.stop_gradient(reward_loss))

        def _get_goal_output(params: hk.Params, x: Any):
            policy_output, v_output = self.f_qfunc.apply(params, x)
            # Since the problem is shortest path and we're trying to maximize the reward anyways, the signals are essentially identical.
            gamma_output = policy_output
            return policy_output, v_output, gamma_output

        def _update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, reward_loss = jax.grad(_loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, reward_loss
            
        self.f_get_goal_output = jax.jit(_get_goal_output)
        self.f_update = jax.jit(_update)
        # self.f_update = _update

        # Initialization
        dummy_state = jnp.zeros(state_shape)
        self.params = self.f_qfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.f_opt.init(self.params)

        # target params for the network
        self.target_params = copy.deepcopy(self.params)

    def get_goal_outputs(self, x: npt.ArrayLike) -> np.ndarray:
        policy_output, v_output, gamma_output = self.f_get_goal_output(self.params, x)
        return policy_output, v_output, gamma_output

    def update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, (policy_loss, reward_loss) = self.f_update(self.params, self.target_params, self.opt_state, data)
        
        def log():
            globals.collector.collect('reward_loss', reward_loss)
            globals.collector.collect('policy_loss', policy_loss)
        run_if_should_log(log)

        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        # return self.params