from typing import Dict, Optional, Union, Tuple
import numpy as np
from numpy import isin
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from src.agents.components.learners import ESarsaLambda, QLearner
from src.agents.components.search_control import ActionModelSearchControl_Tabular
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_Sutton_Tabular, CombinedModel_ESarsa_Tabular
from src.agents.components.approximators import DictModel
from typing import Dict, Union, Tuple, Any, TYPE_CHECKING
from src.utils.log_utils import run_if_should_log

from src.utils.numpy_utils import create_onehot
# from src.environments.GrazingWorldAdam import get_pretrained_option_model, state_index_to_coord, get_all_transitions
from external.dreamerv2.agent import Agent as DreamerAgent
from gym import spaces
from external.dreamerv2 import common
import ruamel.yaml as yaml
import sys
import pathlib
import external
import time
import jax.numpy as jnp

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class Dreamer:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper
        self.env = problem.getEnvironment()
        self.num_actions = problem.actions
        self.params = problem.params
        self.random = np.random.RandomState(problem.seed)
        self.goals = problem.goals
        self.num_goals = len(self.goals)
        self.goal_termination_func = problem.goal_termination_func
        self.goal_initiation_func = problem.goal_initiation_func
        
        param = self.params
        self.dreamer_config_path = param['dreamer_config_path']

        configs = yaml.safe_load(
            (pathlib.Path(self.dreamer_config_path)).read_text())

        # parsed, remaining = common.Flags(configs=['defaults']).parse(
            # known_only=True)
        self.config = common.Config(configs['defaults'])

        for name in ['pinball']:
            self.config = self.config.update(configs[name])

        # config = common.Flags(config).parse(remaining)

        self.step = common.Counter(0)
        act_space = {'action': spaces.Box(0, 1, (5, ), dtype=np.bool)}

        # @property
        # def act_space(self):
        #     shape = (self._env.act_space[self._key].n, )
        #     space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        #     space.sample = self._sample_action
        #     space.n = shape[0]
        #     return {**self._env.act_space, self._key: space}

        # def step(self, action):
        #     index = np.argmax(action[self._key]).astype(int)
        #     reference = np.zeros_like(action[self._key])
        #     reference[index] = 1
        #     if not np.allclose(reference, action[self._key]):
        #         raise ValueError(f'Invalid one-hot action:\n{action}')
        #     return self._env.step({**action, self._key: index})

        # def reset(self):
        #     return self._env.reset()

        # def _sample_action(self):
        #     actions = self._env.act_space.n
        #     index = self._random.randint(0, actions)
        #     reference = np.zeros(actions, dtype=np.float32)
        #     reference[index] = 1.0
        #     return reference



        # act_space = = {'action': spaces.Discrete(5)}
        obs_space = {
            'image': spaces.Box(-1, 1, (4,)),
            'reward': spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            # 'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            # 'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': spaces.Box(0, 1, (), dtype=np.bool),
        }

        config = self.config

        self.dreamer_agent = DreamerAgent(config, obs_space, act_space, self.step)

        self.should_train = common.Every(config.train_every)
        self.should_log = common.Every(config.log_every)
        self.should_video_train = common.Every(config.eval_every)
        self.should_video_eval = common.Every(config.eval_every)
        self.should_expl = common.Until(config.expl_until)

        if config.replay.prioritized:
            buffer_cls = common.PrioritizedJaxSubsequenceParallelEnvReplayBuffer
        else:
            buffer_cls = common.JaxSubsequenceParallelEnvReplayBuffer
            
        # Replay buffers

        # no_framestack_shape = list(train_envs.obs_space["image"].shape)
        no_framestack_shape = list((5,))
        no_framestack_shape[-1] = no_framestack_shape[-1] // config.framestack
        self.train_replay = buffer_cls(
            stack_size=config.framestack,
            n_envs=config.envs,
            batch_size=config.dataset.batch,
            subseq_len=config.dataset.length,
            persistent=config.replay.persistent,
            seed=config.seed,
            observation_shape=tuple(no_framestack_shape),
            replay_capacity=config.replay.capacity,
            action_shape=act_space["action"].shape,
            action_dtype=act_space["action"].dtype,
            extra_storage_types=[common.ReplayElement("is_first", (), np.int32),
                                common.ReplayElement("is_last", (), np.int32)]
        )

        self.eval_replay = buffer_cls(
            stack_size=config.framestack,
            n_envs=1,
            batch_size=config.dataset.batch,
            subseq_len=config.dataset.length,
            persistent=config.replay.persistent,
            seed=config.seed,
            observation_shape=obs_space["image"].shape,
            replay_capacity=config.replay.capacity//100,
            action_shape=act_space["action"].shape,
            action_dtype=act_space["action"].dtype,
            extra_storage_types=[common.ReplayElement("is_first", (), np.int32),
                                common.ReplayElement("is_last", (), np.int32)]
        )

        self._state = self.dreamer_agent.init_policy_state(config.envs)

        self.first_episode_update = True
        self.num_episodes = 0
        self.cumulative_reward = 0

    def FA(self):
        return "NN"

    def __str__(self):
        return "Dreamer"

    # Public method for rlglue
    def selectAction(self, s: int) -> Tuple[int, int] :
        a = self.random.choice(self.num_actions)
        # Random for now
        return a
    
    def _train_policy(self, sample_dict, state, reset):
        return self.dreamer_agent.policy(obs=jnp.expand_dims(sample_dict['image'],1), is_first=sample_dict['is_first'], state=state, reward=sample_dict['reward'],
                                            mode='explore'
                                            if self.should_expl(self.step) else 'train', reset=reset)
                                            
    def _postprocess_sample(sample_dict):
        # Need to transpose last 2 channels to make sure colors are continuous
        image = sample_dict["state"].transpose(0,1,2,3,5,4)
        sample_dict["image"] = image.reshape(*image.shape[:-2], -1)
        sample_dict['same_trajectory'] = sample_dict['same_trajectory'].astype(np.int32)
        # sample_dict = {k: train_driver._convert(jnp.array(v)) for k, v in sample_dict.items()}
        sample_dict = {k: jnp.array(v) for k, v in sample_dict.items()}
        sample_dict["is_terminal"] = sample_dict["terminal"]
        return sample_dict['image'], sample_dict['action'], sample_dict['reward'], sample_dict['is_first'], sample_dict["terminal"]

    def _train_step(self):
        x = self.should_train(self.step)
        train_agent = common.CarryOverState(self.dreamer_agent.train, init_state=self.dreamer_agent.init_policy_state(self.config.dataset.batch)[0])
        for _ in range(x):
            for _ in range(self.config.train_steps):
                start = time.time()
                sample = self.train_replay.sample()[1]
                sample_time = time.time() - start
                obs, action, reward, is_first, terminal = self._postprocess_sample(sample)
                
                start = time.time()
                metrics, rec_obs = train_agent(obs, action, reward, is_first, terminal)
                metrics["train_time"] = time.time() - start
                metrics["sample_time"] = sample_time

    def update(self, s, a, sp, r, gamma, terminal: bool = False):
        s = np.array([s])
        sp = np.array([sp])
        self.step.increment(self.config.envs)
        
        if self.num_episodes > 0:
            self._train_step()
            

        obs = {}
        obs['image'] = sp
        obs['reward'] = r
        obs['is_terminal'] = terminal
        obs['is_first'] = np.array([1 if self.first_episode_update else 0])
        obs['is_last'] = terminal # Hmm this doesn't feel quite right. But leaving this here for now.

        print(f'############### FIRST EP{self.first_episode_update}')

        ap, self._state = self._train_policy(obs, self._state, reset = self.first_episode_update)

        # if not terminal:
        #     ap = self.selectAction(sp)
        # else:
        #     ap = None
        # print(ap)
        # obs['action'] = create_onehot(self.num_actions, ap)
        obs['action'] = ap

        
        self.train_replay.add(
            obs["image"], 
            obs["action"],
            obs["reward"], 
            obs["is_terminal"],
            obs["is_first"],
            obs["is_last"],
            episode_end=obs["is_last"])
    
        self.cumulative_reward += r
        def log(): 
            # globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0
        run_if_should_log(log)

        self.first_episode_update = False

        return ap

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, s, r, gamma, terminal=True)
        self.first_episode_update = True
        self.num_episodes += 1