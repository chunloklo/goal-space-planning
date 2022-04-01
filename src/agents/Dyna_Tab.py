from typing import Dict, List, Optional, Union, Tuple
from matplotlib import type1font
import numpy as np
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from src.agents.components.learners import ESarsaLambda, QLearner
from src.agents.components.search_control import ActionModelSearchControl_Tabular
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import ActionModel_Linear, OptionActionModel_ESARSA, OptionModel_Sutton_Tabular, CombinedModel_ESarsa_Tabular, OptionActionModel_Sutton_Tabular
from src.agents.components.approximators import DictModel
from typing import Dict, Union, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class Dyna_Tab:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper

        self.env = problem.getEnvironment()
        self.representation: Tabular = problem.get_representation('Tabular')
        self.features = self.representation.features
        self.num_actions = problem.actions
        self.actions = list(range(self.num_actions))
        self.params = problem.params
        self.num_states = self.env.nS
        self.random = np.random.RandomState(problem.seed)
        self.options = problem.options
        self.num_options = len(problem.options)

        # This is only needed for RL glue (for convenience reason to have uniform initialization between
        # options and non-options agents). Not used in the algorithm at all.
        params = globals.param

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 
        self.search_control = ActionModelSearchControl_Tabular(self.random)
        
        self.option_alg = param_utils.parse_param(params, 'option_alg', lambda p : p in ['None', 'DecisionTime', 'Background', 'Background_MaxAction'])
        self.learn_model =  param_utils.parse_param(params, 'learn_model', lambda p : isinstance(p, bool))
        self.q_init = param_utils.parse_param(params, 'q_init', lambda p: True, default=0, optional=True)
        self.use_goal_reward = param_utils.parse_param(params, 'use_goal_reward', lambda p: isinstance(p, bool), default=False, optional=True)
 
        # Loading the pretrained model rather than learning from scratch.
        if not self.learn_model:
            self.option_model, self.option_action_model = problem.get_pretrained_option_model()
        else:
            self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
            # self.option_action_model = OptionActionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
            self.option_action_model = OptionActionModel_ESARSA(self.num_states + 1, self.num_actions, self.num_options, self.options)

        # Creating models for actions and options
        self.history_dict = DictModel()

        self.action_model = ActionModel_Linear(self.num_states + 1, self.num_actions)
        
        if not self.learn_model:
            transitions = problem.get_all_transitions()

            # Prefilling dict buffer so no exploration is needed
            for t in transitions:
                # t: s, a, sp, reward. gamma
                self.history_dict.update(self.representation.encode(t[0]), t[1], self.representation.encode(t[2]), t[3], t[4])
                # do a complete update with each transition
                self.action_model.update(self.representation.encode(t[0]), t[1], self.representation.encode(t[2]), t[3], t[4], 1)
        
        self.goals = problem.get_goals()
        # Commenting this line out for now for the HMaze. We need this line for the GrazingWorld
        # self.goals = [self.representation.encode(goal) for goal in self.goals]
        self.tau = np.zeros(len(self.goals))

        # Learning the goal rewards directly
        self.goal_reward_learner = GoalRewardLearner(self.goals, self.alpha)

        # +1 accounts for the terminal state
        if self.behaviour_alg == 'QLearner':
            self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
            self.behaviour_learner.Q[:] = self.q_init
        elif self.behaviour_alg == 'ESarsaLambda':
            self.lmbda = params['lambda']
            self.behaviour_learner = ESarsaLambda(self.num_states + 1, self.num_actions)
        else:
            raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # Logging
        self.cumulative_reward = 0

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Dyna_Tab"

    def get_policy(self, x: int) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)

        if self.option_alg in ['DecisionTime']:
            option_values = self._get_option_action_values(x)

            best_action_option_value = np.max(option_values, axis=0)
            best_option_value = np.max(best_action_option_value)
            best_option_action = np.argmax(best_action_option_value)

            action_values = self.behaviour_learner.get_action_values(x)
            best_action = np.argmax(action_values)
            best_action_value = np.max(action_values)

            if best_option_value >= best_action_value:
                a = best_option_action
            else:
                a = best_action

        else:    
            a = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[a] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, s: int) -> Tuple[int, int] :
        x = self.representation.encode(s)
        a = self.random.choice(self.num_actions, p = self.get_policy(x))
        return a

    def update(self, s, a, sp, r, gamma, terminal: bool = False):

        # Special modification for testing learning speed on goal2_switch
        if globals.param['reward_schedule'] == 'goal2_switch':
            if globals.blackboard['num_steps_passed'] == self.params['reward_sequence_length']:
                # Correcting the one-step model based on 
                _x = 31
                _xp = self.num_states
                _r = 0.5
                _gamma = 0
                for _a in range(self.num_actions):
                    self.update_model(_x, _a, _xp, _r, _gamma)
                    # Always using an update of 1 to perfectly update the action values
                    self.behaviour_learner.update(_x, _a, _xp, _r, _gamma, 1)
                    
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states

        # Exploration bonus tracking
        if not globals.blackboard['in_exploration_phase']:
            self.tau += 1
        self.tau[xp == np.array(self.goals)] = 0

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.update(x, a, xp, r, self.get_policy(xp), self.gamma, self.lmbda, self.alpha)
        else:
            raise NotImplementedError()

        # Updating search control. Order is important.
        self.search_control.update(x, xp)

        self.update_model(x, a, xp, r, gamma)  
        self.planning_step(x, xp)

        if not terminal:
            ap = self.selectAction(sp)
            # print(f'sp: {sp} {self.behaviour_learner.get_action_values(xp)} ap: {ap} exploring: {globals.blackboard["in_exploration_phase"]}')
        else:
            ap = None

        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0

        return ap
    def update_model(self, x, a, xp, r, gamma):
        """updates the model 
        
        Returns:
            Nothing
        """
        self.history_dict.update(x, a, xp, r, gamma)
        self.action_model.update(x, a, xp, r, gamma, self.alpha)
        self.option_model.update(x, a, xp, r, gamma, self.alpha)
        self.option_action_model.update(x, a, xp, r, gamma, self.alpha)
        self.goal_reward_learner.update(x, a, xp, r, gamma)

    def _planning_update(self, x: int, a: int):
        r, discount, transition_prob = self.action_model.predict(x, a)
        xp = np.argmax(transition_prob)

        if globals.param['reward_schedule'] == 'goal2_switch':
            pass
        else:
            # Exploration bonus for +
            if xp in self.goals:
                r += self.kappa * np.sqrt(np.sum(self.tau[xp == np.array(self.goals)]))

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.planning_update(x, a, xp, r, discount, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.planning_update(x, a, xp, r, self.get_policy(xp), discount, self.alpha)
        else:
            raise NotImplementedError()

    def _sample_option(self, x, o, a: Optional[int] = None):
        if a == None:
            r, discount, transition_prob = self.option_model.predict(x, o)
        else:
            r, discount, transition_prob = self.option_action_model.predict(x, a, o)
        

        # norm = np.linalg.norm(transition_prob, ord=1)
        # prob = transition_prob / norm
        # # +1 here accounts for the terminal state
        # xp = self.random.choice(self.num_states + 1, p=prob)

        xp = np.argmax(transition_prob)
    
        return r, discount, xp

    def _get_option_action_values(self, x):
        option_values = np.zeros((self.num_options, self.num_actions))

        if x in self.goals and self.use_goal_reward:
            for a in range(self.num_actions):
                option_values[:, a] = np.max(self.behaviour_learner.get_action_values(x))
            return option_values

        for o in range(self.num_options):
            for a in range(self.num_actions):
                
                r, discount, xp = self._sample_option(x, o, a)
                if self.use_goal_reward:
                    option_value = r + discount * self.goal_reward_learner.get_goal_reward(o)
                else:
                    option_value = r + discount * np.max(self.behaviour_learner.get_action_values(xp))

                option_values[o, a] = option_value

                if globals.param['reward_schedule'] == 'goal2_switch':
                    pass
                else:
                    if self.use_goal_reward:
                        option_values[o, a] += self.kappa * np.sqrt(np.sum(self.tau[o]))
                    else:
                        option_values[o, a] += self.kappa * np.sqrt(np.sum(self.tau[xp == np.array(self.goals)]))
                        
        return option_values

    def _option_constrainted_improvement(self, x: int):
        option_values = self._get_option_action_values(x)
        max_option_values = np.max(option_values, axis=0)

        # Updating towards option values
        max_values = np.maximum(max_option_values, self.behaviour_learner.get_action_values(x))
        # [2022-01-11 chunlok] This update is currently a hacky way of doing it that only works with the QLearner I believe.

        if self.option_alg == 'Background_MaxAction':
            self.behaviour_learner.Q[x, :] += self.alpha * (max_values - self.behaviour_learner.Q[x, :])
        else:
            self.behaviour_learner.Q[x, :] += self.alpha * (max_option_values - self.behaviour_learner.Q[x, :])

    def planning_step(self, x:int, xp: int):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        if self.option_alg == 'None':
            sample_states = self.search_control.sample_states(self.planning_steps, self.history_dict, x, xp)
            
            for plan_x in sample_states:
                visited_actions = self.history_dict.visited_actions(plan_x)
                a = self.random.choice(visited_actions)
                self._planning_update(plan_x, a)

        if self.option_alg in ['Background', 'Background_MaxAction']:
            # Option planning update
            sample_states = self.search_control.sample_states(self.planning_steps, self.history_dict, x, xp)
            for plan_x in sample_states:
                self._option_constrainted_improvement(plan_x)


    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

        self.option_model.episode_end()
        self.option_action_model.episode_end()


class GoalRewardLearner():
    def __init__(self, goals: List, step_size: float):
        self.goals = goals
        self.goal_rewards = np.zeros(len(goals))
        self.step_size = step_size
    
    def update(self, x, a, xp, r, gamma):
        if x in self.goals:
            index = self.goals.index(x)
            self.goal_rewards[index] += self.step_size * (r - self.goal_rewards[index])
    
    def get_goal_reward(self, o):
        return self.goal_rewards[o]