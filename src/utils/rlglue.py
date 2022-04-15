from RlGlue import BaseAgent
import numpy as np
from typing import Union, Tuple
from src.utils import globals

# keeps a one step memory for TD based agents
class OneStepWrapper(BaseAgent):
    def __init__(self, agent, problem):
        self.agent = agent
        self.gamma = problem.getGamma()
        self.exploration_phase = self.agent.params["exploration_phase"]
        self.no_reward_exploration = self.agent.params.get("no_reward_exploration", False)
        self.num_episodes_passed = 0
        globals.blackboard['num_episodes_passed'] = self.num_episodes_passed
        self.num_steps_passed = 0
        globals.blackboard['num_steps_passed'] = self.num_steps_passed
        self.random = np.random.RandomState(problem.seed)
        self.actions = problem.actions

        self.s = None
        self.a = None
        self._update_exploration_phase()

    def __str__(self):
        return self.agent.__str__()

    def _increment_num_steps(self):
        self.num_steps_passed += 1
        globals.blackboard['num_steps_passed'] = self.num_steps_passed
    
    def _increment_num_episodes(self):
        self.num_episodes_passed += 1
        globals.blackboard['num_episodes_passed'] = self.num_episodes_passed

    def start(self, s):
        self.s = s
        self.a = self.agent.selectAction(self.s)
        self.a = self.random_action_if_exploring(self.a, use_option_action_pair = False)

        return self.a

    # Doesn't modify the action/option if not exploring, otherwise sets it to a random action
    def random_action_if_exploring(self, action_or_option: Union[int, Tuple], use_option_action_pair: bool) -> Union[int, Tuple]:
        if globals.blackboard['in_exploration_phase']:
            action = self.random.choice(self.actions)
            if use_option_action_pair:
                return action, action
            else:
                return action
        
        return action_or_option

    def no_reward_if_exploring(self, reward: float) -> float:
        if globals.blackboard['in_exploration_phase'] and self.no_reward_exploration:
            return 0

        return reward

    def _update_exploration_phase(self):
        if self.num_steps_passed < self.exploration_phase:
            globals.blackboard['in_exploration_phase'] = True
        else:
            globals.blackboard['in_exploration_phase'] = False

    def step(self, r, sp, info=None, t=False):
        r = self.no_reward_if_exploring(r)
        ap = self.agent.update(self.s, self.a, sp, r, self.gamma, info=info)

        ap = self.random_action_if_exploring(ap, use_option_action_pair = False)

        self.s = sp
        self.a = ap

        self._increment_num_steps()
        return ap
        
    def end(self, r, info=None):
        r = self.no_reward_if_exploring(r)
        gamma = 0
        self.agent.agent_end(self.s, self.a, r, gamma, info=info)  

        self._increment_num_steps()
        self._increment_num_episodes()
        self._update_exploration_phase()

class OptionOneStepWrapper(OneStepWrapper):
    def start(self, s):
        self.s = s
        self.o, self.a = self.agent.selectAction(self.s)
        self.a, self.o = self.random_action_if_exploring((self.a, self.o), use_option_action_pair = True)

        return self.a

    def step(self, r, sp, info=None, t=False):
        r = self.no_reward_if_exploring(r)
        op, ap = self.agent.update(self.s, self.o, self.a, sp, r, self.gamma, info=info)

        op, ap = self.random_action_if_exploring((op, ap), use_option_action_pair = True)

        self.s = sp
        self.o = op
        self.a = ap

        self._increment_num_steps()
        return ap
    def end(self, r, info=None):
        r = self.no_reward_if_exploring(r)
        gamma = 0
        self.agent.agent_end(self.s, self.o, self.a, r, gamma,  info=info)
        
        self._increment_num_steps()
        self._increment_num_episodes()
        self._update_exploration_phase()

class OptionFullExecuteWrapper(OneStepWrapper):
    def start(self, s):
        self.s = s
        self.o = self.agent.selectAction(self.s)

        action = self._get_action(self.s, self.o)
        return action

    def _get_action(self, s, o) -> int:
        if (self.agent.is_option(o)):
            action, t = self.agent.get_action(s, o)
            return action
        return o

    def step(self, r, sp, t=False):
        r = self.no_reward_if_exploring(r)

        xp = self.rep.encode(sp)

        # Fully follow option phase
        if (self.agent.is_option(self.o)):
            # execute the option and don't update the agent
            action, t = self.agent.get_action(xp, self.o)
            if t == False:
                # Option has not terminated yet, keep giving the action
                return action
                
        op = self.agent.update(self.s, self.o, xp, r, self.gamma)
        op = self.random_action_if_exploring(op, use_option_action_pair = False)

        self.s = sp
        self.o = op

        # [2021-09-03 clo] Whenever _get_action is called, the action chosen is always taken
        # This is perhaps not good if the option can start at one of its termination states,
        # because that would mean that the beginning action will always execute.
        # However, dealing with this will likely involve dealing with an infinite loop of selecting actions,
        # which we avoid for now. The same issue occurs in the start function.
        action = self._get_action(self.s, self.o)

        self._increment_num_steps()
        return action

    def end(self, r):
        r = self.no_reward_if_exploring(r)
        
        self.num_episodes_passed += 1
        gamma = 0  
        self.agent.agent_end(self.s, self.o, r, gamma)

        self._increment_num_steps()
        self._increment_num_episodes()
        self._update_exploration_phase()
