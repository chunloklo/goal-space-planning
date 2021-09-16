from RlGlue import BaseAgent
import numpy as np

# keeps a one step memory for TD based agents
class OneStepWrapper(BaseAgent):
    def __init__(self, agent, problem):
        self.agent = agent
        self.gamma = problem.getGamma()
        self.rep = problem.rep
        self.exploration_phase = self.agent.params["exploration_phase"]
        self.num_episodes_passed = 0
        self.random = np.random.RandomState(problem.seed)
        self.actions = problem.actions

        self.s = None
        self.a = None
        self.x = None

        self.options = self.agent.options

    def __str__(self):
        return self.agent.__str__()

    def start(self, s):
        self.s = s
        self.x = self.rep.encode(s)
        self.a = self.agent.selectAction(self.x)

        return self.a

    def step(self, r, sp, t=False):
        xp = self.rep.encode(sp)

        ap = self.agent.update(self.x, self.a, xp, r, self.gamma)
        
        if (self.num_episodes_passed < self.exploration_phase):
            ap = self.random.choice(self.actions)

        self.s = sp
        self.a = ap
        self.x = xp

        return ap

    def state_encoding(self, s):
        return self.rep.encode(s)

    def end(self, r):
        self.num_episodes_passed += 1
        gamma = 0
        self.agent.agent_end(self.x, self.a, r, gamma)  
        # reset agent here if necessary (e.g. to clear traces)

class OptionOneStepWrapper(OneStepWrapper):
    def start(self, s):
        self.s = s
        self.x = self.rep.encode(s)
        self.o, self.a = self.agent.selectAction(self.x)
        return self.a

    def step(self, r, sp, t=False):
        xp = self.rep.encode(sp)

        op, ap = self.agent.update(self.x, self.o, self.a, xp, r, self.gamma)

        if (self.num_episodes_passed < self.exploration_phase):
            ap = self.random.choice(self.actions)
            op = ap

        self.s = sp
        self.o = op
        self.a = ap
        self.x = xp

        return ap
    def end(self, r):
        self.num_episodes_passed += 1
        gamma = 0
        self.agent.agent_end(self.x, self.o, self.a, r, gamma)

class OptionFullExecuteWrapper(OneStepWrapper):
    def start(self, s):
        self.s = s
        self.x = self.rep.encode(s)
        self.o = self.agent.selectAction(self.x)

        action = self._get_action(self.x, self.o)
        return action

    def _get_action(self, x, o) -> int:
        if (self.agent.is_option(o)):
            action, t = self.agent.get_action(x, o)
            return action
        return o

    def step(self, r, sp, t=False):
        xp = self.rep.encode(sp)

        # Exploration phase
        if (self.num_episodes_passed < self.exploration_phase):
            ap = self.random.choice(self.actions)
            op = ap

        else:
            # Fully follow option phase
            if (self.agent.is_option(self.o)):
                # execute the option and don't update the agent
                action, t = self.agent.get_action(xp, self.o)
                if t == False:
                    # Option has not terminated yet, keep giving the action
                    return action
                
        op = self.agent.update(self.x, self.o, xp, r, self.gamma)

        self.s = sp
        self.o = op
        self.x = xp

        # [2021-09-03 clo] Whenever _get_action is called, the action chosen is always taken
        # This is perhaps not good if the option can start at one of its termination states,
        # because that would mean that the beginning action will always execute.
        # However, dealing with this will likely involve dealing with an infinite loop of selecting actions,
        # which we avoid for now. The same issue occurs in the start function.
        action = self._get_action(self.x, self.o)
        return action

    def end(self, r):
        self.num_episodes_passed += 1
        gamma = 0  
        self.agent.agent_end(self.x, self.o, r, gamma)
