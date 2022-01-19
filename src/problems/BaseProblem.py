from src.experiment.ExperimentModel import ExperimentModel
from src.agents.registry import getAgent
from typing import Any

class BaseProblem:
    def __init__(self, exp: ExperimentModel, idx: int, seed: int):
        self.exp = exp
        self.idx = idx

        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']

        self.agent = None
        self.env = None
        self.gamma = self.params

        self.seed = seed
        self.actions = None
        self.options = None

    def getEnvironment(self):
        return self.env

    def getGamma(self):
        return self.gamma

    def getAgent(self):
        if self.agent is None:
            Agent = getAgent(self.exp.agent)
            self.agent = Agent(self)
        return self.agent
    
    def get_representation(self, rep_type: Any):
        raise NotImplementedError(f'get_representation is not implemented for {type(self).__name__}')
