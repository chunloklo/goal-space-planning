from ..environments.PinballGoals import PinballGoals, PinballOracleGoals
from src.problems.BaseProblem import BaseProblem
from src.environments.pinball import PinballEnvironment
import numpy as np

class PinballProblem(BaseProblem):
    """Pinball Problem. This problem is formulated as naturally continuing

    Args:
        BaseProblem (_type_): _description_
    """
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self._init_with_goals(PinballGoals())

    def _init_with_goals(self, goals: PinballGoals):
        self.goals = goals
        # [chunlok 2022-04-15] TODO There's a lot of duplicated code here for essentially just replacing the goals. Perhaps there's a better way to do this?
        self.env = PinballEnvironment(self.params['pinball_configuration_file'], self.goals, self.params['render'], self.params['explore_env'], continuing=True)
        self.actions = 5
        self.gamma = self.params['gamma']

class PinballTermProblem(PinballProblem):
    # THIS IS SPECIFICALLY FOR DREAMER. THIS ENV TERMINATES
    def __init__(self, exp, idx, seed: int):
        raise NotImplementedError('This code has not been tested yet. So proceed with caution.')
        super().__init__(exp, idx, seed)
        self.goals = PinballGoals()
        self.env = PinballEnvironment(self.goals, self.params['pinball_configuration_file'], self.params['render'], self.params['explore_env'], continuing=True)
        self.actions = 5
        self.gamma = self.params['gamma']

class PinballOracleProblem(PinballProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self._init_with_goals(PinballOracleGoals())