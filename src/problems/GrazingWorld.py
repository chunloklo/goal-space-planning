from src.problems.BaseProblem import BaseProblem
from src.environments.GrazingWorld import GrazingWorld as GWEnv
from src.environments.GrazingWorldSimple import GrazingWorldSimple as GWSimpleEnv
from src.environments.GrazingWorldAdam import GrazingWorldAdam as GWEnvAdam
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option
from src.utils.create_options import get_options
from src.utils import globals

class GrazingWorld(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = GWEnv(self.seed, reward_sequence_length=self.params['reward_sequence_length'], initial_learning=self.params['exploration_phase'])
        self.actions = 4
        self.options = get_options("Grazing")[0:3]
        self.rep = Tabular(self.env.shape, self.actions)

        self.features = self.rep.features()
        self.gamma = self.params['gamma']

        globals.blackboard['terminal_state'] = 100

class GrazingWorldWithMiddleOption(GrazingWorld):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.options = get_options("Grazing")[0:4]

class GrazingWorldSimple(GrazingWorld):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.options = get_options("Grazing")
        self.env = GWSimpleEnv(self.seed, reward_sequence_length=self.params['reward_sequence_length'])
        # Representation space is the same as normal GWEnv. No need to change it.

class GrazingWorldAdam(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = GWEnvAdam(self.seed, reward_sequence_length=self.params['reward_sequence_length'], initial_learning=self.params['exploration_phase'])
        self.actions = 4
        self.options = get_options("GrazingAdam")
        self.rep = Tabular(self.env.shape, self.actions)
        self.features = self.rep.features()
        self.gamma = self.params['gamma']

        globals.blackboard['terminal_state'] = 96