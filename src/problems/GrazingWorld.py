from src.problems.BaseProblem import BaseProblem
from src.environments.GrazingWorld import GrazingWorld as GWEnv
from src.environments.GrazingWorldSimple import GrazingWorldSimple as GWSimpleEnv
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
from src.utils.options import load_option

class GrazingWorld(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = GWEnv(self.seed, reward_sequence_length=self.params['reward_sequence_length'], initial_learning=self.params['exploration_phase'])
        self.actions = 4
        self.options = [load_option('GrazingO1'), load_option('GrazingO2'),load_option('GrazingO3')]
        self.rep = Tabular(self.env.shape, self.actions)

        self.features = self.rep.features()
        self.gamma = self.params['gamma']

class GrazingWorldWithMiddleOption(GrazingWorld):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.options = [load_option('GrazingO1'), load_option('GrazingO2'),load_option('GrazingO3'), load_option('GrazingO4')]

class GrazingWorldSimpleProblem(GrazingWorld):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.options = [load_option('IdealGrazingO1'), load_option('IdealGrazingO2'),load_option('IdealGrazingO3')]
        self.env = GWSimpleEnv(self.seed, reward_sequence_length=self.params['reward_sequence_length'])
        # Representation space is the same as normal GWEnv. No need to change it.

class GrazingWorldSimpleProblemDirectOptions(GrazingWorld):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.options = [load_option('GrazingO1'), load_option('GrazingO2'),load_option('GrazingO3'), load_option('GrazingO4')]
        self.env = GWSimpleEnv(self.seed, reward_sequence_length=self.params['reward_sequence_length'])
        # Representation space is the same as normal GWEnv. No need to change it.