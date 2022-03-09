from src.problems.BaseProblem import BaseProblem
from src.environments.pinball import PinballEnvironment
# from src.problems.BaseProblem import BaseProblem
# from src.environments.GrazingWorld import GrazingWorld as GWEnv
# from src.environments.GrazingWorldSimple import GrazingWorldSimple as GWSimpleEnv
# from src.environments.GrazingWorldAdam import GrazingWorldAdam as GWEnvAdam, get_pretrained_option_model, get_all_transitions
# from src.environments.GrazingWorldAdam import GrazingWorldAdamImageFeature
# from PyFixedReps.TileCoder import TileCoder
# from PyFixedReps.Tabular import Tabular
# from src.utils.options import load_option
# from src.utils.create_options import get_options
# from src.utils import globals, param_utils

class PinballProblem(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = PinballEnvironment(self.params['pinball_configuration_file'])
        self.actions = 5
        # self.options = get_options("Grazing")[0:3]
        self.gamma = self.params['gamma']