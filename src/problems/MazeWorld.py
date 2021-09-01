from src.problems.BaseProblem import BaseProblem
from src.environments.MazeWorld import MazeWorld as MWEnv
from PyFixedReps.TileCoder import TileCoder
from PyFixedReps.Tabular import Tabular
#from src.utils.options import load_option

class MazeWorld(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = MWEnv(self.seed)
        self.actions = 4
        self.options = [load_option('MazeO1'), load_option('MazeO2'),load_option('MazeO3')]

        if 'Tab' in exp.agent:
            self.rep = Tabular(self.env.shape, self.actions)
        else:
            self.rep = TileCoder({
                'dims': 2,
                'tiles': 4,
                'tilings': 16,
                'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)],
                'scale_output': True,
            })            

        self.features = self.rep.features()
        self.gamma = 1.0
