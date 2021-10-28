from typing import Any, Callable, List, Dict, Tuple, Literal
import numpy as np
import numpy.typing as npt
from utils import numpy_utils, param_utils

# Right now for action values only. Need additional params and implementation if we want something different
class Trace():
    def __init__(self, num_state_features: int, output_features: int, feature_type: Literal['tabular', 'vector']):
        # Num weights X Num output features
        self.z: npt.ArrayLike = np.zeros((num_state_features, output_features))
        self.feature_type:str = param_utils.check_valid(feature_type, lambda x: x in ['tabular', 'vector'])

    def update(self, gamma: npt.ArrayLike, lmbda: float, step_size: float, grad: Any, delta: Any, rho: Any):
        
        # For action values
        if self.feature_type == 'vector':
            self.z = rho * lmbda * gamma * self.z + grad
        elif self.feature_type == 'tabular':
            self.z = lmbda * rho * gamma * self.z
            self.z[grad] += 1
            
    def episode_end(self):
        self.z[:] = 0
