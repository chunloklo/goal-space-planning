from typing import Any, Callable, List, Dict, Tuple, Literal
import numpy as np
import numpy.typing as npt
from src.utils import numpy_utils, param_utils, globals
from src.utils.run_utils import InvalidRunException

# Right now for action values only. Need additional params and implementation if we want something different
class Trace():
    def __init__(self, num_state_features: int, output_features: int, feature_type: Literal['tabular', 'vector']):
        # Num weights X Num output features
        self.z: npt.ArrayLike = np.zeros((num_state_features, output_features))
        self.feature_type:str = param_utils.check_valid(feature_type, lambda x: x in ['tabular', 'vector'])

    def update(self, gamma: npt.ArrayLike, lmbda: float, grad: Any, rho: Any):
        
        # For action values
        if self.feature_type == 'vector':
            # accumulating trace
            self.z = rho * lmbda * gamma * self.z + grad
        elif self.feature_type == 'tabular':
            # replacing trace
            self.z = lmbda * rho * gamma * self.z
            self.z[grad] = 1
        
        if (np.isnan(np.sum(self.z))):
            raise InvalidRunException("Nan encountered in Trace")
            
    def episode_end(self):
        self.z[:] = 0
