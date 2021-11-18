from typing import Any, List, Dict, Tuple
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from src.utils.run_utils import InvalidRunException
from src.utils.numpy_utils import create_onehot

class DictModel():
    def __init__(self):
        self.dictionary: Dict[Any, Dict[Any, Tuple[Any, Any]]] = {}
    
    def update(self, x: Any, a: Any, xp: Any, r: Any, gamma: Any):
        """updates the model 
        
        Returns:
            Nothing
        """
        
        if x not in self.dictionary:
            self.dictionary[x] = {a:(xp, r, gamma)}
        else:
            self.dictionary[x][a] = (xp, r, gamma)

    def visited_states(self) -> List:
        return list(self.dictionary.keys())

    def visited_actions(self, x: Any) -> List:
        return list(self.dictionary[x].keys())

    def predict(self, x: Any, a: Any) -> Any:
        return self.dictionary[x][a]

class LinearApproximator():
    def __init__(self, input_features: int, output_features: int):
        self.input_features = input_features
        self.output_features = output_features
        self.weights = np.zeros((input_features, output_features))

    def grad(self, input_vec: npt.ArrayLike):
        # Not sure whether this works with multi-dimensional input vecs, but that's not necessary right now I think
        return np.tile(input_vec[:, np.newaxis], self.output_features)

    def update(self, step_size: float, grad: npt.ArrayLike):
        self.weights += step_size * grad

        if (np.isnan(np.sum(self.weights))):
            raise InvalidRunException("Nan encountered in approximator")
    
    def predict(self, input_vec: npt.ArrayLike):
        return np.dot(input_vec, self.weights)

class TabularApproximator(LinearApproximator):
    def predict(self, input_vec: List):
        return np.copy(self.weights[input_vec, :].flatten())