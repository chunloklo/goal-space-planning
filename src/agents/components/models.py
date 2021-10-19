from typing import Any, List, Dict, Tuple
import numpy as np
import numpy.typing as npt

from utils import numpy_utils

class DictModel():
    def __init__(self):
        self.dictionary: Dict[Any, Dict[Any, Tuple[Any, Any]]] = {}
    
    def update(self, x: Any, a: Any, xp: Any, r: Any):
        """updates the model 
        
        Returns:
            Nothing
        """
        
        if x not in self.dictionary:
            self.dictionary[x] = {a:(xp,r)}
        else:
            self.dictionary[x][a] = (xp,r)

    def visited_states(self) -> List:
        return list(self.dictionary.keys())

    def visited_actions(self, x: Any) -> List:
        return list(self.dictionary[x].keys())

    def predict(self, x: Any, a: Any) -> Any:
        return self.dictionary[x][a]

class LinearModel():
    def __init__(self, input_features: int, output_features: int, step_size: float):
        self.input_features = input_features
        self.output_features = output_features
        self.step_size = step_size
        self.weights = np.zeros((input_features, output_features))

    def update(self, input_vec: npt.ArrayLike, target_vec: npt.ArrayLike):
        prediction = self.predict(input_vec)
        self.weights += self.step_size * np.outer(input_vec, target_vec - prediction)
    
    def predict(self, input_vec: npt.ArrayLike):
        return np.dot(input_vec, self.weights)

class TabularModel(LinearModel):
    def update(self, input_vec: List, target_vec: npt.ArrayLike):
        prediction = self.predict(input_vec)
        self.weights[input_vec, :] += self.step_size * (target_vec - prediction)
    
    def predict(self, input_vec: List):
        return self.weights[input_vec, :].flatten()


def stacked_features(x: npt.ArrayLike, a: int, num_actions: int):
    assert len(x.shape) == 1, 'x must be a vector for stacked features'
    vec_len = x.shape[0]
    feature = np.zeros(vec_len * num_actions)
    feature[a * vec_len : (a+1) * vec_len] = x
    return feature

def stacked_tabular_features(x: int, a: int, num_features: int):
    return a * num_features + x

class LinearExpectationOptionModel():
    def __init__(self, num_features: int, num_actions: int, step_size: np.number):
        self.num_features: int = num_features
        self.num_actions: int = num_actions
        self.step_size: np.number = step_size
        assert self.step_size >= 0, 'Sanity check for negative step size?'
        # self.reward_model = np.zeros((self.num_features, self.num_actions))
        # self.discount_model = np.zeros((self.num_features, self.num_actions))
        self.transition_model = np.zeros((self.num_features, self.num_actions, self.num_features))

        self.reward_model = LinearModel(self.num_features * self.num_actions, 1, self.step_size)
        self.discount_model = LinearModel(self.num_features * self.num_actions, 1, self.step_size)
        self.transition_model = LinearModel(self.num_features * self.num_actions, self.num_features, self.step_size)
        
    def update(self, x: npt.ArrayLike, a: int, xp: npt.ArrayLike, r: float, env_gamma: float, option_gamma: float):
        """updates the model 
        
        Returns:
            Nothing
        """
        x_vec = stacked_features(x, a, self.num_actions)
        xp_vec = stacked_features(xp, a, self.num_actions)
        reward_target = r + env_gamma * option_gamma * self.reward_model.predict(xp_vec)
        self.reward_model.update(x_vec, np.array([reward_target]))

        discount_target = (1 - option_gamma) + option_gamma * env_gamma * self.discount_model.predict(xp_vec)
        self.discount_model.update(x_vec, np.array([discount_target]))

        transition_target = (1 - option_gamma) * xp + option_gamma * self.transition_model.predict(xp_vec)
        self.transition_model.update(x_vec, transition_target)

    def predict(self, x: npt.ArrayLike, a: int) -> Any:
        x_vec = stacked_features(x, a, self.num_actions)
        reward = self.reward_model.predict(x_vec)
        discount = self.discount_model.predict(x_vec)
        next_state = self.transition_model.predict(x_vec)
        return reward, discount, next_state

class TabularOptionModel():
    def __init__(self, num_features: int, num_actions: int, step_size: np.number):
        self.num_features: int = num_features
        self.num_actions: int = num_actions
        self.step_size: np.number = step_size
        assert self.step_size >= 0, 'Sanity check for negative step size?'
        
        self.reward_model = TabularModel(self.num_features * self.num_actions, 1, self.step_size)
        self.discount_model = TabularModel(self.num_features * self.num_actions, 1, self.step_size)
        self.transition_model = TabularModel(self.num_features * self.num_actions, self.num_features, self.step_size)
        
    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, option_gamma: float):
        """updates the model 
        
        Returns:
            Nothing
        """
        x_index = [stacked_tabular_features(x, a, self.num_features)]
        xp_index = [stacked_tabular_features(xp, a, self.num_features)]

        reward_target = r + env_gamma * option_gamma * self.reward_model.predict(xp_index)
        self.reward_model.update(x_index, np.array([reward_target]))

        discount_target = (1 - option_gamma) + option_gamma * env_gamma * self.discount_model.predict(xp_index)
        self.discount_model.update(x_index, np.array([discount_target]))
    
        xp_onehot = numpy_utils.create_onehot(self.num_features, xp)
        transition_target = (1 - option_gamma) * xp_onehot + option_gamma * self.transition_model.predict(xp_index)
        self.transition_model.update(x_index, transition_target)

    def predict(self, x: npt.ArrayLike, a: int) -> Any:
        x_index = stacked_tabular_features(x, a, self.num_features)
        # indexing to 0 to return the plain number
        reward = self.reward_model.predict(x_index)[0]
        discount = self.discount_model.predict(x_index)[0]
        next_state = self.transition_model.predict(x_index)
        return reward, discount, next_state