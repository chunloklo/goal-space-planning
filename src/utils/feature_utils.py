import numpy.typing as npt
import numpy as np

def stacked_features(x: npt.ArrayLike, a: int, num_actions: int):
    assert len(x.shape) == 1, 'x must be a vector for stacked features'
    vec_len = x.shape[0]
    feature = np.zeros(vec_len * num_actions)
    feature[a * vec_len : (a+1) * vec_len] = x
    return feature

def stacked_tabular_features(x: int, a: int, num_state_features: int):
    return a * num_state_features + x