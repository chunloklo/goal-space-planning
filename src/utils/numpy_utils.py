import numpy as np
import numpy.typing as npt

def create_onehot(num_features: np.number, index: int):
    onehot = np.zeros(num_features)
    onehot[index] = 1
    return onehot

def softmax(vector: npt.ArrayLike) -> npt.ArrayLike:
    exp = np.exp(vector - np.max(vector))
    return exp / np.sum(exp)
