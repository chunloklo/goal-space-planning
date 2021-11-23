from typing import Dict, Tuple
from PyExpUtils.utils.random import sample
import numpy as np

class Buffer():
    """Class for efficient buffer implementation when storing various numpy/matrix data
    """
    def __init__(self, buffer_size: int, keys: Dict[str, Tuple], random_seed: int):
        """[summary]

        Args:
            buffer_size (int): max size of the buffer
            keys_sizes (Dict[str, Tuple]): dictionary of keys to the size of the input you want to store in the buffer
        """
        self.buffer_size = buffer_size
        self.keys = keys

        self.buffer_head = 0
        self.buffer_full = self.buffer_head >= self.buffer_size

        self.buffer = {}
        for key in keys:
            self.buffer[key] = np.zeros((self.buffer_size, *keys[key]))

        self.random = np.random.RandomState(random_seed)
        
    def update(self, data: Dict):
        for k in self.keys:
            self.buffer[k][self.buffer_head] = data[k]
        
        self.buffer_head += 1
        if self.buffer_head == self.buffer_size:
            self.buffer_full = True
        self.buffer_head %= self.buffer_size
        pass

    def sample(self, num_samples: int):
        # Currently sampling with replacement which shouldn't make too much of a difference
        sample_range = self.buffer_size if self.buffer_full else self.buffer_head 
        sample_indices = self.random.choice(sample_range, num_samples)

        return_dict = {}
        for k in self.keys:
            return_dict[k] = self.buffer[k][sample_indices]

        return return_dict

    pass