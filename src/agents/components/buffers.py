from concurrent.futures import process
from typing import Callable, Dict, Set, Tuple, Any
from PyExpUtils.utils.random import sample
import numpy as np

class DictBuffer():
    def __init__(self):
        self.dict_buffer: Dict = {}
    
    def update(self, item: Any, priority = 1):
        prev_len = len(self.dict_buffer.keys())
        self.dict_buffer[item] = priority
        new_len = len(self.dict_buffer.keys())
        # if (prev_len != new_len):
        #     print(item)
        #     if (new_len >= 345):
        #         raise NotImplementedError()
            
            
        # print(len(self.dict_buffer.keys()))
    
    def sample(self, random: np.random, num_samples: int, alpha: float=0.5): 
        items = list(self.dict_buffer.items())
        priorities = [item[1] for item in items]

        # calculating according to prioritized experience replay
        # https://arxiv.org/pdf/1511.05952.pdf equation 1
        exp = np.power(priorities, alpha)
        probs = exp / np.sum(exp)

        elements = [item[0] for item in items]
        sample_indices = random.choice(range(len(elements)), size=num_samples, p=probs)
        samples = [elements[i] for i in sample_indices]
        return samples


class Buffer():
    """Class for efficient buffer implementation when storing various numpy/matrix data
    """
    def __init__(self, buffer_size: int, keys: Dict[str, Tuple], random_seed: int, type_map={}):
        """[summary]

        Args:
            buffer_size (int): max size of the buffer
            keys_sizes (Dict[str, Tuple]): dictionary of keys to the size of the input you want to store in the buffer
            random_seed (int): seed for the random state for sampling
            type_map (Optional[dict]): A dict of non-default types that you want to use
        """
        self.buffer_size = buffer_size
        self.keys = keys

        self.buffer_head = 0
        self.num_in_buffer = 0
        self.buffer_full = self.buffer_head >= self.buffer_size

        self.buffer = {}
        for key in keys:
            dtype = type_map.get(key, np.float64)
            self.buffer[key] = np.zeros((self.buffer_size, *keys[key]), dtype=dtype)

        self.random = np.random.RandomState(random_seed)
        
    def update(self, data: Dict):
        for k in self.keys:
            self.buffer[k][self.buffer_head] = data[k]
        
        self.buffer_head += 1
        if not self.buffer_full:
            self.num_in_buffer += 1

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
            return_dict[k] = np.copy(self.buffer[k][sample_indices])

        return return_dict

    def weighted_sample(self, num_samples: int, weight_key: str, process_func: Callable = lambda p: p):
        # print(self.buffer.keys())
        sample_range = self.buffer_size if self.buffer_full else self.buffer_head 
        processed_weights = process_func(self.buffer[weight_key][:sample_range])
        processed_sum = np.sum(processed_weights)

        if any(processed_weights < 0):
            raise ValueError('processed weights contain negative value')

        if processed_sum == 0:
            # Won't be able to get linalg.norm. Just return none
            return None

        prob = processed_weights / processed_sum
        
        # print(self.buffer_size)
        # print(self.buffer_head)
        # print(sample_range)
        # print(num_samples)
        # print(prob.shape)
        sample_indices = self.random.choice(sample_range, num_samples, p=prob)

        return_dict = {}
        for k in self.keys:
            return_dict[k] = np.copy(self.buffer[k][sample_indices])

        return return_dict

    pass