from typing import Any, Callable, List, Dict, Tuple, Literal
import numpy as np
import numpy.typing as npt
from src.agents.components.approximators import DictModel
from src.utils import numpy_utils, globals, param_utils

# This right now assumes that you use an action model. However, 'td' and 'close' doesn't use
# the action model at all, thus we might want to refactor this later on.
class ActionModelSearchControl_Tabular():
    def __init__(self, random):
        state_sample_type = param_utils.parse_param(globals.param, 'search_control', lambda p : p in ['random', 'current', 'td'])
        self.state_sample_type = state_sample_type
        self.random = random

        if self.state_sample_type == 'td':
            self.replay_buffer = []
            self.replay_size = 5000

    def update(self, x: int, xp: int):
        if self.state_sample_type == 'td':
            # This function should go between behaviour learner update and the planning update
            delta = globals.blackboard['learner_delta']
            epsilon = 0.01
            self.replay_buffer.append((x, np.abs(delta) + epsilon))
            if len(self.replay_buffer) > self.replay_size:
                self.replay_buffer.pop(0)
        # Need to figure out what to do here

    def sample_states(self, num_samples: int, action_model: DictModel, x: int, xp: int) -> List[int]:
        if self.state_sample_type == 'random' or self.state_sample_type == 'current':
            samples = []
            for _ in range(num_samples):
                if self.state_sample_type =="random":
                    plan_x = self.random.choice(np.array(action_model.visited_states()))
                elif self.state_sample_type =="current":
                    visited_states = list(action_model.visited_states())
                    if (xp in list(visited_states)):
                        plan_x = xp
                    else:
                        # The previous state if you haven't visited the next state yet
                        plan_x = x
                samples.append(plan_x)

        elif self.state_sample_type == 'td':
            state_buffers = np.array([s[0] for s in self.replay_buffer])
            weights = np.array([s[1] for s in self.replay_buffer])
            probs = numpy_utils.softmax(weights)
            samples = self.random.choice(state_buffers, num_samples, p=probs)

        return samples
            