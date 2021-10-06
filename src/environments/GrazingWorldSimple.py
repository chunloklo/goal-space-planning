import numpy as np
from RlGlue import BaseEnvironment
from src.utils import globals
import random
from src.environments.GrazingWorld import GrazingWorld

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GrazingWorldSimple(GrazingWorld):
    def __init__(self, seed:int, size=10, reward_sequence_length=10):
        super().__init__(seed, size, reward_sequence_length)

        # Overriding goals
        self.goals = {
            1:{
                "position" : (2,2),
                "current_reward":0,
            },
            2:{
                "position" : (2,size-3),
                "current_reward":50,
            },
            3:{
                "position" : (size-3,size-3),
                "current_reward":0,
            }
        }

        # Changing terminal state to be the far right corner
        self.terminal_state_positions = [self.goals[2]["position"]]
            
    def update_goals(self):
        # All static goals. No need to update any of them
        pass

