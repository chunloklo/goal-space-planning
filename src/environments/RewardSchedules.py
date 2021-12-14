import abc
from typing import List, Literal
from src.utils import globals, param_utils

class ConstantRewardSchedule():
    def __init__(self, constant: float):
        self.constant = constant
        pass

    def __call__(self):
        return self.constant

class CyclingRewardSchedule():
    def __init__(self, rewards: List[float], duration: int, cycle_offset: int = 0, cycle_type:Literal['episode', 'step']='episode', start_pause: int=0):
        """[summary]

        Args:
            rewards (List): list of rewards that will be cycled between in order
            duration (int): duration of stay for each reward
            cycle_offset (int, optional): [description]. Defaults to 0. Offset in cycle you want to start with (by cycleType)
            cycle_type (Literal['episode', 'step', optional): [description]. Defaults to 'episode'. Whether the reward cycles based on steps or episodes
            start_pause (int): Number of iterations to pause the cycle in the beginning before starting
        """
        
        self.rewards = rewards
        self.duration = duration
        self.cycle_offset = cycle_offset
        self.cycle_type = param_utils.check_valid(cycle_type, lambda x: x in ['episode', 'step'])
        self.start_pause = start_pause
        pass

    def __call__(self):
        try:
            timeline_location = globals.blackboard['num_episodes_passed'] if self.cycle_type == 'episode' else globals.blackboard['num_steps_passed']
        except Exception as e:
            print(type(e))
            timeline_location = 0

        if timeline_location < self.start_pause:
            timeline_location = 0
        else:
            timeline_location -= self.start_pause
        
        timeline_location += self.cycle_offset

        rewards_location = (timeline_location // self.duration) % len(self.rewards)
        return self.rewards[rewards_location]
