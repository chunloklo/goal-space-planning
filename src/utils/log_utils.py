from typing import Callable
from src.utils import globals

def run_if_should_log(log_func: Callable):
    try:
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            log_func()
    except KeyError as e:
        pass
    
