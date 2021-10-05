import dill
from typing import Union
import numpy as np
from src.utils.Option import QOption

def load_option(file_name):
    input_file = open("src/options/" + file_name + ".pkl", 'rb')
    return dill.load(input_file)


def save_option(file_name, option):
    print(file_name)
    output = open("src/options/" + file_name + ".pkl", 'wb')
    dill.dump(option, output)
    output.close()


###################################
# Agent utils for options. This might belong somewhere else later, but its still really small
####################################

# This converts an action index to an option index
def from_action_to_option_index(a: int, num_actions: int):
    return a - num_actions

# This converts an option index to an action index
def from_option_to_action_index(o: int, num_actions: int):
    return o + num_actions

def get_option_info(x, o, options):
    if (o >= len(options)):
        raise Exception(f'tried to get info for option {o} when there are only {len(options)} options')
    return options[o].step(x)

def get_action_consistent_options(x: int, a: Union[list, int], options: list, convert_to_actions=False, num_actions=None):
    action_consistent_options = []
    for o in range(len(options)):
        a_o, t = get_option_info(x, o, options)
        if (isinstance(a, (int, np.integer))):
            if (a_o == a):
                action_consistent_options.append(o)
        elif (isinstance(a, list)):
            if (a_o in a):
                action_consistent_options.append(o)
        else:
            raise NotImplementedError(f'_get_action_consistent_option does not yet supports this type {type(a)}. Feel free to add support!')

    if convert_to_actions:
        assert num_actions != None, 'num_actions must be set if convert_to_actions is set to true'
        action_consistent_options = [from_option_to_action_index(o, num_actions) for o in action_consistent_options]
    return action_consistent_options