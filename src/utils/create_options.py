
import sys
import os
sys.path.append(os.getcwd())

from src.environments.MazeWorld import MazeWorld
from src.environments.GrazingWorld import GrazingWorld
from src.utils.options import load_option, save_option
from src.utils.Option import QOption



# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def create_options(env, option_name):

    # option 1 - go to goal state 1, i.e. the state that gives reward 50 or 0


    if "Grazing" in option_name:

        policy_option_1 = {

            0:  RIGHT, 1: RIGHT,  2:  DOWN,    3: LEFT,   4: LEFT,  5: LEFT,   6: LEFT,     7: LEFT,   8: LEFT,  9: LEFT,
            10: UP,    11: UP,    12: DOWN,   13: UP,    14: UP,    15: UP,    16: UP,     17: UP,    18: UP,   19: UP,
            20: UP,    21: UP,    22: UP,     23: UP,    24: UP,    25: UP,    26: UP,     27: UP,    28: UP,   29: UP,     
            30: UP,    31: UP,    32: UP,     33: UP,    34: UP,    35: UP,    36: UP,     37: UP,    38: UP,   39: UP,    
            40: UP,    41: LEFT,  42: RIGHT,  43: RIGHT, 44: UP,    45: UP,    46: LEFT,   47: LEFT,  48: LEFT, 49: LEFT,   
            50: UP,    51: LEFT,  52: RIGHT,  53: RIGHT, 54: UP,    55: UP,    56: LEFT,   57: LEFT,  58: LEFT, 59: LEFT, 
            60: UP,    61: LEFT,  62: RIGHT,  63: RIGHT, 64: UP,    65: UP,    66: LEFT,   67: LEFT,  68: LEFT, 69: LEFT,
            70: UP,    71: LEFT,  72: RIGHT,  73: RIGHT, 74: UP,    75: UP,    76: LEFT,   77: UP,    78: LEFT, 79: LEFT, 
            80: UP,    81: LEFT,  82: RIGHT,  83: RIGHT, 84: UP,    85: UP,    86: LEFT,   87: LEFT,  88: LEFT, 89: LEFT,  
            90: UP,    91: LEFT,  92: RIGHT,  93: RIGHT, 94: UP,    95: UP,    96: LEFT,   97: LEFT,  98: LEFT, 99: LEFT
            
        }

        
        term_set_1 = [22,27,77]

        def policy_selection_1(policy, state):
            return policy[state]

        def termination_condition_1(termination_set, state):

            if state in termination_set:
                return True
            return False





        option1 = QOption([x for x in range(env.nS)], policy_option_1,
                        term_set_1, policy_selection_1,
                        termination_condition_1, env.nA)

        save_option('GrazingO1', option1)

        # Option 2 - go to goal state 2, i.e. the state that gives reward 40 or 0

        policy_option_2 = {

            0:  RIGHT, 1: RIGHT,   2:  RIGHT,   3: RIGHT,  4: RIGHT,  5: RIGHT,  6: RIGHT,   7: DOWN,   8: LEFT,   9: LEFT,
            10: UP,    11: UP,     12: UP,     13: UP,    14: UP,    15: UP,    16: UP,     17: DOWN,  18: UP,    19: UP,
            20: UP,    21: UP,     22: UP,     23: UP,    24: UP,    25: UP,    26: UP,     27: UP,    28: UP,    29: UP,     
            30: UP,    31: UP,     32: UP,     33: UP,    34: UP,    35: UP,    36: UP,     37: UP,    38: UP,    39: UP,    
            40: RIGHT, 41: RIGHT,  42: RIGHT,  43: RIGHT, 44: UP,    45: UP,    46: LEFT,   47: LEFT,  48: RIGHT, 49: UP,   
            50: RIGHT, 51: RIGHT,  52: RIGHT,  53: RIGHT, 54: UP,    55: UP,    56: LEFT,   57: LEFT,  58: RIGHT, 59: UP, 
            60: RIGHT, 61: RIGHT,  62: RIGHT,  63: RIGHT, 64: UP,    65: UP,    66: LEFT,   67: LEFT,  68: RIGHT, 69: UP,
            70: RIGHT, 71: RIGHT,  72: RIGHT,  73: RIGHT, 74: UP,    75: UP,    76: LEFT,   77: UP,    78: RIGHT, 79: UP, 
            80: RIGHT, 81: RIGHT,  82: RIGHT,  83: RIGHT, 84: UP,    85: UP,    86: LEFT,   87: LEFT,  88: RIGHT, 89: UP,  
            90: RIGHT, 91: RIGHT,  92: RIGHT,  93: RIGHT, 94: UP,    95: UP,    96: LEFT,   97: LEFT,  98: RIGHT, 99: UP
            
        }
        term_set_2 = [22,27,77]

        def policy_selection_2(policy, state):
            return policy[state]

        def termination_condition_2(termination_set, state):
            if state in termination_set:
                return True
            return False

        option2 = QOption([x for x in range(env.nS)], policy_option_2,
                        term_set_2, policy_selection_2,
                        termination_condition_2, env.nA)


        save_option('GrazingO2', option2)

        # return option1, option2


        # Option 3 - go to the goal 3, i.e. state that gives a small but steady reward

        policy_option_3 = {

            0:  RIGHT,  1: RIGHT,   2: RIGHT,   3: RIGHT,  4: DOWN,   5: DOWN,   6: LEFT,    7: LEFT,   8: RIGHT, 9:  DOWN,
            10: DOWN,  11: UP,     12: UP,     13: UP,    14: DOWN,  15: DOWN,  16: UP,     17: UP,    18: UP,    19: DOWN,
            20: DOWN,  21: UP,     22: UP,     23: UP,    24: DOWN,  25: DOWN,  26: UP,     27: UP,    28: UP,    29: DOWN,     
            30: DOWN,  31: UP,     32: UP,     33: UP,    34: DOWN,  35: DOWN,  36: UP,     37: UP,    38: UP,    39: DOWN,    
            40: RIGHT, 41: RIGHT,  42: RIGHT,  43: RIGHT, 44: RIGHT, 45: RIGHT, 46: RIGHT,  47: DOWN,  48: LEFT,  49: LEFT,   
            50: RIGHT, 51: RIGHT,  52: RIGHT,  53: RIGHT, 54: RIGHT, 55: RIGHT, 56: RIGHT,  57: DOWN,  58: LEFT,  59: LEFT, 
            60: RIGHT, 61: RIGHT,  62: RIGHT,  63: RIGHT, 64: RIGHT, 65: RIGHT, 66: RIGHT,  67: DOWN,  68: LEFT,  69: LEFT,
            70: RIGHT, 71: RIGHT,  72: RIGHT,  73: RIGHT, 74: RIGHT, 75: RIGHT, 76: RIGHT,  77: UP,    78: LEFT,  79: LEFT, 
            80: RIGHT, 81: RIGHT,  82: RIGHT,  83: RIGHT, 84: RIGHT, 85: RIGHT, 86: RIGHT,  87: UP,    88: LEFT,  89: LEFT,  
            90: RIGHT, 91: RIGHT,  92: RIGHT,  93: RIGHT, 94: RIGHT, 95: RIGHT, 96: RIGHT,  97: UP,    98: LEFT,  99: LEFT
            
        }
        term_set_3 = [22,27,77]

        def policy_selection_3(policy, state):
            return policy[state]

        def termination_condition_3(termination_set, state):
            if state in termination_set:
                return True
            return False

        option3 = QOption([x for x in range(env.nS)], policy_option_3,
                        term_set_3, policy_selection_3,
                        termination_condition_3, env.nA)

        save_option('GrazingO3', option3)

    elif "Maze" in option_name:
        policy_option_1 = {

            0:  RIGHT, 1: RIGHT,  2:  DOWN,    3: LEFT,  4: LEFT,   5: LEFT,   6: LEFT,    7: LEFT,   8: LEFT,  9: LEFT,
            10: RIGHT, 11: RIGHT, 12: DOWN,   13: LEFT,  14: LEFT,  15: LEFT,  16: LEFT,   17: LEFT,  18: LEFT, 19: LEFT,
            20: RIGHT, 21: RIGHT, 22: DOWN,   23: LEFT,  24: LEFT,  25: LEFT,  26: LEFT,   27: LEFT,  28: LEFT, 29: LEFT,     
            30: UP,    31: UP,    32: UP,     33: UP,    34: UP,    35: UP,    36: LEFT,   37: LEFT,  38: LEFT, 39: LEFT,    
            40: UP,    41: UP,    42: UP,     43: UP,    44: UP,    45: UP,    46: LEFT,   47: LEFT,  48: LEFT, 49: LEFT,   
            50: UP,    51: UP,    52: UP,     53: UP,    54: UP,    55: UP,    56: LEFT,   57: LEFT,  58: LEFT, 59: LEFT, 
            60: UP,    61: UP,    62: UP,     63: UP,    64: UP,    65: UP,    66: LEFT,   67: LEFT,  68: LEFT, 69: LEFT,
            70: UP,    71: UP,    72: UP,     73: UP,    74: UP,    75: UP,    76: LEFT,   77: LEFT,  78: LEFT, 79: LEFT, 
            80: UP,    81: UP,    82: UP,     83: UP,    84: UP,    85: UP,    86: LEFT,   87: LEFT,  88: LEFT, 89: LEFT,  
            90: UP,    91: UP,    92: UP,     93: UP,    94: UP,    95: UP,    96: LEFT,   97: LEFT,  98: LEFT, 99: LEFT
            
        }

        term_set_1 = [22,27,77]

        def policy_selection_1(policy, state):
            return policy[state]

        def termination_condition_1(termination_set, state):

            if state in termination_set:
                return True
            return False

        option1 = QOption([x for x in range(env.nS)], policy_option_1,
                        term_set_1, policy_selection_1,
                        termination_condition_1, env.nA)

        save_option('MazeO1', option1)

        # Option 2 - go to goal state 2, i.e. the state that gives reward 40 or 0

        policy_option_2 = {

            0:  RIGHT, 1:  RIGHT,   2: RIGHT,  3:  RIGHT, 4: RIGHT,  5: RIGHT,  6: RIGHT,   7: DOWN,   8: LEFT,  9:  LEFT,
            10: RIGHT, 11: RIGHT,  12: RIGHT,  13: RIGHT, 14: RIGHT, 15: RIGHT, 16: RIGHT,  17: DOWN,  18: LEFT, 19: LEFT,
            20: RIGHT, 21: RIGHT,  22: RIGHT,  23: RIGHT, 24: RIGHT, 25: RIGHT, 26: RIGHT,  27: DOWN,  28: LEFT, 29: LEFT,     
            30: RIGHT, 31: RIGHT,  32: RIGHT,  33: RIGHT, 34: UP,    35: UP,    36: UP,     37: UP,    38: UP,   39: UP,    
            40: RIGHT, 41: RIGHT,  42: RIGHT,  43: RIGHT, 44: UP,    45: UP,    46: UP,     47: UP,    48: UP,   49: UP,   
            50: RIGHT, 51: RIGHT,  52: RIGHT,  53: RIGHT, 54: UP,    55: UP,    56: UP,     57: UP,    58: UP,   59: UP, 
            60: RIGHT, 61: RIGHT,  62: RIGHT,  63: RIGHT, 64: UP,    65: UP,    66: UP,     67: UP,    68: UP,   69: UP,
            70: RIGHT, 71: RIGHT,  72: RIGHT,  73: RIGHT, 74: UP,    75: UP,    76: UP,     77: UP,    78: UP,   79: UP, 
            80: RIGHT, 81: RIGHT,  82: RIGHT,  83: RIGHT, 84: UP,    85: UP,    86: UP,     87: UP,    88: UP,   89: UP,  
            90: RIGHT, 91: RIGHT,  92: RIGHT,  93: RIGHT, 94: UP,    95: UP,    96: UP,     97: UP,    98: UP,   99: UP
            
        }
        term_set_2 = [22,27,77]

        def policy_selection_2(policy, state):
            return policy[state]

        def termination_condition_2(termination_set, state):
            if state in termination_set:
                return True
            return False

        option2 = QOption([x for x in range(env.nS)], policy_option_2,
                        term_set_2, policy_selection_2,
                        termination_condition_2, env.nA)
        option2 = QOption([x for x in range(env.nS)], policy_option_2,
                        term_set_2, policy_selection_2,
                        termination_condition_2, env.nA)


        save_option('MazeO2', option2)

        # return option1, option2


        # Option 3 - go to the goal 3, i.e. state that gives a small but steady reward

        policy_option_3 = {

            0:  DOWN,  1:  DOWN,   2:  RIGHT,  3: RIGHT,   4: DOWN,  5:  DOWN,  6:  DOWN,   7:  RIGHT, 8:  DOWN,  9:  DOWN,
            10: DOWN,  11: DOWN,   12: RIGHT,  13: RIGHT, 14: DOWN,  15: DOWN,  16: DOWN,   17: RIGHT, 18: DOWN,  19: DOWN,
            20: DOWN,  21: DOWN,   22: RIGHT,  23: RIGHT, 24: DOWN,  25: DOWN,  26: DOWN,   27: LEFT,  28: DOWN,  29: DOWN,     
            30: DOWN,  31: DOWN,   32: RIGHT,  33: RIGHT, 34: DOWN,  35: DOWN,  36: DOWN,   37: DOWN,  38: DOWN,  39: DOWN,    
            40: RIGHT, 41: RIGHT,  42: RIGHT,  43: RIGHT, 44: RIGHT, 45: RIGHT, 46: RIGHT,  47: DOWN,  48: LEFT,  49: LEFT,   
            50: RIGHT, 51: RIGHT,  52: RIGHT,  53: RIGHT, 54: RIGHT, 55: RIGHT, 56: RIGHT,  57: DOWN,  58: LEFT,  59: LEFT, 
            60: RIGHT, 61: RIGHT,  62: RIGHT,  63: RIGHT, 64: RIGHT, 65: RIGHT, 66: RIGHT,  67: DOWN,  68: LEFT,  69: LEFT,
            70: RIGHT, 71: RIGHT,  72: RIGHT,  73: RIGHT, 74: RIGHT, 75: RIGHT, 76: RIGHT,  77: DOWN,  78: LEFT,  79: LEFT, 
            80: RIGHT, 81: RIGHT,  82: RIGHT,  83: RIGHT, 84: RIGHT, 85: RIGHT, 86: RIGHT,  87: UP,    88: LEFT,  89: LEFT,  
            90: RIGHT, 91: RIGHT,  92: RIGHT,  93: RIGHT, 94: RIGHT, 95: RIGHT, 96: RIGHT,  97: UP,    98: LEFT,  99: LEFT
            
        }
        term_set_3 = [22,27,77]

        def policy_selection_3(policy, state):
            return policy[state]

        def termination_condition_3(termination_set, state):
            if state in termination_set:
                return True
            return False

        option3 = QOption([x for x in range(env.nS)], policy_option_3,
                        term_set_3, policy_selection_3,
                        termination_condition_3, env.nA)

        save_option('MazeO3', option3)


if __name__ == '__main__':

    if sys.argv[1].lower() == "maze":
        env = MazeWorld(0)
        option_name = "Maze0"
    elif sys.argv[1].lower() == "graze" or sys.argv[1].lower() == "grazing":
        env = GrazingWorld(0)
        option_name = "Grazing0"
   
    create_options(env, option_name)

    # test load
    # o1 = load_option('GrazingO1')
    # print(o1.policy)
    # o2 = load_option('Grazing02')
    # print(o2.policy)
    # o3 = load_option('GrazingO3')
    # print(o3.policy)
