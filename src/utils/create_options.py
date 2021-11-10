
import sys
import os
sys.path.append(os.getcwd())
from src.utils.Option import QOption



# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def get_options(option_name):
    if "GrazingAdam" in option_name:
        num_states = 95
        num_actions = 4
        policy_option_1 = {

            0:  RIGHT, 1: DOWN,   2:  LEFT,   3: LEFT, 4: LEFT, 5: LEFT,  6: LEFT,   7: LEFT,   8: LEFT,   9: LEFT,   10: LEFT,  11: LEFT,
            12: RIGHT, 13: RIGHT, 14: RIGHT,  15: UP,  16: LEFT,17: LEFT, 18: LEFT,  19: LEFT,  20: LEFT,  21: LEFT,  22: LEFT,  23: LEFT,
            24: RIGHT, 25: RIGHT, 26: RIGHT,  27: UP,  28: UP,  29: UP,   30: LEFT,  31: UP,    32: UP,    33: UP,    34: UP,    35: UP, 
            36: RIGHT, 37: RIGHT, 38: RIGHT,  39: UP,  40: UP,  41: UP,   42: LEFT,  43: LEFT,  44: RIGHT, 45: UP,    46: UP,    47: UP,    
            48: RIGHT, 49: RIGHT, 50: RIGHT,  51: UP,  52: UP,  53: UP,   54: LEFT,  55: LEFT,  56: LEFT,  57: UP,    58: UP,    59: UP, 
            60: RIGHT, 61: RIGHT, 62: RIGHT,  63: UP,  64: UP,  65: UP,   66: LEFT,  67: LEFT,  68: LEFT,  69: UP,    70: UP,    71: UP,
            72: RIGHT, 73: RIGHT, 74: RIGHT,  75: UP,  76: UP,  77: UP,   78: LEFT,  79: LEFT,  80: LEFT,  81: LEFT,  82: UP,    83: UP,
            84: RIGHT, 85: RIGHT, 86: RIGHT,  87: UP,  88: UP,  89: UP,   90: LEFT,  91: LEFT,  92: LEFT,  93: LEFT,  94: UP,    95: UP 
            
        }
        
        term_set = [13,30,81]

        def policy_selection(policy, state):
            return policy[state]

        def termination_condition(termination_set, state):

            if state in termination_set:
                return True
            return False

        option1 = QOption([x for x in range(num_states)], policy_option_1,
                        term_set, policy_selection,
                        termination_condition, num_actions)
                        
        policy_option_2 = {

            0:  RIGHT, 1: RIGHT,  2:  RIGHT,  3: RIGHT,  4: RIGHT,  5: RIGHT,  6: RIGHT,  7: DOWN,   8: LEFT,   9: LEFT,   10: LEFT,  11: LEFT,
            12: RIGHT, 13: RIGHT, 14: RIGHT,  15: RIGHT, 16: RIGHT, 17: RIGHT, 18: RIGHT, 19: DOWN,  20: LEFT,  21: LEFT,  22: LEFT,  23: LEFT,
            24: RIGHT, 25: RIGHT, 26: RIGHT,  27: UP,    28: UP,    29: UP,    30: LEFT,  31: UP,    32: UP,    33: UP,    34: UP,    35: UP, 
            36: RIGHT, 37: RIGHT, 38: RIGHT,  39: UP,    40: UP,    41: UP,    42: LEFT,  43: LEFT,  44: RIGHT, 45: UP,    46: UP,    47: UP,    
            48: RIGHT, 49: RIGHT, 50: RIGHT,  51: UP,    52: UP,    53: UP,    54: LEFT,  55: LEFT,  56: RIGHT, 57: UP,    58: UP,    59: UP, 
            60: RIGHT, 61: RIGHT, 62: RIGHT,  63: UP,    64: UP,    65: UP,    66: LEFT,  67: LEFT,  68: RIGHT, 69: UP,    70: UP,    71: UP,
            72: RIGHT, 73: RIGHT, 74: RIGHT,  75: UP,    76: UP,    77: UP,    78: LEFT,  79: LEFT,  80: UP,    81: RIGHT, 82: UP,    83: UP,
            84: RIGHT, 85: RIGHT, 86: RIGHT,  87: UP,    88: UP,    89: UP,    90: LEFT,  91: LEFT,  92: UP,    93: RIGHT, 94: UP,    95: UP 
            
        }
        

        option2 = QOption([x for x in range(num_states)], policy_option_2,
                        term_set, policy_selection,
                        termination_condition, num_actions)

        policy_option_3 = {

            0:  RIGHT, 1: RIGHT,  2:  RIGHT,  3: RIGHT,  4: RIGHT,  5: RIGHT,  6: RIGHT,  7: RIGHT,   8: RIGHT,  9: DOWN,  10: DOWN,  11: DOWN,
            12: RIGHT, 13: RIGHT, 14: RIGHT,  15: RIGHT, 16: RIGHT, 17: RIGHT, 18: RIGHT, 19: RIGHT,  20: RIGHT, 21: DOWN, 22: DOWN,  23: DOWN,
            24: RIGHT, 25: RIGHT, 26: RIGHT,  27: DOWN,  28: DOWN,  29: DOWN,  30: LEFT,  31: UP,     32: UP,    33: DOWN, 34: DOWN,  35: DOWN, 
            36: RIGHT, 37: RIGHT, 38: RIGHT,  39: DOWN,  40: DOWN,  41: DOWN,  42: LEFT,  43: LEFT,   44: RIGHT, 45: DOWN, 46: DOWN,  47: DOWN,    
            48: RIGHT, 49: RIGHT, 50: RIGHT,  51: RIGHT, 52: RIGHT, 53: RIGHT, 54: RIGHT, 55: RIGHT,  56: RIGHT, 57: DOWN, 58: DOWN,  59: DOWN, 
            60: RIGHT, 61: RIGHT, 62: RIGHT,  63: RIGHT, 64: RIGHT, 65: RIGHT, 66: RIGHT, 67: RIGHT,  68: RIGHT, 69: DOWN, 70: LEFT,  71: LEFT,
            72: RIGHT, 73: RIGHT, 74: RIGHT,  75: RIGHT, 76: RIGHT, 77: RIGHT, 78: RIGHT, 79: RIGHT,  80: RIGHT, 81: DOWN, 82: LEFT,  83: LEFT,
            84: RIGHT, 85: RIGHT, 86: RIGHT,  87: RIGHT, 88: RIGHT, 89: RIGHT, 90: RIGHT, 91: RIGHT,  92: RIGHT, 93: UP,   94: LEFT,  95: LEFT 
            
        }

        option3 = QOption([x for x in range(num_states)], policy_option_3,
                        term_set, policy_selection,
                        termination_condition, num_actions)

        return [option1, option2, option3]
    elif "Grazing" in option_name:
        num_states = 100
        num_actions = 4
        
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
        
        term_set = [22,27,77]

        def policy_selection(policy, state):
            return policy[state]

        def termination_condition(termination_set, state):

            if state in termination_set:
                return True
            return False

        option1 = QOption([x for x in range(num_states)], policy_option_1,
                        term_set, policy_selection,
                        termination_condition, num_actions)

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

        option2 = QOption([x for x in range(num_states)], policy_option_2,
                        term_set, policy_selection,
                        termination_condition, num_actions)

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

        option3 = QOption([x for x in range(num_states)], policy_option_3,
                        term_set, policy_selection,
                        termination_condition, num_actions)


        # Option 4 lead to the middle of the gridworld

        option_4_term_set = [44] + term_set

        policy_option_4 = {

            0:  RIGHT,  1: RIGHT,   2: RIGHT,   3: RIGHT,  4: DOWN,   5: LEFT,   6: LEFT, 7: LEFT,   8: LEFT,    9: LEFT,
            10: DOWN,  11: UP,     12: UP,     13: UP,    14: DOWN,  15: LEFT,  16: UP,   17: UP,    18: UP,    19: DOWN,
            20: DOWN,  21: UP,     22: UP,     23: UP,    24: DOWN,  25: LEFT,  26: UP,   27: UP,    28: UP,    29: DOWN,     
            30: DOWN,  31: UP,     32: UP,     33: UP,    34: DOWN,  35: LEFT,  36: UP,   37: UP,    38: UP,    39: DOWN,    
            40: RIGHT, 41: RIGHT,  42: RIGHT,  43: RIGHT, 44: UP,    45: LEFT, 46: LEFT,  47: LEFT,  48: LEFT,  49: LEFT,   
            50: RIGHT, 51: RIGHT,  52: RIGHT,  53: RIGHT, 54: UP,    55: LEFT, 56: LEFT,  57: LEFT,  58: LEFT,  59: LEFT, 
            60: RIGHT, 61: RIGHT,  62: RIGHT,  63: RIGHT, 64: UP,    65: LEFT, 66: LEFT,  67: LEFT,  68: LEFT,  69: LEFT,
            70: RIGHT, 71: RIGHT,  72: RIGHT,  73: RIGHT, 74: UP,    75: LEFT, 76: LEFT,  77: UP,    78: LEFT,  79: LEFT, 
            80: RIGHT, 81: RIGHT,  82: RIGHT,  83: RIGHT, 84: UP,    85: LEFT, 86: LEFT,  87: LEFT,  88: LEFT,  89: LEFT,  
            90: RIGHT, 91: RIGHT,  92: RIGHT,  93: RIGHT, 94: UP,    95: LEFT, 96: LEFT,  97: LEFT,  98: LEFT,  99: LEFT
        }

        option4 = QOption([x for x in range(num_states)], policy_option_4,
                        option_4_term_set, policy_selection,
                        termination_condition, num_actions)
        return [option1, option2, option3, option4]
    else:
        raise NotImplementedError()

