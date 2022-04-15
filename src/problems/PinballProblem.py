from src.problems.BaseProblem import BaseProblem
from src.environments.pinball import PinballEnvironment
import numpy as np

class PinballProblem(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = PinballEnvironment(self.params['pinball_configuration_file'], self.params['render'], self.params['explore_env'])
        self.actions = 5
        # self.options = get_options("Grazing")[0:3]
        self.gamma = self.params['gamma']

        self.goals = []
        # n is the number of goals "per side". Total number of goals is n x n
        n = 4
        border = 0.05
    
        for y in np.linspace(0 + border, 1 - border, n):
            for x in np.linspace(0 + border, 1 - border, n):
                self.goals.append([x,y])

        self.num_goals = len(self.goals)

        self.goal_radius = 0.04
        # 5x5 grid. There's approx 4 "gaps" between goals
        self.goal_initiation_radius = 0.6
        
        self.goals[10][1] += 0.01
        self.goals[11][1] += 0.06

        self.goal_matrix = np.array(self.goals)
        self.goal_speeds = np.zeros(self.goal_matrix.shape)
        self.max_speed_difference = 0.2
        def goal_termination(s):
            coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_radius)
            speed_close = np.sum(np.power(s[2:] - self.goal_speeds, 2), axis=1) <= np.power(self.max_speed_difference, 2)
            terms = np.logical_and(np.all(coordinate_wise_close, axis=1), speed_close)

            return terms
        
        def goal_initiation(s):
            coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_initiation_radius)
            inits = np.all(coordinate_wise_close, axis=1)
            return inits

        self.goal_termination_func = goal_termination
        self.goal_initiation_func = goal_initiation
        
class PinballContinuingProblemWithTermState(PinballProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)

        # Replacing top right goal with terminal state
        terminal_state = self.env.terminal_state
        
        def goal_termination(s):
            coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_radius)
            speed_close = np.sum(np.power(s[2:] - self.goal_speeds, 2), axis=1) <= np.power(self.max_speed_difference, 2)
            terms = np.logical_and(np.all(coordinate_wise_close, axis=1), speed_close)

            if np.array_equal(s, terminal_state):
                # print('termianted!')
                terms[3] = True
            else:
                terms[3] = False

            return terms
        
        goal_initiation_location_matrix = np.copy(self.goal_matrix)
        goal_initiation_location_matrix[3] = [0.9, 0.2]
        def goal_initiation(s):
            coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_initiation_radius)
            inits = np.all(coordinate_wise_close, axis=1)
            return inits

        self.goal_termination_func = goal_termination
        self.goal_initiation_func = goal_initiation

class PinballTermProblem(PinballProblem):
    # THIS IS SPECIFICALLY FOR DREAMER. THIS ENV TERMINATES
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = PinballEnvironment(self.params['pinball_configuration_file'], self.params['render'], self.params['explore_env'], False)

        # Replacing top right goal with terminal state
        terminal_state = self.env.terminal_state

        def goal_termination(s):
            coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_radius)
            speed_close = np.sum(np.power(s[2:] - self.goal_speeds, 2), axis=1) <= np.power(self.max_speed_difference, 2)
            terms = np.logical_and(np.all(coordinate_wise_close, axis=1), speed_close)

            if np.array_equal(s, terminal_state):
                # print('termianted!')
                terms[3] = True
            else:
                terms[3] = False

            return terms
        
        goal_initiation_location_matrix = np.copy(self.goal_matrix)
        goal_initiation_location_matrix[3] = [0.9, 0.2]
        def goal_initiation(s):
            coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_initiation_radius)
            inits = np.all(coordinate_wise_close, axis=1)
            return inits

        self.goal_termination_func = goal_termination
        self.goal_initiation_func = goal_initiation

class PinballOracleProblem(PinballProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)

        # Replacing top right goal with terminal state
        terminal_state = self.env.terminal_state
        
        # self.goals[3] = terminal_state[:2]

        oracle_adjusted_goals = {
            9: np.array([0.26634864519615703, 0.6574340050572484, 0.38341283870096077, -0.9752487531218751]),
            5: np.array([0.35442983272352163, 0.34282851754474797, 0.7613925418191196, -0.7762487531218751]),
            6: np.array([0.6409535893184335, 0.3271380768559643, 0.995, -0.17232614294967943])
        }

        for goal_num in oracle_adjusted_goals:
            self.goals[goal_num] = oracle_adjusted_goals[goal_num][:2]
            self.goal_speeds[goal_num] = oracle_adjusted_goals[goal_num][2:]

        self.goal_matrix = np.array(self.goals)
        def goal_termination(s):
            # coordinate_wise_close = np.isclose(s[:2], self.goal_matrix, atol = self.goal_radius)
            coordinate_wise_close = np.sum(np.power(self.goal_matrix - s[:2], 2), axis=1) <= np.power(self.goal_radius, 2)
            # print(coordinate_wise_close)
            speed_close = np.sum(np.power(s[2:] - self.goal_speeds, 2), axis=1) <= np.power(self.max_speed_difference, 2)
            terms = np.logical_and(coordinate_wise_close, speed_close)
            
            # if any(terms):
            #     print(terms, np.all(coordinate_wise_close, axis=1), speed_close)
            # if terms[1]:
            #     print(s)
            #     sdfds

            if np.array_equal(s, terminal_state):
                # print('termianted!')
                terms[3] = True
            else:
                terms[3] = False

            return terms
        
        goal_initiation_location_matrix = np.copy(self.goal_matrix)
        goal_initiation_location_matrix[3] = [0.9, 0.2]
        def goal_initiation(s):
            coordinate_wise_close = np.sum(np.power(self.goal_matrix - s[:2], 2), axis=1) <= np.power(self.goal_initiation_radius, 2)
            # inits = np.all(coordinate_wise_close, axis=1)
            return coordinate_wise_close

        self.goal_termination_func = goal_termination
        self.goal_initiation_func = goal_initiation