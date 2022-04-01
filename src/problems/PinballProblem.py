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
        self.goal_initiation_radius = 0.35

        self.goals[11][1] += 0.06

        goal_matrix = np.array(self.goals)
        max_speed = 0.2
        def goal_termination(s):
            coordinate_wise_close = np.isclose(s[:2], goal_matrix, atol = self.goal_radius)
            speed_close = np.sum(np.power(s[2:], 2)) <= np.power(max_speed, 2)
            terms = np.logical_and(np.all(coordinate_wise_close, axis=1), speed_close)

            return terms
        
        def goal_initiation(s):
            coordinate_wise_close = np.isclose(s[:2], goal_matrix, atol = self.goal_initiation_radius)
            inits = np.all(coordinate_wise_close, axis=1)
            return inits

        self.goal_termination_func = goal_termination
        self.goal_initiation_func = goal_initiation
        
class PinballTermProblem(PinballProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)

        goal_matrix = np.array(self.goals[:-2])

        # Replacing top right goal with terminal state
        terminal_state = self.env.terminal_state
        
        # self.goals[3] = terminal_state[:2]

        goal_matrix = np.array(self.goals)
        max_speed = 0.2
        def goal_termination(s):
            coordinate_wise_close = np.isclose(s[:2], goal_matrix, atol = self.goal_radius)
            speed_close = np.sum(np.power(s[2:], 2)) <= np.power(max_speed, 2)
            terms = np.logical_and(np.all(coordinate_wise_close, axis=1), speed_close)

            # print(s, terminal_state)
            # print(np.array_equal(s, terminal_state))
            # print(s == terminal_state)
            if np.array_equal(s, terminal_state):
                # print('termianted!')
                terms[3] = True
            else:
                terms[3] = False

            return terms
        
        goal_initiation_location_matrix = np.copy(goal_matrix)
        goal_initiation_location_matrix[3] = [0.9, 0.2]
        def goal_initiation(s):
            coordinate_wise_close = np.isclose(s[:2], goal_matrix, atol = self.goal_initiation_radius)
            inits = np.all(coordinate_wise_close, axis=1)
            return inits

        self.goal_termination_func = goal_termination
        self.goal_initiation_func = goal_initiation