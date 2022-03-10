from src.problems.BaseProblem import BaseProblem
from src.environments.pinball import PinballEnvironment
import numpy as np

class PinballProblem(BaseProblem):
    def __init__(self, exp, idx, seed: int):
        super().__init__(exp, idx, seed)
        self.env = PinballEnvironment(self.params['pinball_configuration_file'], self.params['render'])
        self.actions = 5
        # self.options = get_options("Grazing")[0:3]
        self.gamma = self.params['gamma']

        self.goals = []
        border = 0.05
        for x in np.linspace(0 + border, 1 - border, 5):
            for y in np.linspace(0 + border, 1 - border, 5):
                self.goals.append((x,y))

        self.goal_radius = 0.02
        # 5x5 grid. There's approx 4 "gaps" between goals
        self.goal_initiation_radius = 0.25

        def goal_termination_gen(x, y):
            # Takes observation and tells you whether you are in the goal or not
            
            def goal_termination(s):
                # Defining goal func here to be relatively slow 
                speed_tolerance = 2
                # ignoring speed tolerance for now
                #  and np.isclose(np.square(s[2]) + np.square(s[3]), 0, np.square(speed_tolerance)):
                if np.isclose(s[0], x, atol = self.goal_radius) and np.isclose(s[1], y, atol = self.goal_radius):
                    return True
                else:
                    return False
            return goal_termination
        
        def goal_initiation_gen(x, y):
            def goal_initiation(s):
                # Defining goal func here to be relatively slow 
                speed_tolerance = 2
                # ignoring speed tolerance for now
                #  and np.isclose(np.square(s[2]) + np.square(s[3]), 0, np.square(speed_tolerance)):
                if np.isclose(s[0], x, atol = self.goal_initiation_radius) and np.isclose(s[1], y, atol = self.goal_initiation_radius):
                    return True
                else:
                    return False
            return goal_initiation

        self.goal_termination_funcs = [goal_termination_gen(g[0], g[1]) for g in self.goals]
        self.goal_initiation_funcs = [goal_initiation_gen(g[0], g[1]) for g in self.goals]