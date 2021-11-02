class RlGlue:
    def __init__(self, agent, env):
        self.environment = env
        self.agent = agent
        self.last_action = None
        self.last_state = None
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self):
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_state = obs
        self.last_action = self.agent.start(obs)

        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action)
        self.num_episodes += 1
        obs = self.observationChannel(s)
        self.total_reward += reward
        if term:
            self.agent.end(reward)

            roat = (reward, obs, None, term)
        else:
            self.last_action = self.agent.step(reward, obs)     
            roat = (reward, obs, self.last_action, term)
        

        self.last_state = obs
        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def execute_option(self, a):
        t = False
        total_reward = 0
        total_steps = 0

        option = self.agent.options[a - self.environment.nA]
        termination_condition = False

        while not termination_condition:
            x = self.agent.state_encoding(self.last_state)
            a, termination_condition = option.step(x)

            r,xp, t = self.environment.step(a)  # taking action
            
            total_reward += r * (self.agent.gamma ** total_steps)
            total_steps += 1

            if t:
                break

            self.last_state = xp

        return total_reward, x, t, total_steps     

    def runEpisode(self, max_steps = 0):
        is_terminal = False
        
        self.start()
        self.num_steps = 0
        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            self.num_steps += 1
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]
        return is_terminal

    def observationChannel(self, s):
        return s

    def recordTrajectory(self, s, a, r, t):
        pass
