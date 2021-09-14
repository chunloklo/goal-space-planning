from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice

class Dyna_Linear_Dist:
    def __init__(self, features: int, actions: int, params: Dict, seed: int):
        self.features = features
        self.actions = actions
        self.params = params

        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.gamma = params['gamma']

        self.a = -1
        self.x = -1

        self.F = np.zeros((actions, actions))
        self.b = np.zeros(features)

        # create initial weights
        self.w = np.zeros((actions, features))
        self.model = {}

    def FA(self):
        return "Linear"

    def __str__(self):
        return "Dyna_Linear_Dist"


    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            return choice(np.arange(self.actions), self.random)

        return argmax(self.w.dot(x))

    def update(self, x, a, xp, r, gamma):
        q_a = self.w[a].dot(x)
        ap = self.selectAction(xp)
        qp_ap = max(self.w.dot(xp))

        g = r + gamma * qp_ap
        delta = g - q_a

        self.w[a] += self.alpha * delta * x

        self.F += self.alpha*(xp-self.F@x).dot(x)
        self.b += self.alpha*(r-self.b.dot(x))*x

        temp =xp

        self.update_model(x,a,xp,r)  
        self.planning_step(gamma)

        return ap

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        if x not in self.model:
            self.model[x] = {}
        # if we are not at a goal state, just store xp and r
        if xp !=-1:
            self.model[x][a] = (xp,r)
        # only need to calculate probabilities for last state
        else:
            # if model[x] is empty, set total_times to 1, otherwise increase it by 1 and set a = -1
            if not self.model[x]:
                self.model[x]["total_times"] = 0
                self.model[x]["actions"] = set()
            self.model[x]["total_times"] +=1
            self.model[x]["actions"].add(a)
            # if xp and r have been encountered before, increase count by 1, else add 1
            if (xp,r) not in self.model[x]:
                self.model[x][(xp,r)] = {}
            self.model[x][(xp,r)]["count"] = self.model[x][(xp,r)].get("count",0) +1

            # (re)calculate probabilities of all xp-r pairs
            for k,v in self.model[x].items(): 
                if k != "total_times" and k!="actions":
                    self.model[x][k]["prob"] = self.model[x][k]["count"] / self.model[x]["total_times"]

        
    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """


        # distribution model: k=1, sample according to probability
        # sample model: keep track of encountered xp's and r's, and sample from them according to an _arbitrary_ distribution
        # expectation model: xp and r are chosen according to the expectation of the distribution
        for i in range(self.planning_steps):
            x = np.random.rand(self.actions, self.features)
            xp = self.F@x
            r = self.b.dot(x)
            self.w += self.alpha * (r + gamma * self.w.dot(xp) - self.w.dot(x))*x
            
            # if in terminal state: get rewards and associated probabilities to be able to sample accordingly
            if "total_times" in self.model[x]:
                actions = list(self.model[x]["actions"])
                a = choice(np.array(actions ), self.random) 
                xp_r_probs = [ (k, v["prob"]) for k,v in self.model[x].items() if k!="total_times" and k!= "actions" ]

                # calculate expectation if there are more than one (xp,r) tuples
                xp_r_pairs = [xp_r_probs[i][0] for i in range(len(xp_r_probs)) ]
                probs = [xp_r_probs[i][1] for i in range(len(xp_r_probs)) ]
                sampled_xpr_pairs = random.choices(xp_r_pairs, weights=probs, k=1 ) # sample just weighted by probs
                xp,r = sampled_xpr_pairs[0]


            # if not in terminal state just get xp and r
            else:
                
                actions = list(self.model[x].keys()) 
                a = choice(np.array(actions ), self.random)
                xp,r = self.model[x][a]       
            if xp ==-1:
                max_q = 0
            else:
                max_q = max(self.w.dot(xp))

            self.w[a] = self.w[a] + self.alpha * (r + gamma * max_q - self.Q[x, a])

        q_a = self.w[a].dot(x)
        qp_ap = max(self.w.dot(xp))

        g = r + gamma * qp_ap
        delta = g - q_a

        self.w[a] += self.alpha * delta * x


            
    def agent_end(self, x, a, r, gamma):
        # Model Update step
        self.update_model(x,a, -1, r)
        # Planning
        self.planning_step(gamma)