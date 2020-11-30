"""
random.py
Purpose: Implements an agent that makes random decisions
"""

import multiagent.policy as policy
import multiagent
import random
import numpy as np

class RandomAgent(policy.Policy):
    def __init__(self, env, agent_index, lr):
        super(RandomAgent, self).__init__()
        self.id = agent_index
        self.name = f'RandomAgent_{self.id}'
        self.env = env

        actions = env.action_space[self.id] # env.action_space is a list of Discrete actions for every agent
        if isinstance(actions, multiagent.multi_discrete.MultiDiscrete):
            self.action_space = actions.shape
        else:
            self.action_space = actions.n
        #print(self.action_space)

    # returns an action randomly.
    def action(self, state):
        a = np.zeros(self.action_space)
        a[random.randrange(0, self.action_space)] = 1
        return np.concatenate([a, np.zeros(self.env.world.dim_c)])

    def update(self):
        return 0

    def learn(self, state, reward, state_, done):
        return 0
