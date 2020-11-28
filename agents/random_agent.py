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
            self.actions = actions.shape
        else:
            self.actions = actions.n
        print(self.actions)

    # returns an action randomly.
    def action(self, state):
        a = np.zeros(self.actions)
        a[random.randrange(0, self.actions)] = 1
        return np.concatenate([a, np.zeros(self.env.world.dim_c)])

    def update(self, reward):
        return 0

    def learn(self):
        return 0
