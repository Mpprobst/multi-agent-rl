"""
random.py
Purpose: Implements an agent that makes random decisions
"""

import multiagent.policy as policy
import multiagent

class RandomAgent:
    def __init__(self, env, agent_index, lr):
        super(ReinforceAgent, self).__init__()
        self.id = agent_index
        self.name = f'RandomAgent_{agent_index}'
        self.actions = env.action_space

    def action(self, state):
        return self.actions.sample()

    def update(self, reward):
        return 0

    def learn(self):
        return 0
