"""
random.py
Purpose: Implements an agent that makes random decisions
"""

import marl
from marl.agent import TrainableAgent
from marl.agent import MATrainable

from marl.tools import ClassSpec, _std_repr, is_done, reset_logging
import marl.policy.policies as policy
from marl.exploration import ExplorationProcess
from marl.experience import ReplayMemory, PrioritizedReplayMemory

class RandomAgent(TrainableAgent, MATrainable):
    def __init__(self, env, lr, index=None, mas=None):
        TrainableAgent.__init__(self, policy=policy.RandomPolicy(env.action_space), observation_space=env.observation_space, action_space=env.action_space, gamma=0.98, lr=lr, name="RandomAgent", )
        MATrainable.__init__(self, mas, index)
        
    def update_model(self, t):
        """
        Update the model.
        """
        return 0
