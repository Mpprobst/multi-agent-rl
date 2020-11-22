"""
actor_critic.py
Author:
Purpose: Implements an actor critic agent
"""

import cnn

class actor_critic:
    def __init__(self):
        self.net = cnn(1,1,1)
