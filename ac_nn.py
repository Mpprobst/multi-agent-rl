#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:14:54 2020
This is neural network for actor_critic implementation
@author: subashkhanal
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):# The actor critic network. 2 hidden layers are shared with separate final layer for actor and critic each.
    def __init__(self,input_size, actions_size,fc1_dims=128, fc2_dims=256,lr=0.001):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy = nn.Linear(fc2_dims, actions_size)
        self.value = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state).to(self.device).float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = T.tensor(self.policy(x)).to(self.device).float()
        value = T.tensor(self.value(x)).to(self.device).float()

        return (policy, value)
