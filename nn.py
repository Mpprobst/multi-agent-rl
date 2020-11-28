"""
nn.py
Author: Michael Probst
Purpose: Implements a neural network for agents to use
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(NN, self).__init__()
        self.outputDims = outputDims
        self.inputDims = inputDims
        self.fc1 = nn.Linear(self.inputDims, 32)    #first layer
        self.fc2 = nn.Linear(32, 16)                #second layer
        self.fc3 = nn.Linear(16, self.outputDims)   #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    #Implements a feed forward network. state an array of 4 floats describing current state
    def forward(self, obs):
        obs = T.tensor(obs).to(self.device).float()
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
