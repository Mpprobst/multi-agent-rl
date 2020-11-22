"""
cnn.py
Author: Michael Probst
Purpose: Implements a convolutional neural network for agents to use
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Net, self).__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.fc1 = nn.Linear(self.input_dims, 32)    #first layer
        self.fc2 = nn.Linear(32, 16)                #second layer
        self.fc3 = nn.Linear(16, self.output_dims)   #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    #Implements a feed forward network
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
