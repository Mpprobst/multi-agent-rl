"""
cnn.py
Author: Michael Probst
Purpose: Implements a convolutional neural network for agents to use
"""
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

IMAGE_HEIGHT = 250
IMAGE_WIDTH = 160
INPUT_CHANNELS = 3

class CNN(nn.Module):
    # input_dims are for the image size, output_dims is number of actions
    def __init__(self, num_features):
        super(CNN, self).__init__()
        # Conv network
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.num_features = num_features
        #(color_channels=3 due to RGB, num_filters, filter_size=2)
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)    # num_filters = same as correlating conv layer
        self.conv2 = nn.Conv2d(32, 32, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        "TODO: Consider making 3 more conv layers and 1 more max pool"

        # Deep network
        self.input_dims = self.calc_input_dims()
        print(f'input_dims={self.input_dims}')
        self.fc1 = nn.Linear(self.input_dims, self.num_features)    #first layer

        self.to(self.device)

    def calc_input_dims(self):
        batch_data = T.zeros((1, INPUT_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT ))
        batch_data = self.conv1(batch_data)
        #batch_data = self.conv2(batch_data)
        #batch_data = self.conv3(batch_data)
        batch_data = self.maxpool1(batch_data)
        return int(np.prod(batch_data.size()))

    #Implements a feed forward network. input is an image
    def forward(self, input):
        input = input.transpose()
        input = T.tensor(input, dtype=T.float).to(self.device)
        input = input.unsqueeze(0)

        batch_data = self.conv1(input)
        batch_data = F.relu(batch_data)

        #batch_data = self.conv2(batch_data)
        #batch_data = F.relu(batch_data)

        #batch_data = self.conv3(batch_data)
        #batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)
        batch_data = batch_data.view(batch_data.size()[0], -1) #flatten

        out = self.fc1(batch_data)
        return out
