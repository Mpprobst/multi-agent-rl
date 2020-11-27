#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:45:42 2020
This is the implementation of a vanila Actor Critic Policy gradient algorithm
@author: subashkhanal
"""
#It is closely based on https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code

import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class AC_NN(nn.Module):
    def __init__(self, lr, input_size, actions_size, fc1_dims=256, fc2_dims=256):
        super(AC_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy = nn.Linear(fc2_dims, actions_size)
        self.value = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = self.policy(x)
        value = self.value(x)

        return (policy, value)

class Agent():
    def __init__(self, lr, input_space, fc1_size, fc2_size, action_space, 
                 gamma=0.99):
       
        self.lr = lr
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.actor_critic = AC_NN(lr, input_space, action_space, 
                                               fc1_size, fc2_size)
        self.gamma = gamma
        self.log_prob = None

    def action_selection(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()


# env = gym.make('LunarLander-v2')
# agent = Agent(gamma=0.99, lr=5e-6, input_dims=[8], n_actions=4,
#               fc1_dims=2048, fc2_dims=1536)
# n_games = 3000
#run_avg = 100

env = gym.make("CartPole-v0")
agent = Agent(gamma=0.99, lr=0.001, input_space=4, action_space=2,
              fc1_size=128, fc2_size=256)
n_games = 600
run_avg = 10
scores = []
print('Let the game begin')

for i in tqdm(range(n_games)):
    
    done = False
    observation = env.reset()
    score = 0
    while not done:
        #print('Let one episode begin')
        action = agent.action_selection(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.learn(observation, reward, observation_, done)
        observation = observation_
    scores.append(score)
    #print(scores)

    avg_score = np.mean(scores[-run_avg:]) #last 10 for cartpole, last 100 for lunarlander
    print('episode ', i, 'score %.1f' % score,
            'average score %.1f' % avg_score)

np.save('/Users/subashkhanal/Desktop/Fall_2020/Sequential_Decision_Making/Assignment3/cartpole3.npy',scores)
running_avg = np.zeros(len(scores))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(scores[max(0, i-run_avg):(i+1)])
plt.plot(running_avg)
plt.title('Running average of previous 10 scores')