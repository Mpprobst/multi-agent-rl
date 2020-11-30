"""
reinforce.py
Author: Diego Andrade
Purpose: Implements actor critic with seperate classes for actor and critic so the actos can share a critic
BASED ON A TUTORIAL BY MACHINE LEARNING WITH PHIL
https://www.youtube.com/watch?v=53y49DBxz8U
"""

import os,sys

import multiagent
import multiagent.policy as policy
import gym
import numpy as np
import random
import nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.98

class Actor():
    def __init__(self, env, agent_index, lr):
        ### I DON'T KNOW WHAT THE OBSERVATION SPACE IS I CANT RUN THIS -DIEGO
        self.id = agent_index
        self.env = env
        self.critic = None
        actions = env.action_space[self.id] # need action space for this agent
        #### I PUT THIS HERE BECAUSE MICHAEL HAD IT, AGAIN I CAN'T VISUALIZE WHATS GOING ON -DIEGO
        if isinstance(actions, multiagent.multi_discrete.MultiDiscrete):
            self.action_space = actions.shape
        else:
            self.action_space = actions.n
        self.observation_space = env.observation_space[self.id].shape[0]

        self.net = nn.NN(self.observation_space, self.action_space)
        self.optimizer = optim.Adam(self.net.parameters(), lr = lr)

class Critic():
    def __init__(self, env, lr):
        self.env = env
        "FIXME: unsure how to find full observation space for the whole gamestate"
        self.observation_space = env.observation_space[0].shape[0] # this is essentially the input space
        self.net = nn.NN(self.observation_space, 1) # critic output always has 1 value
        self.optimizer = optim.Adam(self.net.parameters(), lr = lr)

    def fix_obs(self, obs):
        resize = []
        # ensure obs is the proper length
        for i in range(self.observation_space):
            if i < len(obs):
                resize.append(obs[i])
            else:
                resize.append(0)
        return resize

class MASCAgent(policy.Policy):
    def __init__(self, env, agent_index, lr):
        super(MASCAgent, self).__init__()
        self.lr = lr
        self.gamma = GAMMA
        self.env = env
        self.actor = Actor(env, agent_index, lr)
        self.critic = None
        self.name = "MASCAgent"

    def assign_critic(self, critic):
        self.critic = critic

    # sometimes the observation is not the correct size, this enforces the obs to be a sigular size
    def fix_obs(self, obs):
        resize = []
        # ensure obs is the proper length
        for i in range(self.actor.observation_space):
            if i < len(obs):
                resize.append(obs[i])
            else:
                resize.append(0)
        return resize

    def action(self, observation):
        observation = self.fix_obs(observation)
        state = T.tensor([observation], dtype=T.float).to(self.actor.net.device)
        probabilities = self.actor.net.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob
        a = np.zeros(self.actor.action_space)
        a[action.item()] = 1
        return np.concatenate([a, np.zeros(self.env.world.dim_c)])

    def update(self):
        return None

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        state = self.critic.fix_obs(state)
        new_state = self.critic.fix_obs(new_state)

        state = T.tensor([state], dtype=T.float).to('cpu')
        new_state = T.tensor([new_state], dtype=T.float).to('cpu')
        reward = T.tensor(reward, dtype=T.float).to('cpu')

        critic_val = self.critic.net.forward(state)
        new_critic_val = self.critic.net.forward(new_state)

        delta = reward + self.gamma * new_critic_val * (1 - int(done)) - critic_val

        actor_loss = -self.log_prob * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()
