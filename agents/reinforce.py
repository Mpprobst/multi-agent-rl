"""
reinforce.py
Author: Michael Probst
Purpose: Implements an agent using the REINFORCE policy gradient algorithm
"""
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
BATCH_SIZE = 5

class ReinforceAgent(policy.Policy):
    def __init__(self, env, agent_index, lr):
        super(ReinforceAgent, self).__init__()
        self.id = agent_index
        self.name = f'REINFORCE_{agent_index}'
        #self.env = env
        self.env = env
        actions = env.action_space[self.id] # env.action_space is a list of Discrete actions for every agent
        if isinstance(actions, multiagent.multi_discrete.MultiDiscrete):
            self.action_space = actions.shape
        else:
            self.action_space = actions.n

        self.observation_space = env.observation_space[self.id].shape[0]
        print(f'total obs: {self.observation_space} total actions: {self.action_space}')

        self.net = nn.NN(self.observation_space, self.action_space)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.recent_action = None
        self.rewards = []
        self.actions = []
        self.action_memory = []
        self.reward_memory = []
        self.return_memory = []
        self.max_ep_length = 0
        self.losses = []

    # sometimes the observation is not the correct size, this enforces the obs to be a sigular size
    def fix_obs(self, obs):
        resize = []
        # ensure obs is the proper length
        for i in range(self.observation_space):
            if i < len(obs):
                resize.append(obs[i])
            else:
                resize.append(0)
        return resize

    def action(self, obs):
        #print(f'{self.name} saw obs size {len(obs)}. expects size {self.observation_space}')
        obs = self.fix_obs(obs)
        probabilities = F.softmax(self.net.forward(obs), dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.recent_action = log_probs
        a = np.zeros(self.action_space)
        a[action.item()] = 1
        return np.concatenate([a, np.zeros(self.env.world.dim_c)])

    def update(self, reward):
        self.actions.append(self.recent_action)
        self.rewards.append(reward)

    def learn(self):
        #print(f'learning on: {self.rewards}')
        Gt = np.zeros_like(self.rewards, dtype=np.float64)
        for t in range(len(self.rewards)):
            sum = 0
            discount = 1
            for k in range(t, len(self.rewards)):
                sum += self.rewards[k] * discount
                discount *= GAMMA
            Gt[t] = sum

        self.action_memory.append(self.actions)
        self.reward_memory.append(self.rewards)
        self.return_memory.append(Gt)

        if len(self.actions) > self.max_ep_length:
            self.max_ep_length = len(self.actions)

        #complete the batch
        if len(self.return_memory) >= BATCH_SIZE:
            # pad memory with 0s
            for ep in range(len(self.return_memory)):
                self.return_memory[ep] = np.pad(self.return_memory[ep], (0, self.max_ep_length - len(self.return_memory[ep])), 'constant')

            avgReturn = np.zeros([self.max_ep_length])
            avgReturn = np.mean(self.return_memory, axis=0)

            losses = []
            for batch in range(len(self.action_memory)):
                loss = 0
                baseline_index = 0
                for g, logprob in zip(self.return_memory[batch], self.action_memory[batch]):
                    loss += (g - avgReturn[baseline_index]) * -logprob
                    baseline_index += 1
                losses.append(loss)

            loss = T.mean(T.stack(losses))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())

            self.action_memory = []
            self.return_memory = []
            self.reward_memory = []
            self.max_ep_length = 0

        self.actions = []
        self.rewards = []
