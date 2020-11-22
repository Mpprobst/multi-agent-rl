"""
reinforce.py
Author: Michael Probst
Purpose: Implements an agent using the REINFORCE policy gradient algorithm
"""
import gym
import numpy as np
import random
import cnn
import torch as T
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.98
BATCH_SIZE = 5

class reinforce_agent:
    def __init__(self, env, lr):
        self.name = 'REINFORCE'
        self.epsilon = 1
        self.state_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.net = cnn.cnn(self.state_dims, self.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.recent_action = None
        self.rewards = []
        self.actions = []
        self.action_memory = []
        self.reward_memory = []
        self.return_memory = []
        self.max_ep_length = 0
        self.losses = []

    def get_state_tensor(self, state):
        return T.tensor(state).to(self.net.device).float()

    def best_action(self, state):
        state_tensor = self.get_state_tensor(state)
        probabilities = F.softmax(self.net.Forward(state_tensor), dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.recent_action = log_probs
        return action.item()

    def get_action(self, env, state):
        return self.best_action(state)

    def update(self, state, nextState, action, reward):
        self.actions.append(self.recent_action)
        self.rewards.append(reward)

    def learn(self):
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
            self.max_ep_length = 0

        self.actions = []
        self.rewards = []
