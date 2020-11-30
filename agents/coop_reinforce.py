"""
coop_reinforce.py
Author: Michael Probst
Purpose: Creates a reinforce agent that treats all agents in the environment as one
by combining their observations before sending it to a neural net
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

class CoopReinforce(policy.Policy):
    def __init__(self, env, num_agents, lr):
        super(CoopReinforce, self).__init__()
        self.id = 0
        self.name = f'COOP_REINFORCE_{self.id}'
        self.env = env
        self.action_space = []
        self.observation_space = 0

        for i in range(num_agents):
            actions = [] # env.action_space is a list of Discrete actions for every agent
            actions = env.action_space[i]
            if isinstance(actions, multiagent.multi_discrete.MultiDiscrete):
                self.action_space.append(actions.shape)
            else:
                self.action_space.append(actions.n)

            self.observation_space += env.observation_space[i].shape[0]

        self.total_actions = sum(self.action_space)
        print(f'total obs: {self.observation_space} total actions: {self.action_space}')
        self.net = nn.NN(self.observation_space, self.total_actions)
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
        flat = []
        for ob in obs:
            for o in ob:
                flat.append(o)
        obs = flat
        resize = []
        # ensure obs is the proper length
        for i in range(self.observation_space):
            if i < len(obs):
                resize.append(obs[i])
            else:
                resize.append(0)
        return resize

    # given a matrix of observations from each agent.
    def action(self, obs):
        #print(f'{self.name} saw obs size {len(obs)}. expects size {self.observation_space}')
        obs = self.fix_obs(obs)

        probabilities = F.softmax(self.net.forward(obs), dim=0)
        prob_group = []
        prob_n = []
        i = 0
        for prob in probabilities:
            prob_group.append(prob.item())
            if len(prob_group) >= self.action_space[i]:
                prob_n.append(T.tensor(prob_group).to(self.net.device))
                prob_group = []
                i += 1
        action_probs = []
        for probs in prob_n:
            action_probs.append(T.distributions.Categorical(probs))

        log_probs = []
        actions = []
        for i in range(len(action_probs)):
            action = action_probs[i].sample()
            log_prob = action_probs[i].log_prob(action)
            log_probs.append(log_prob)

            actions.append(action)

        self.recent_action = log_probs

        a = []
        for i in range(len(actions)):
            a.append(np.zeros(self.action_space[i]).tolist())
            a[i][actions[i].item()] = 1
            a[i] = np.concatenate([a[i], np.zeros(self.env.world.dim_c)])
        return a

    def update(self, reward):
        self.actions.append(self.recent_action)
        self.rewards.append(reward)

    def learn(self):
        #print(f'learning on: {self.rewards}')
        Gt_n = []
        for t in range(len(self.rewards)):
            sum = np.zeros(self.env.n)
            discount = 1
            for k in range(t, len(self.rewards)):
                for i in range(len(sum)):
                    sum[i] += self.rewards[k][i] * discount
                discount *= GAMMA
            Gt_n.append(sum)

        self.action_memory.append(self.actions)
        self.reward_memory.append(self.rewards)
        self.return_memory.append(Gt_n)

        if len(self.actions) > self.max_ep_length:
            self.max_ep_length = len(self.actions)

        #complete the batch
        if len(self.return_memory) >= BATCH_SIZE:
            #print(f'a: {len(self.action_memory)} rew: {len(self.reward_memory)} ret: {len(self.return_memory)}')

            # pad memory with 0s
            for ep in range(len(self.return_memory)):
                padding = []
                for i in range(self.max_ep_length):
                    if i < len(self.return_memory[ep]):
                        padding.append(self.return_memory[ep][i])
                    else:
                        padding.append(np.zeros(self.env.n))
                    #self.return_memory[ep] = np.pad(self.return_memory[ep], (0, self.max_ep_length - len(self.return_memory[ep])), 'constant')
                self.return_memory[ep] = padding

            avgReturn = np.mean(self.return_memory, axis=0)

            losses = []
            for batch in range(len(self.action_memory)):
                loss = 0
                baseline_index = 0
                for g, logprob in zip(self.return_memory[batch], self.action_memory[batch]):
                    #print(f'g: {len(g)} baseline: {len(avgReturn[baseline_index])} logprob: {len(logprob)}')
                    for i in range(len(g)):
                        loss += (g[i] - avgReturn[baseline_index][i]) * -logprob[i]
                    baseline_index += 1
                losses.append(loss)

            loss = T.mean(T.stack(losses))
            loss = T.autograd.Variable(loss, requires_grad=True)
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
