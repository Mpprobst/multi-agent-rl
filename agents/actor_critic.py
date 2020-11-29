"""
actor_critic.py
Author: Subash Khanal
Purpose: Implements an agent using the actor_critic policy gradient algorithm
"""
import multiagent
import multiagent.policy as policy
import gym
import numpy as np
import random
import ac_nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

GAMMA = 0.99


class ACAgent(policy.Policy):
    def __init__(self, env, agent_index, lr):
        super(ACAgent, self).__init__()
        self.id = agent_index
        self.name = f'ActorCritic_{agent_index}'
        #self.env = env
        self.env = env
        actions = env.action_space[self.id] # env.action_space is a list of Discrete actions for every agent
        if isinstance(actions, multiagent.multi_discrete.MultiDiscrete):
            self.action_space = actions.shape
        else:
            self.action_space = actions.n

        self.observation_space = env.observation_space[self.id].shape[0]
        self.net = nn.NN(self.observation_space, self.action_space)
    
        self.recent_action = None
        self.gamma = GAMMA

    def action(self, obs):
        
        net_op, _ = self.net.forward(obs) #extract the policy output as prob over actions
        #print(net_op.shape)
        probabilities = F.softmax(net_op, dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob
        #print(log_prob)
        a = np.zeros(self.action_space)
        a[action.item()] = 1
        return np.concatenate([a, np.zeros(self.env.world.dim_c)])

    def update(self):
        return None
               
    def learn(self, state, reward, state_, done):
        self.net.optimizer.zero_grad() #need this else gradients will be accumulated
       #push the tensors to the device
        state = T.tensor([state], dtype=T.float).to(self.net.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.net.device)
        reward = T.tensor(reward, dtype=T.float).to(self.net.device)

        _, critic_value = self.net.forward(state)
        _, critic_value_ = self.net.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value #Line 3 in the while loop of pseudocode
        critic_loss = delta**2 # Equivalent to Line 4 in the while loop of pseudocode
        actor_loss = -self.log_prob*delta #Line 5 in the while loop of pseudocode
        #print(actor_loss, critic_loss)
        final_loss = actor_loss + critic_loss
        final_loss = Variable(final_loss, requires_grad = True)
        #print(final_loss)
        final_loss.backward()
        self.net.optimizer.step()     