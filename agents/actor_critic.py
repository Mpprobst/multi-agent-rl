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
        self.rewards = [] #To collect reward per episode
        self.actions = [] #to store action  at last step of the episode??? why needed?
        self.gamma = GAMMA

    def action(self, obs):
        print(self.observation_space, self.action_space)
        print(np.array(obs).shape)
        net_op, _ = self.net.forward(obs) #extract the policy output as prob over actions
        print(net_op.shape)
        probabilities = F.softmax(net_op, dim=0)
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
        " This is to learn one agent for actor critic:"
        done = False
        observation = self.env.reset()
        print(observation)
        score = 0
        step = 0
        while not done and step < 200:
        #print('Let one episode begin')
            #print(step)
            action = self.action(observation) #Line 1 in while loop of pseudocode
            observation_, reward, done, info = self.env.step(action) #Line 2 in while loop of pseudocode
            score += reward
            self.actor_critic.optimizer.zero_grad() #need this else gradients will be accumulated
            #push the tensors to the device
            observation = T.tensor([observation], dtype=T.float).to(self.net.device)
            observation_ = T.tensor([observation_], dtype=T.float).to(self.net.device)
            reward = T.tensor(reward, dtype=T.float).to(self.net.device)
    
            _, critic_value = self.net.forward(observation)
            _, critic_value_ = self.net.forward(observation_)
    
            delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value #Line 3 in the while loop of pseudocode
            critic_loss = delta**2 # Equivalent to Line 4 in the while loop of pseudocode
            actor_loss = -self.log_prob*delta #Line 5 in the while loop of pseudocode
           
            (actor_loss + critic_loss).backward()
            self.net.optimizer.step()

           
            observation = observation_
            step += 1
           
        self.update(reward)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        