#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:30:28 2020
To test the avaliable agents working as single independent agents.
@author: subashkhanal
"""

import marl
from marl.agent import DQNAgent,QTableAgent, DQNAgent, DeepACAgent, PHCAgent, DDPGAgent, MADDPGAgent
from marl.model.nn import MlpNet

import gym

env = gym.make("LunarLander-v2")

obs_s = env.observation_space
act_s = env.action_space

mlp_model = MlpNet(8,4, hidden_size=[64, 32])
actor_model = mlp_model
critic_model = MlpNet(8,1, hidden_size=[64, 32])
agent = DQNAgent(actor_model, obs_s, act_s, experience="ReplayMemory-5000", exploration="EpsGreedy", lr=0.001, name="DQNAgent")
agent = QTableAgent(obs_s, act_s, exploration="EpsGreedy", gamma=0.99, lr=0.1, target_update_freq=None, name="QTableAgent")
agent = DeepACAgent(critic_model,actor_model,obs_s, act_s, experience="ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.95, batch_size=32, target_update_freq=None, name="DeepACAgent")
agent = PHCAgent(obs_s, act_s, exploration="EpsGreedy", delta=0.01, lr_critic=0.01, gamma=0.95, target_update_freq=None, name="PHCAgent")
agent = DDPGAgent(critic_model, actor_model, obs_s, act_s, experience="ReplayMemory-1000", exploration="OUNoise", lr_actor=0.01, lr_critic=0.01, gamma=0.95, batch_size=32, target_update_freq=None, name="DDPGAgent")
agent = MADDPGAgent(critic_model, actor_model, obs_s, act_s, index=None, experience="ReplayMemory-1000", exploration="OUNoise", lr_actor=0.01, lr_critic=0.01, gamma=0.95, batch_size=32, tau=0.01, use_target_net=100, name="MADDPGAgent")
# # Train the agent for 100 000 timesteps
#agent.learn(env, nb_timesteps=100000)

# Test the agent for 10 episodes
#agent.test(env, nb_episodes=10)