#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:17:45 2020

@author: subashkhanal
"""

import marl
from marl import MARL
from marl.agent import MAACAgent, DeepACAgent
from marl.exploration import EpsGreedy
from marl.model.nn import MlpNet

from soccer import DiscreteSoccerEnv
# Environment available here "https://github.com/blavad/soccer"
env = DiscreteSoccerEnv(nb_pl_team1=3, nb_pl_team2=3) # for teamsize 1: (Discrete(800), Discrete(5))

obs_s = env.observation_space
act_s = env.action_space

critic_model = MlpNet(800,1, hidden_size=[6400, 3200])
actor_model = MlpNet(800,5, hidden_size=[6400, 3200])
print(actor_model)
ac_agent1 = DeepACAgent(critic_model, actor_model, obs_s, act_s, experience="ReplayMemory-50000", exploration="EpsGreedy", name="SoccerJ1")
# ac_agent2 = MAACAgent(critic_model, actor_model, obs_s, act_s, experience="ReplayMemory-50000", exploration="EpsGreedy", name="SoccerJ2")
#This is throwing error. If we can debug we can easily set up game. Why is not it accepting the actor model?


# # Create the trainable multi-agent system
# mas = MARL(agents_list=[ac_agent1, ac_agent2])

# # Assign MAS to each agent
# ac_agent1.set_mas(mas)
# ac_agent2.set_mas(mas)

# # Train the agent for 100 000 timesteps
# mas.learn(env, nb_timesteps=100000)

# # Test the agents for 10 episodes
# mas.test(env, nb_episodes=10, time_laps=0.5)

