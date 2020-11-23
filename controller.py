"""
project3.py
Author: Michael Probst
Purpose: Solve Cartpole and Lunar Lander openAI gym environments
"""
import argparse
import csv
import double_dunk
import actor_critic
import reinforce
import random_agent

AGENT_MAP = {'reinforce' : reinforce.ReinforceAgent,
             'ac' : actor_critic.ACAgent,
             'random' : random_agent.RandomAgent }

parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--agent1', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('--agent2', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('--episodes', type=int, default = 500, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Visualize the environment.', action='store_true')
args = parser.parse_args()

agent_func = AGENT_MAP[args.agent1]

env = double_dunk.DoubleDunk(args.episodes, agent_func, args.verbose)
