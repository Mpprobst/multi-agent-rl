"""
project3.py
Author: Michael Probst
Purpose: Solve Cartpole and Lunar Lander openAI gym environments
"""
import argparse
import csv
import reinforce

AGENT_MAP = {'ac' : cartpole.CartPole,
             'reinforce' : lunarlander.LunarLander }

parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--agent1', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('--agent2', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('--numEpisodes', type=int, default = 500, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Visualize the environment.', action='store_true')
args = parser.parse_args()

envFunc = ENVIRONMENTS_MAP[args.env]
agengFunc = AGENT_MAP[args.agent1]

env = envFunc(args.numEpisodes, reinforce.ReinforceAgent, args.verbose)
