"""
project3.py
Author: Michael Probst
Purpose: Solve Cartpole and Lunar Lander openAI gym environments
"""
import argparse
import csv
import double_dunk
import agents.reinforce as reinforce
import agents.actor_critic as ac
import agents.random_agent as random_agent
import sys
sys.path.append("/Users/subashkhanal/Desktop/Fall_2020/Sequential_Decision_Making/Final_Project/multi-agent-rl/multiagent-particle-envs/bin")
import interactive as Interactive

AGENT_MAP = {'reinforce' : reinforce.ReinforceAgent,
             'ac' : ac.ACAgent,
             'random' : random_agent.RandomAgent }

parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--agent1', choices=AGENT_MAP.keys(), default='ac', help='Can be ac, reinforce, or random')
parser.add_argument('--agent2', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
parser.add_argument('--episodes', type=int, default = 5, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Visualize the environment.', action='store_true')
args = parser.parse_args()

agent_func1 = AGENT_MAP[args.agent1]
agent_func2 = AGENT_MAP[args.agent1]

Interactive.Interactive(args.scenario, args.episodes, agent_func1, agent_func2, args.verbose)

#env = double_dunk.DoubleDunk(args.episodes, agent_func, args.verbose)
