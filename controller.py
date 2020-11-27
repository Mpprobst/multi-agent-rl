"""
project3.py
Author: Michael Probst
Purpose: Solve Cartpole and Lunar Lander openAI gym environments
"""
import argparse
import csv
import double_dunk
import soccer_env
import marl.agent as agent
import reinforce
import random_agent

AGENT_MAP = {'reinforce' : reinforce.ReinforceAgent,
             'ac' : agent.MAACAgent,
             'random' : random_agent.RandomAgent }

"TODO: add option to include agents on the same team to be different types of agents"
parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--agent1', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('--agent2', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('--team_size', type=int, default=1, help='The number of agents on each team')
parser.add_argument('--episodes', type=int, default = 500, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Visualize the environment.', action='store_true')
args = parser.parse_args()

team1_agents = []
team2_agents = []

for i in range(args.team_size):
    agent_func1 = AGENT_MAP[args.agent1]
    agent_func2 = AGENT_MAP[args.agent2]
    team1_agents.append(agent_func1)
    team2_agents.append(agent_func2)

env = soccer_env.SoccerEnv(args.episodes, team1_agents, team2_agents, args.verbose)
#env = double_dunk.DoubleDunk(args.episodes, agent_func, args.verbose)
