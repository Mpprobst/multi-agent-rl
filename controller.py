"""
project3.py
Author: Michael Probst
Purpose: Solve Cartpole and Lunar Lander openAI gym environments
"""
import argparse
from agents.reinforce import ReinforceAgent
from agents.actor_critic import ACAgent
from agents.random_agent import RandomAgent
from agents.coop_reinforce import CoopReinforce
from agents.multiactor_singlecritic import MASCAgent
from  bin.interactive import Interactive
from  bin.coop_interactive import CoopInteractive


AGENT_MAP = {'reinforce' : ReinforceAgent,
             'ac' : ACAgent,
             'masc' : MASCAgent,
             'random' : RandomAgent }

COOP_AGENT_MAP = {'reinforce' : CoopReinforce,
                  'random' : RandomAgent }

parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('--agent1', choices=AGENT_MAP.keys(), default='ac', help='Can be ac, reinforce, masc, or random')
parser.add_argument('--agent2', choices=AGENT_MAP.keys(), default='random', help='Can be ac, reinforce, or random')
parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
parser.add_argument('--episodes', type=int, default = 5, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Visualize the environment.', action='store_true')
parser.add_argument('--coop', help='Choose coop agents if true, else independent agents.', action='store_true')

args = parser.parse_args()

agents = AGENT_MAP
if args.coop:
    agents = COOP_AGENT_MAP

agent_func1 = agents[args.agent1]
agent_func2 = agents[args.agent2]

if args.coop:
    CoopInteractive(args.scenario, args.episodes, agent_func1, agent_func2, args.verbose)
else:
    Interactive(args.scenario, args.episodes, agent_func1, agent_func2, args.verbose)
