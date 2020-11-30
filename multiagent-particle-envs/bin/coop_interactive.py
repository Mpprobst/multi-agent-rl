#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import csv

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

from agents.coop_reinforce import CoopReinforce

TEST_INDEX = 10   # test after every 10 training episodes
NUM_TESTS = 10

class CoopInteractive():
    def __init__(self, scenario_file, episodes, good_agents, adversary_agents, verbose):
        # load scenario from script
        scenario = scenarios.load(scenario_file).Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
        # render call to create viewer window (necessary only for interactive policies)
        if verbose:
            env.render()
        "TODO: determine which agents are good and bad and give them different policies"
        agent = good_agents(env, env.n, 0.1)

        # find directory to save results
        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.split(current_directory)[0]
        parent_directory = os.path.split(parent_directory)[0]

        scenario_name = os.path.splitext(scenario_file)[0]
        filename = f'{parent_directory}/results/{scenario_name}_{agent.name}.csv'
        print(filename)
        with open(filename, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')

            # execution loop
            for i in range(1, episodes):
                if i % TEST_INDEX == 0:
                    scores = []
                    for t in range(NUM_TESTS):
                        value = self.run(env, agent, True, verbose)
                        scores.append(value)

                    avg_scores = np.mean(scores, axis=0)
                    avg_scores = np.mean(avg_scores)
                    #avg_scores_string = ["%.3f" % avg for avg in avg_scores]

                    print(f'TEST  %d:\t Avg Agent Rewards = %.3f' %(int(i / TEST_INDEX), avg_scores))
                    writer.writerow([i / TEST_INDEX, avg_scores])
                else:
                    self.run(env, agent, False, verbose)
                    agent.learn()

    def run(self, env, agent, istest, verbose):
        num_agents = env.n
        obs_n = env.reset()
        step_count = 0
        done_n = []
        scores = np.zeros(num_agents)
        while step_count < 200 and sum(done_n) == 0:
            # query for action from each agent's policy
            act_n = agent.action(obs_n)
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            step_count += 1
            #print(f'scores: {scores} rewards: {reward_n}')
            for i in range(num_agents):
                scores[i] += reward_n[i]

            if not istest:
                agent.update(reward_n)
            # render all agent views
            if verbose:
                env.render()

            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        return scores
