#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import csv

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

TEST_INDEX = 10   # test after every 10 training episodes
NUM_TESTS = 10

class Interactive():
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
        # create interactive policies for each agent
        policies = [good_agents(env, i, 0.1) for i in range(env.n)]

        # find directory to save results
        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.split(current_directory)[0]
        parent_directory = os.path.split(parent_directory)[0]

        scenario_name = os.path.splitext(scenario_file)[0]
        filename = f'{parent_directory}/results/{scenario_name}_{policies[0].name}.csv'
        with open(filename, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')

            # execution loop
            for i in range(1, episodes):
                if i % TEST_INDEX == 0:
                    scores = []
                    for t in range(NUM_TESTS):
                        value = self.run(env, policies, True, verbose)
                        scores.append(value)

                    avg_scores = np.mean(scores, axis=0)
                    avg_scores_string = ["%.3f" % avg for avg in avg_scores]

                    print(f'TEST  %d:\t Avg Agent Rewards = %s' %(int(i / TEST_INDEX), avg_scores_string))
                    writer.writerow([i / TEST_INDEX, avg_scores])
                else:
                    self.run(env, policies, False, verbose)
                    for policy in policies:
                        policy.learn()

    def run(self, env, policies, istest, verbose):
        obs_n = env.reset()
        step_count = 0
        done_n = []
        scores = np.zeros(len(policies))
        while step_count < 200 and sum(done_n) == 0:# and done_n == False:
            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            step_count += 1

            for i, policy in enumerate(policies):
                scores[i] += reward_n[i]
                if not istest:
                    policy.update(reward_n[i])
            # render all agent views
            if verbose:
                env.render()

            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
            return scores
