#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:04:13 2020
Just a small modification to get running average
@author: subashkhanal
"""


import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import csv
import agents.reinforce as reinforce
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


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
       
        run_avg_window = 1000
        train_ep_rewards = np.array([]) #average scores for each episode
        train_run_avg = np.array([]) #running average for 1000 previous episodes to capture the learning trend
                # create interactive policies for each agent
        policies = [good_agents(env, i, 0.1) for i in range(env.n)]

        # find directory to save results
        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.split(current_directory)[0]
        parent_directory = os.path.split(parent_directory)[0]

        scenario_name = os.path.splitext(scenario_file)[0]
        scores_filename = f'{parent_directory}/results/{scenario_name}_{policies[0].name}_scores.npy'
        run_avg_filename = f'{parent_directory}/results/{scenario_name}_{policies[0].name}_run_avg.npy'
       
        # execution loop
        for i in range(episodes):
            train_ep_reward = self.run(env, policies, False, verbose)
            train_ep_reward = np.array(train_ep_reward).mean()
            train_ep_rewards = np.append(train_ep_rewards,train_ep_reward)
            run_avg_ep_score = np.mean(train_ep_rewards[-run_avg_window:]) # average for last 10000 episodes of training
            train_run_avg = np.append(train_run_avg, run_avg_ep_score)
            print('episode ', i, 'score %.3f' % train_ep_reward,
        'average score %.3f' % run_avg_ep_score)

            for policy in policies:
                policy.update()

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
            #print(act_n)
            obs_n_, reward_n, done_n, _ = env.step(act_n)
            #print(reward_n)
            step_count += 1
            #print(reward_n)
            for i, policy in enumerate(policies):
                scores[i] += reward_n[i]
                if not istest:
                    policy.learn(obs_n[i],reward_n[i],obs_n_[i],done_n[i])
            # render all agent views
            if verbose:
                env.render()

            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
            obs_n = obs_n_
        return scores
