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
import agents.multiactor_singlecritic as MASCAgent
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

        run_avg_window = 10
        train_ep_rewards = np.array([]) #average scores for each episode
        train_run_avg = np.array([]) #running average for 1000 previous episodes to capture the learning trend
                # create interactive policies for each agent
        policies = [good_agents(env, i, 0.01) for i in range(env.n)]
        # create MASC agent if needed
        if isinstance(policies[0], MASCAgent.MASCAgent):
            critic = MASCAgent.Critic(env, 0.01)
            for policy in policies:
                policy.critic = critic
        # find directory to save results
        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.split(current_directory)[0]
        parent_directory = os.path.split(parent_directory)[0]

        scenario_name = os.path.splitext(scenario_file)[0]
        scores_filename = f'{parent_directory}/results/{scenario_name}_{policies[0].name}_scores.npy'
        run_avg_filename = f'{parent_directory}/results/{scenario_name}_{policies[0].name}_run_avg.npy'

        # execution loop
        for i in range(episodes+1):
            istest = False
            if i % run_avg_window == 0:
                istest = True
            train_ep_reward = self.run(env, policies, istest, verbose)
            train_ep_reward = np.mean(np.array(train_ep_reward)) #average score per episode : it is averaged across the scores for each agent
            train_ep_rewards = np.append(train_ep_rewards,train_ep_reward) #store that average score per episode

            run_avg_ep_score = np.mean(train_ep_rewards[-run_avg_window:]) # Running average for the last 10000 episodes of training
            train_run_avg = np.append(train_run_avg, run_avg_ep_score)
            if i % run_avg_window == 0:

                print('episode ',i, 'score %.3f' % train_ep_reward,
            'Running average score %.3f' % run_avg_ep_score)

            for policy in policies:
                policy.update()
        np.save(scores_filename,train_ep_rewards)
        np.save(run_avg_filename,train_run_avg)

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
            if verbose and istest:
                env.render()

            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
            obs_n = obs_n_
        return scores
