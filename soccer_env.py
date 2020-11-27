"""
soccer.py
Author: Michael Probst
Purpose: Imlements the soccer environment
"""

#import gym
import csv
import numpy as np
import time
import marl
from marl import MARL
from soccer import DiscreteSoccerEnv

TEST_INDEX = 10   # test after every 10 training episodes
NUM_TESTS = 10

class SoccerEnv:
    def __init__(self, episodes, agent_funcs1, agent_funcs2, verbose):
        env = DiscreteSoccerEnv(nb_pl_team1=len(agent_funcs1), nb_pl_team2=len(agent_funcs2))
        print(f'num actions: {env.action_space.n}')
        self.verbose = verbose
        agents = []

        for agent_func in agent_funcs1:
            agent = agent_func(observation_space=env.observation_space,
                               action_space=env.action_space, lr=0.1)
            agents.append(agent)

        for agent_func in agent_funcs2:
            agent = agent_func(observation_space=env.observation_space,
                               action_space=env.action_space, lr=0.1)
            agents.append(agent)

        mas = MARL(agents_list=agents)

        for agent in agents:
            agent.set_mas(mas)

        filename = f'results/dd_{agent.name}.csv'
        with open(filename, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            ep_times = []
            test_time = time.time()
            for i in range(1, episodes):
                episode_time = time.time()
                if i % TEST_INDEX == 0:
                    scores = []
                    for t in range(NUM_TESTS):
                        value = self.run(mas, env, is_test=True)
                        value = value['mean_by_episode'][0][0]
                        scores.append(value)

                    avgs = np.mean(scores, axis=0)
                    rewardString = ["%.3f" % avg for avg in avgs]
                    print(f'TEST  %d:\t Avg Rewards = %s\tTime=%.5f' %(int(i / TEST_INDEX), rewardString, (time.time() - test_time)))
                    agent.losses = []
                    test_time = time.time()
                    writer.writerow([i / TEST_INDEX, np.average(scores)])
                else:
                    self.run(mas, env, is_test=False)
                ep_times.append(time.time() - episode_time)
        env.close()

    def run(self, mas, env, is_test=False):
        score = 0
        if not is_test:
            mas.learn(env, nb_timesteps=1, render=self.verbose)

        if is_test:
            score = mas.test(env, nb_episodes=1, time_laps=0.25, render=self.verbose)
            if (self.verbose):
                env.render()

        return score
