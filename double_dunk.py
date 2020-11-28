"""
double_dunk.py
Purpose: Implement an interface to interact with the double dunk environment
"""

import gym
import csv
import numpy as np
import time

TEST_INDEX = 10   # test after every 1000 training episodes
NUM_TESTS = 10

class DoubleDunk:
    def __init__(self, episodes, agent_func, verbose):
        env = gym.make('DoubleDunk-v0')
        self.verbose = verbose
        agent = agent_func(env, 0.1)
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
                        value = self.run(env, agent, is_test=True)
                        scores.append(value)
                    print(f'TEST %d:\t Avg Reward = %.3f\tTime=%.5f' %(int(i / TEST_INDEX), np.average(scores), (time.time() - test_time)))
                    agent.losses = []
                    test_time = time.time()
                    writer.writerow([i / TEST_INDEX, np.average(scores)])
                else:
                    self.run(env, agent, is_test=False)
                    agent.learn()
                ep_times.append(time.time() - episode_time)
                #print(f'ep: {i} took {time.time() - episode_time}s')

        env.close()

    def run(self, env, agent, is_test=False):
        # state is an array of 4 floats [position, velocity, pole angle, pole angular velocity]
        current_state = env.reset()

        done = False
        step_count = 0
        score = 0
        #Loop until either the agent finishes or takes 200 actions, whichever comes first.
        while step_count < 200 and done == False:
            step_count += 1

            action = agent.get_action(env, current_state)

            #Execute actions using the step function. Returns the next_state, reward, a boolean indicating whether this is a terminal state. The final thing it returns is a probability associated with the underlying transition distribution, but we shouldn't need that for this assignment.
            next_state, reward, done, _ = env.step(action)

            score += reward

            if not is_test:
                agent.update(reward)

            if is_test and self.verbose:
                #Render visualizes the environment
                env.render()

            current_state = next_state
        return score
