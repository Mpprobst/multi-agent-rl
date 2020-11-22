"""
double_dunk.py
Purpose: Implement an interface to interact with the double dunk environment
"""

import gym
import csv
import numpy as np

TEST_INDEX = 100   # test after every 1000 training episodes
NUM_TESTS = 10

class DoubleDunk:
    def __init__(self, episodes, agent_func, verbose):
        env = gym.make('DoubleDunk-v0')
        print(f'num actions: {env.action_space.n}')
        self.verbose = verbose
        agent = agent_func(env, 0.1)
        filename = f'results/dd_{agent.name}.csv'
        with open(filename, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            for i in range(1, episodes):
                if i % TEST_INDEX == 0:
                    scores = []
                    for t in range(NUM_TESTS):
                        value = self.run(env, agent, is_test=True)
                        scores.append(value)
                    print(f'TEST {i / TEST_INDEX}:\t Avg Reward = {np.average(scores)}')
                    agent.losses = []
                    writer.writerow([i / TEST_INDEX, np.average(scores)])
                else:
                    self.run(env, agent, is_test=False)
                    agent.learn()

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

            action = 0
            if is_test:
                action = agent.best_action(current_state)
            else:
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
