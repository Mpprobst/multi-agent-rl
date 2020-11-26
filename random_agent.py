"""
random.py
Purpose: Implements an agent that makes random decisions
"""

import marl
from marl.agent import TrainableAgent
from marl.agent import MATrainable

from marl.tools import ClassSpec, _std_repr, is_done, reset_logging
import marl.policy.policies as policy
from marl.exploration import ExplorationProcess
from marl.experience import ReplayMemory, PrioritizedReplayMemory

class RandomAgent(TrainableAgent, MATrainable):
    def __init__(self, env, lr, index=None, mas=None):
        TrainableAgent.__init__(self, policy=policy.RandomPolicy(env.action_space), observation_space=env.observation_space, action_space=env.action_space, gamma=0.98, lr=lr, name="RandomAgent", )
        MATrainable.__init__(self, mas, index)

        self.istest = False
    """
    def action(self, state):
        return 0

    def greedy_action(self, state):
        return 0

    def reset(self):
        pass

    def worst_rew(self):
        return -100

    def get_best_rew(self, rew1, rew2):
        return rew2 if rew1 < rew2 else rew1

    def test(self, env, nb_episodes=1, max_num_step=200, render=True, time_laps=0.):
        # we are just defining this to make marl happy
        self.istest = True
        print("random is testing")

        reward = self.run(env, nb_episodes, max_num_step, render, time_laps)
        return reward

    def learn(self, env, nb_timesteps, max_num_step=100, test_freq=1000, save_freq=1000, save_folder="models", render=False, time_laps=0., verbose=0, timestep_init=0, log_file=None):
        self.istest = False
        print("random is learning")
        reward = self.run(env, nb_timesteps, max_num_step, render, time_laps)
        # now do the learning
        return reward

    def run(self, env, nb_episodes, max_num_step=200, render=True, time_laps=0.5):
        current_state = env.reset()
        print("random is running")

        done = False
        step_count = 0
        score = 0
        #Loop until either the agent finishes or takes 200 actions, whichever comes first.
        while step_count < 200 and done == False:
            step_count += 1

            action = self.get_action(current_state)

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

    def __repr__(self):
        return _std_repr(self)

    def set_mas(self, mas):
        self.mas = mas
        for ind, ag in enumerate(self.mas.agents):
            if ag.id == self.id:
                self.index = ind
    """

    def update_model(self, t):
        """
        Update the model.
        """
        return 0
