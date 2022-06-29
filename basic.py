
import numpy as np

class MarsRoverEnv:
    class action_space:
        n = 2

    def __init__(self, num_states=5):
        self.num_states = num_states
        self.reset()

    def build_observation(self):
        observation = np.zeros(shape=self.num_states, dtype=np.float32)
        observation[self.state] = 1.0
        return observation

    def reset(self):
        self.state = 1
        return self.build_observation()

    def step(self, action):
        if self.state > 0 and self.state < self.num_states-1:
            self.state = min(max(self.state+action*2-1, 0), self.num_states-1)

        reward = 0
        done = False

        if self.state == 0:
            reward = 1
            done = True

        if self.state == self.num_states - 1:
            reward = 10
            done = True

        return self.build_observation(), reward, done, {}
