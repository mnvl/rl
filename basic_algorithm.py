
import numpy as np
import skvideo.io
from torch.utils.tensorboard import SummaryWriter

class BasicAlgorithm:
    def __init__(self):
        self.rendered_frames = []
        self.writer = SummaryWriter()

    def render_frame(self):
        image = self.env.render(mode="rgb_array")
        image = image.copy()
        image = np.expand_dims(image, axis=0)
        self.rendered_frames.append(image)

    def write_video(self, episode=None, filename=None):
        if episode is not None:
            filename = "episode_%06d.mp4" % episode

        images = np.concatenate(self.rendered_frames, axis=0)
        images = (images * 255).astype(np.uint8)

        skvideo.io.vwrite(filename, images)

        print("wrote %s from %s" % (filename, str(images.shape)))

        self.rendered_frames = []


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
