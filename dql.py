
import os
import random
import unittest
import copy

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import skvideo.io


class Settings:
    lr = 0.0001
    batch_size = 256
    gamma = 0.9
    update_target_every_episodes = 25
    epsilon1 = 1.0
    epsilon2 = 0.05
    epsilon_decay = 10000
    steps_per_batch = 256
    replay_memory_size = 100000


class DQL:
    def __init__(self, env, net, target_net=None, episode=0, device="cpu", prepare=lambda x: x):
        if target_net is None:
            target_net = copy.deepcopy(net)

        self.env = env
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.prepare = prepare

        self.replay_memory = []
        self.done = False
        self.episode = episode
        self.frame = 0

        self.render_frames = []

        self.optimizer = optim.Adam(
            net.parameters(), lr=Settings.lr, weight_decay=0.0)

        self.writer = SummaryWriter()

    def select_action(self):
        if random.random() < self.epsilon:
            return int(random.randint(0, self.env.action_space.n-1))

        X = np.expand_dims(self.observation, 0)
        with torch.no_grad():
            Q = self.net(torch.tensor(X, device=self.device))
        return int(np.argmax(Q.cpu().numpy()[0]))

    def render_frame(self):
        image = self.env.render(mode="rgb_array")
        image = image.copy()
        image = np.expand_dims(image, axis=0)
        self.render_frames.append(image)

    def step(self):
        action = self.select_action()

        new_observation, reward, self.done, info = self.env.step(action)
        new_observation = self.prepare(new_observation)

        memory = (
            self.observation,
            int(action),
            float(reward),
            new_observation,
            self.done)

        if len(self.replay_memory) >= Settings.replay_memory_size:
            self.replay_memory = self.replay_memory[1:]
        self.replay_memory.append(memory)
        self.observation = new_observation

        if self.frame % 1000 == 0:
            self.writer.add_image("step_observation", make_grid(
                torch.tensor(np.expand_dims(self.observation, 1))), self.episode)
            self.writer.add_histogram(
                "step_observation_hist", self.observation.reshape(-1), self.episode)

        self.frame += 1

        return reward

    def optimize(self):
        batch = [random.choice(self.replay_memory)
                 for i in range(Settings.batch_size)]

        Xi = []
        Xj = []
        for sample in batch:
            observation, action, reward, new_observation, done = sample
            Xi.append(np.expand_dims(observation, 0))
            Xj.append(np.expand_dims(new_observation, 0))

        Xj = np.concatenate(Xj, axis=0)
        with torch.no_grad():
            Qj = self.target_net(torch.tensor(Xj, device=self.device))
        self.writer.add_scalar("mean_Qj", Qj.mean(), self.episode)
        self.writer.add_histogram("Qj", Qj.reshape(-1), self.episode)

        self.optimizer.zero_grad()

        Xi = np.concatenate(Xi, axis=0)
        Qi = self.net(torch.tensor(Xi, device=self.device))
        self.writer.add_scalar("mean_Qi", Qi.mean(), self.episode)
        self.writer.add_histogram("Qi", Qi.reshape(-1), self.episode)

        actions = []
        y = []

        for sample, qj in zip(batch, Qj):
            observation, action, reward, new_observation, done = sample

            actions.append(action)

            if done:
                y.append(reward)
            else:
                best = float(qj.cpu().max())
                y.append(reward + Settings.gamma * best)

        y = torch.tensor(y, device=self.device)

        loss = torch.square(Qi[range(Qi.shape[0]), actions] - y)
        loss = torch.mean(loss)

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu())

    def train(self, render=False, steps_per_batch=Settings.steps_per_batch):
        self.observation = self.prepare(self.env.reset())
        self.done = False

        rewards = 0.0
        losses = []

        alpha = min(self.episode / Settings.epsilon_decay, 1.0)
        self.epsilon = Settings.epsilon1 + alpha * \
            (Settings.epsilon2-Settings.epsilon1)

        if self.episode % Settings.update_target_every_episodes == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        while not self.done:
            for i in range(steps_per_batch):
                rewards += self.step()
                if self.done:
                    break
                if render:
                    self.render_frame()

            losses.append(self.optimize())

        loss = np.mean(losses)

        self.writer.add_scalar("loss", loss, self.episode)
        self.writer.add_scalar("reward", rewards, self.episode)

        self.episode += 1

        return rewards, loss

    def write_video(self, episode=None, filename=None):
        if episode is not None:
            filename = "episode_%06d.mp4" % episode

        images = np.concatenate(self.render_frames, axis=0)
        images = (images * 255).astype(np.uint8)

        skvideo.io.vwrite(filename, images)

        print("wrote %s from %s" % (filename, str(images.shape)))

        self.render_frames = []


class MockEnv:
    class action_space:
        n = 3

    def __init__(self, randomized):
        self.randomized = randomized
        self.reset()

    def reset(self):
        self.state = 0.0
        self.done = False
        return np.array([np.random.normal(), 0.0]).astype(np.float32)

    def step(self, action):
        assert action >= 0 and action < 3

        if action == 0 and not self.done:
            self.state += 100

        if action == 1 and not self.done:
            if self.randomized:
                self.state += random.randint(1, 10)
            else:
                self.state += 10

        if action == 2:
            self.done = True

        if self.state > 21 or self.state < 0:
            self.done = True

        reward = 0.0
        if self.done and self.state > 0 and self.state <= 21:
            reward = self.state

        observation = np.array(
            [self.state, self.done]).astype(np.float32)

        return observation, reward, self.done, None


class TestDQL(unittest.TestCase):
    def setUp(self):
        self.save = (
            Settings.lr,
            Settings.gamma,
            Settings.update_target_every_episodes,
            Settings.epsilon1,
            Settings.epsilon2,
            Settings.epsilon_decay)

        Settings.update_target_every_episodes = 100
        Settings.epsilon_decay = 100

    def tearDown(self):
        (Settings.lr,
         Settings.gamma,
         Settings.update_target_every_episodes,
         Settings.epsilon1,
         Settings.epsilon2,
         Settings.epsilon_decay) = self.save

    def test_select_action(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 3)

            def forward(self, x):
                x = self.linear(x)
                x[0, 0] = 0
                x[0, 1] = 200
                x[0, 2] = 0
                return x

        trainer = DQL(MockEnv(randomized=False), Net())
        trainer.observation = trainer.env.reset()
        trainer.epsilon = 0.0
        self.assertEqual(trainer.select_action(), 1)

    def test_deterministic(self):
        Settings.lr = 0.005

        env = MockEnv(randomized=False)
        net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 3))

        trainer = DQL(env, net)

        for i in range(2000):
            rewards, loss = trainer.train()
            if i % 100 == 0:
                print("deterministic", rewards, loss)

        X = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]).type(
            torch.float32)
        y = net(X)

        print(y)

        # take 10 if we have 0
        self.assertEqual(torch.argmax(y[0, :]), 1)

        # take 10 if we have 10
        self.assertEqual(torch.argmax(y[1, :]), 1)

        # finish if we have 20
        self.assertEqual(torch.argmax(y[2, :]), 2)

        trainer.observation = env.reset()
        trainer.epsilon = 0.0
        self.assertEqual(trainer.select_action(), 1)

    def test_overfit(self):
        Settings.lr = 0.005

        env = MockEnv(randomized=False)
        net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 3))

        trainer = DQL(env, net)

        for i in range(10000):
            rewards, loss = trainer.train()
            if i % 100 == 0:
                print("overfit", rewards, loss)

        X = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]).type(
            torch.float32)
        y = net(X)

        print(y)

        # take 10 if we have 0
        self.assertEqual(torch.argmax(y[0, :]), 1)

        # take 10 if we have 10
        self.assertEqual(torch.argmax(y[1, :]), 1)

        # finish if we have 20
        self.assertEqual(torch.argmax(y[2, :]), 2)

        trainer.observation = env.reset()
        trainer.epsilon = 0.0
        self.assertEqual(trainer.select_action(), 1)

    def test_randomized(self):
        Settings.lr = 0.005

        env = MockEnv(randomized=True)
        net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 3))

        trainer = DQL(env, net)

        for i in range(5000):
            rewards, loss = trainer.train()
            if i % 100 == 0:
                print("randomized", rewards, loss)

        X = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [12.0, 0.0], [18.0, 0.0], [20.0, 0.0]]).type(
            torch.float32)
        y = net(X)

        print(y)

        # take if we have 0
        self.assertEqual(torch.argmax(y[0, :]), 1)

        # take if we have 5
        self.assertEqual(torch.argmax(y[1, :]), 1)

        # take if we have 10
        self.assertEqual(torch.argmax(y[2, :]), 1)

        # take if we have 12
        self.assertEqual(torch.argmax(y[3, :]), 1)

        # finish if we have 18
        self.assertEqual(torch.argmax(y[4, :]), 2)

        # finish if we have 20
        self.assertEqual(torch.argmax(y[5, :]), 2)

        trainer.observation = env.reset()
        trainer.epsilon = 0.0
        self.assertEqual(trainer.select_action(), 1)

    def test_cartpole(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        Settings.lr = 0.001

        env = gym.make("CartPole-v1")

        net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n))

        trainer = DQL(env, net, copy.deepcopy(net))

        num_episodes = 2000
        for i in range(num_episodes):
            magic = (i > num_episodes - 5)
            reward, loss = trainer.train(render=magic)

            if i % 100 == 0 or magic:
                print("cartpole", i, reward, loss)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(reward, 200)


if __name__ == '__main__':
    unittest.main()
