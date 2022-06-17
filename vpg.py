
import os
import random
import unittest
import copy

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import skvideo.io


class Settings:
    lr_pi = 0.001
    lr_v = 0.001
    temp = 1.0
    gamma = 0.99
    lagrange = 0.1


class VPG:
    def __init__(self, env, pi_net, v_net, episode=0, device="cpu", prepare=lambda x: x):
        self.env = env
        self.device = torch.device(device)
        self.pi_net = pi_net.to(self.device)
        self.v_net = v_net.to(self.device)
        self.prepare = prepare

        self.optimizer_pi = optim.Adam(
            pi_net.parameters(), lr=Settings.lr_pi, weight_decay=0.0)
        self.optimizer_v = optim.Adam(
            v_net.parameters(), lr=Settings.lr_v, weight_decay=0.0)

        self.episode = episode

        self.writer = SummaryWriter()

    def select_action(self, observation):
        s = np.expand_dims(observation, 0)
        with torch.no_grad():
            scores = self.pi_net(torch.tensor(s, device=self.device))
        distr = D.Categorical(probs = torch.softmax(scores[0] / Settings.temp, axis=0))
        action = distr.sample()
        return action

    def render_frame(self):
        image = self.env.render(mode="rgb_array")
        image = image.copy()
        image = np.expand_dims(image, axis=0)
        self.render_frames.append(image)

    def sample_episode(self):
        episode = []

        observation = self.env.reset()
        done = False

        while not done:
            action = self.select_action(observation)

            new_observation, reward, done, _ = self.env.step(int(action))

            episode.append((observation, action, reward, done))
            new_observation = observation

        return episode

    def optimize(self, episode):
        S = []
        R = []
        A = []

        for observation, action, reward, done in episode:
            observation = np.expand_dims(observation, 0)
            S.append(observation)
            A.append(action)

        G = 0
        for observation, action, reward, done in reversed(episode):
            if done: G = 0.0
            G = Settings.gamma * G + reward
            R.append(G)
        R = list(reversed(R))

        S = np.concatenate(S, axis = 0)
        A = np.array(A)
        R = np.array(R)

        S = torch.Tensor(S, device=self.device)
        A = torch.LongTensor(A, device=self.device)
        R = torch.Tensor(R, device=self.device)

        with torch.no_grad():
            V = self.v_net(S)

        self.optimizer_pi.zero_grad()
        scores = self.pi_net(S)
        log_pi = torch.log_softmax(scores, axis=1)
        log_pi = log_pi[range(S.shape[0]), A]
        loss = -torch.mean(log_pi * (R - V))
        loss.backward()
        self.optimizer_pi.step()
        pi_loss = loss.detach()

        for i in range(20):
            self.optimizer_v.zero_grad()
            V = self.v_net(S)
            loss = torch.mean(torch.square(V - R))
            loss.backward()
            self.optimizer_v.step()
        v_loss = loss.detach()

        return float(pi_loss), float(v_loss)

    def train(self, render=False):
        episodes = []
        for i in range(10):
            episode = self.sample_episode()
            episodes.extend(episode)

        pi_loss, v_loss = self.optimize(episodes)

        rewards = sum([reward for observation, action, reward, done in episodes]) / 100.0

        self.writer.add_scalar("pi_loss", pi_loss, self.episode)
        self.writer.add_scalar("v_loss", v_loss, self.episode)
        self.writer.add_scalar("reward", rewards, self.episode)

        self.episode += 1

        return rewards, pi_loss, v_loss

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
        return np.array([0.0, 0.0]).astype(np.float32)

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


class TestVPG(unittest.TestCase):
    def setUp(self):
        self.save = (
            Settings.lr_pi,
            Settings.lr_v,
            Settings.temp,
            Settings.gamma)

    def tearDown(self):
        (Settings.lr_pi,
         Settings.lr_v,
         Settings.temp,
         Settings.gamma) = self.save

    def test_select_action(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = self.linear(x)
                x[0, 0] = 0
                x[0, 1] = 200
                x[0, 2] = 0
                return 1.0, x

        trainer = VPG(MockEnv(randomized=False), Net())
        self.assertEqual(trainer.select_action(trainer.env.reset()), 1)

    def test_deterministic(self):
        Settings.lr_pi = 0.1
        Settings.lr_v = 0.01

        env = MockEnv(randomized=False)
        pi_net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 3))
        v_net = nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 1))

        trainer = VPG(env, pi_net, v_net)

        rewards = []
        pi_losses = []
        v_losses = []
        for i in range(20000):
            reward, pi_loss, v_loss = trainer.train()
            rewards.append(reward)
            pi_losses.append(pi_loss)
            v_losses.append(v_loss)
            if i % 100 == 0:
                print("deterministic", np.mean(rewards), np.mean(pi_losses), np.mean(v_losses))
                rewards = []
                pi_losses = []
                v_losses = []

        X = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]).type(
            torch.float32)
        y = trainer.pi_net(X)
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

        env = gym.make("CartPole-v1")

        pi_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n))
        v_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        trainer = VPG(env, pi_net, v_net)

        num_episodes = 2000
        for i in range(num_episodes):
            magic = (i > num_episodes - 5)
            reward, pi_loss, v_loss = trainer.train(render=magic)

            if i % 100 == 0 or magic:
                print("cartpole", i, reward, pi_loss, v_loss)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(reward, 200)


if __name__ == '__main__':
    unittest.main()
