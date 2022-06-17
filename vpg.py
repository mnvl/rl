
import os
import unittest

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

from basic_algorithm import BasicAlgorithm, MarsRoverEnv

class Settings:
    lr_pi = 0.001
    lr_v = 0.001
    temp = 1.0
    gamma = 0.99
    lagrange = 0.1
    sample_episodes = 10


class VPG(BasicAlgorithm):
    def __init__(self, env, pi_net, v_net, episode=0, device="cpu", prepare=lambda x: x):
        BasicAlgorithm.__init__(self)

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

    def select_action(self, observation):
        s = np.expand_dims(observation, 0)
        with torch.no_grad():
            scores = self.pi_net(torch.tensor(s, device=self.device))
        distr = D.Categorical(probs=torch.softmax(
            scores[0] / Settings.temp, axis=0))
        action = distr.sample()
        return action

    def sample_episode(self, render):
        episode = []

        observation = self.env.reset()
        done = False

        while not done:
            action = self.select_action(observation)

            new_observation, reward, done, _ = self.env.step(int(action))

            episode.append((observation, action, reward, done))
            observation = new_observation

            if render:
                self.render_frame()

        return episode

    def optimize(self, episode):
        S = []
        R = []
        A = []

        for observation, action, reward, done in episode:
            observation = np.expand_dims(observation, 0)
            S.append(observation)
            A.append(action)

        G = 0.0
        for observation, action, reward, done in reversed(episode):
            if done:
                G = 0.0
            G = Settings.gamma * G + reward
            R.append(G)
        R = list(reversed(R))

        S = np.concatenate(S, axis=0)
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
        for i in range(Settings.sample_episodes):
            episode = self.sample_episode(render)
            episodes.extend(episode)

        pi_loss, v_loss = self.optimize(episodes)

        rewards = sum([reward for observation, action, reward,
                      done in episodes]) / Settings.sample_episodes

        self.writer.add_scalar("pi_loss", pi_loss, self.episode)
        self.writer.add_scalar("v_loss", v_loss, self.episode)
        self.writer.add_scalar("reward", rewards, self.episode)

        self.episode += 1

        return rewards, pi_loss, v_loss


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
                self.dummy = nn.Linear(5, 2)

            def forward(self, x):
                x = self.dummy(x)
                x[0, 0] = 0
                x[0, 1] = 200
                return x

        trainer = VPG(MarsRoverEnv(), Net(), Net())
        self.assertEqual(trainer.select_action(trainer.env.reset()), 1)

    def test_mars_rover(self):
        Settings.lr_pi = 0.1
        Settings.lr_v = 0.01

        env = MarsRoverEnv()
        pi_net = nn.Sequential(nn.Linear(5, 2))
        v_net = nn.Sequential(nn.Linear(5, 1))

        trainer = VPG(env, pi_net, v_net)

        for i in range(100):
            reward, pi_loss, v_loss = trainer.train()
            if i % 10 == 0:
                print("mars rover", reward, pi_loss, v_loss)

        assert reward == 10.0

        X = torch.eye(5).type(torch.float32)
        pi = pi_net(X)
        v = v_net(X)

        print(pi)
        self.assertEqual(torch.argmax(pi[1, :]), 1)
        self.assertEqual(torch.argmax(pi[2, :]), 1)
        self.assertEqual(torch.argmax(pi[3, :]), 1)

        print(v)


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

        num_episodes = 1000
        n = num_episodes // Settings.sample_episodes
        for i in range(n):
            magic = (i == n-1)
            reward, pi_loss, v_loss = trainer.train(render=magic)

            if i % 100 == 0 or magic:
                print("cartpole", i, reward, pi_loss, v_loss)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(reward, 200)


if __name__ == '__main__':
    unittest.main()
