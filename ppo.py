
import os
import unittest

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

from basic_algorithm import BasicAlgorithm, MarsRoverEnv


class Settings:
    lr = 0.001
    temp = 1.0
    gamma = 0.99
    epsilon = 0.2
    beta = 1.0

    sample_frames = 1024

    c_v = 0.1


class PPO(BasicAlgorithm):
    def __init__(self, env, net, device="cpu", prepare=lambda x: x):
        BasicAlgorithm.__init__(self)

        self.env = env
        self.done = True

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.prepare = prepare

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=Settings.lr, weight_decay=0.0)

        self.frames_seen = 0
        self.episodes_seen = 0
        self.step = 0

        self.last_episode_rewards = 0
        self.episode_rewards = 0

    def select_action(self):
        s = np.expand_dims(self.observation, 0)
        with torch.no_grad():
            scores, V = self.net(torch.tensor(s, device=self.device))
        probs = torch.softmax(scores[0] / Settings.temp, axis=0)
        distr = D.Categorical(probs=probs)
        action = distr.sample()
        return action, probs

    def sample_frames(self, render):
        frames = []

        for i in range(Settings.sample_frames):
            if self.done:
                self.observation = self.env.reset()
                self.done = False
                self.episodes_seen += 1

            action, prob = self.select_action()
            new_observation, reward, self.done, _ = self.env.step(int(action))

            frames.append((self.observation, action, prob, reward, self.done))
            self.observation = new_observation

            self.episode_rewards += reward
            if self.done:
                self.last_episode_rewards = self.episode_rewards
                self.episode_rewards = 0

            self.frames_seen += 1

            if render: self.render_frame()

        return frames

    def optimize(self, frames):
        S = []
        R = []
        A = []
        pi_old = []

        for observation, action, pi, reward, done in frames:
            S.append(np.expand_dims(observation, 0))
            A.append(action)
            pi_old.append(np.expand_dims(pi, 0))

        discounted_reward = 0.0
        for observation, action, prob, reward, done in reversed(frames):
            if done:
                discounted_reward = 0.0
            discounted_reward = Settings.gamma * discounted_reward + reward
            R.append(discounted_reward)
        R = list(reversed(R))

        S = np.concatenate(S, axis=0)
        A = np.array(A)
        R = np.array(R)
        pi_old = np.concatenate(pi_old, axis=0)

        S = torch.Tensor(S, device=self.device)
        A = torch.LongTensor(A, device=self.device)
        R = torch.Tensor(R, device=self.device)
        pi_old = torch.Tensor(pi_old, device=self.device)

        self.optimizer.zero_grad()

        scores, V = self.net(S)

        pi = torch.softmax(scores/Settings.temp, axis=1)
        rate = pi[range(S.shape[0]), A] / pi_old[range(S.shape[0]), A]
        loss_cpi = -torch.mean(rate * (R - V.detach()))

        log_pi = torch.log_softmax(scores/Settings.temp, axis=1)
        loss_kl = torch.mean(F.kl_div(pi_old, log_pi, log_target=True))

        loss_v = torch.mean(torch.square(R - V))

        loss = loss_cpi + Settings.beta * loss_kl + Settings.c_v * loss_v

        loss.backward()
        self.optimizer.step()

        return float(loss_cpi), float(loss_kl), float(loss_v)

    def train(self, render=False):
        frames = self.sample_frames(render)

        cpi_loss, kl_loss, v_loss = self.optimize(frames)

        self.writer.add_scalar("cpi_loss", cpi_loss, self.step)
        self.writer.add_scalar("kl_loss", kl_loss, self.step)
        self.writer.add_scalar("v_loss", v_loss, self.step)
        self.writer.add_scalar("reward", self.last_episode_rewards, self.step)

        self.step += 1

        return self.last_episode_rewards, cpi_loss, kl_loss, v_loss


class TestPPO(unittest.TestCase):
    def setUp(self):
        self.save = (
            Settings.lr,
            Settings.temp,
            Settings.gamma,
            Settings.sample_frames)

    def tearDown(self):
        (Settings.lr,
         Settings.temp,
         Settings.gamma,
         Settings.sample_frames) = self.save

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

        trainer = PPO(MarsRoverEnv(), Net(), Net())
        self.assertEqual(trainer.select_action(trainer.env.reset()), 1)

    def test_mars_rover(self):
        Settings.lr = 0.1

        env = MarsRoverEnv()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 2)
                self.linear2 = nn.Linear(5, 1)

            def forward(self, x):
                return self.linear1(x), self.linear2(x)

        net = Net()

        trainer = PPO(env, net)

        for i in range(100):
            reward, cpi_loss, kl_loss, v_loss = trainer.train()
            if i % 10 == 9:
                print("mars rover", trainer.frames_seen,
                      trainer.episodes_seen, reward, cpi_loss, kl_loss, v_loss)

        assert reward == 10.0, str(reward)

        X = torch.eye(5).type(torch.float32)
        pi, V = net(X)

        print(pi)
        self.assertEqual(torch.argmax(pi[1, :]), 1)
        self.assertEqual(torch.argmax(pi[2, :]), 1)
        self.assertEqual(torch.argmax(pi[3, :]), 1)

        print(V)

    def test_cartpole(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        Settings.lr = 0.01

        env = gym.make("CartPole-v1")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 64)
                self.linear2 = nn.Linear(64, 64)
                self.linear3 = nn.Linear(64, env.action_space.n)
                self.linear4 = nn.Linear(64, 1)

            def forward(self, x):
                x = F.relu(self.linear1(x))
                x = F.relu(self.linear2(x))
                return self.linear3(x), self.linear4(x)

        net = Net()

        trainer = PPO(env, net)

        n = 200
        for i in range(n):
            magic = (i == n - 1)
            reward, cpi_loss, kl_loss, v_loss = trainer.train(render=magic)

            if i % 10 == 0 or magic:
                print("cart pole", trainer.frames_seen,
                      trainer.episodes_seen, reward, cpi_loss, kl_loss, v_loss)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(reward, 400)


if __name__ == '__main__':
    unittest.main()
