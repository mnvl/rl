
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

    sample_frames = 1024

    epsilon = 0.2
    beta = 0.0
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
            self.net.parameters(), maximize=True, lr=Settings.lr, weight_decay=0.0)

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
        return action.cpu(), probs.cpu()

    def sample_frames(self, render):
        frames = []

        for i in range(Settings.sample_frames):
            if self.done:
                self.observation = self.prepare(self.env.reset())
                self.done = False
                self.episodes_seen += 1

            action, prob = self.select_action()
            new_observation, reward, self.done, _ = self.env.step(int(action))

            frames.append((self.observation, action, prob, reward, self.done))
            self.observation = self.prepare(new_observation)

            self.episode_rewards += reward
            if self.done:
                self.last_episode_rewards = self.episode_rewards
                self.episode_rewards = 0

            self.frames_seen += 1

            if render: self.render_frame()

        return frames

    def optimize(self, frames):
        observations = []
        rewards = []
        actions = []
        pi_old = []

        for observation, action, pi, reward, done in frames:
            observations.append(torch.Tensor(np.expand_dims(observation, 0)))
            actions.append(action)
            pi_old.append(torch.Tensor(np.expand_dims(pi, 0)))

        discounted_reward = 0.0
        for observation, action, prob, reward, done in reversed(frames):
            if done:
                discounted_reward = 0.0
            discounted_reward = Settings.gamma * discounted_reward + reward
            rewards.append(discounted_reward)
        rewards = list(reversed(rewards))

        observations = torch.cat(observations, axis=0).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        pi_old = torch.cat(pi_old, axis=0).to(self.device)

        N = observations.shape[0]

        self.optimizer.zero_grad()

        scores, V = self.net(observations)

        pi = torch.softmax(scores/Settings.temp, axis=1)

        rate = pi[range(N), actions] / pi_old[range(N), actions]
        clipped_rate = torch.clip(rate, 1.0 - Settings.epsilon, 1.0 + Settings.epsilon)

        adv = rewards - V.detach()

        loss_clip = torch.mean(torch.min(rate * adv, clipped_rate * adv))

        log_pi = torch.log_softmax(scores/Settings.temp, axis=1)
        loss_kl = torch.mean(F.kl_div(pi_old, log_pi, log_target=True))

        loss_v = torch.mean(torch.square(rewards - V))

        loss = loss_clip - Settings.beta * loss_kl - Settings.c_v * loss_v

        loss.backward()
        self.optimizer.step()

        return float(loss_clip), float(loss_kl), float(loss_v), float(loss)

    def train(self, render=False):
        frames = self.sample_frames(render)

        loss_clip, loss_kl, loss_v, loss = self.optimize(frames)

        self.writer.add_scalar("loss/clip", loss_clip, self.step)
        self.writer.add_scalar("loss/kl", loss_kl, self.step)
        self.writer.add_scalar("loss/v", loss_v, self.step)
        self.writer.add_scalar("loss", loss_v, self.step)
        self.writer.add_scalar("rewards", self.last_episode_rewards, self.step)

        self.step += 1

        return self.last_episode_rewards, loss_clip, loss_kl, loss_v, loss


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

        for i in range(20):
            reward, loss_clip, loss_kl, loss_v, loss = trainer.train()
            if i % 10 == 9:
                print("mars rover", trainer.frames_seen,
                      trainer.episodes_seen, reward, loss_clip, loss_kl, loss_v)

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
        Settings.c_v = 0.001

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

        n = 100
        for i in range(n):
            magic = (i == n - 1)
            reward, loss_clip, loss_kl, loss_v, loss = trainer.train(render=magic)

            if i % 10 == 0 or magic:
                print("cart pole", trainer.frames_seen,
                      trainer.episodes_seen, reward, loss_clip, loss_kl, loss_v)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(reward, 400)


if __name__ == '__main__':
    unittest.main()
