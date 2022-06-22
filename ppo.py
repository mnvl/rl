
import time
import os
import unittest

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

from basic_algorithm import BasicActor, BasicAlgorithm, MarsRoverEnv


class Settings:
    lr = 0.001
    gamma = 0.99

    sample_frames = 128
    num_actors = 32

    epsilon = 0.2
    beta = 0.0
    c_value = 0.1
    c_entropy = 0.01


class Actors:
    def __init__(self, env_fn, prepare_fn, device, net):
        BasicActor.__init__(self)

        self.envs = [env_fn() for j in range(Settings.num_actors)]
        self.prepare_fn = prepare_fn
        self.device = device
        self.net = net

        self.observations = [None for j in range(Settings.num_actors)]
        self.done = [True for j in range(Settings.num_actors)]

        self.frames_seen = 0.0
        self.episodes_seen = 0
        self.last_episode_rewards = [0.0 for j in range(Settings.num_actors)]
        self.episode_rewards = [0.0 for j in range(Settings.num_actors)]

    def select_actions(self):
        s = [np.expand_dims(observation, 0) for observation in self.observations]
        s = np.concatenate(s, axis=0)
        with torch.no_grad():
            scores, V = self.net(torch.tensor(s, device=self.device))
        probs = torch.softmax(scores, axis=1)
        distr = D.Categorical(probs=probs)
        actions = distr.sample()
        return actions.cpu(), probs.cpu()

    def sample_frames(self, render=False):
        frames = [[]for j in range(Settings.num_actors)]

        for i in range(Settings.sample_frames):
            for j in range(Settings.num_actors):
                if self.done[j]:
                    self.observations[j] = self.prepare_fn(self.envs[j].reset())
                    self.done[j] = False
                    self.episodes_seen += 1

                    self.last_episode_rewards[j] = self.episode_rewards[j]
                    self.episode_rewards[j] = 0.0

                    self.episodes_seen += 1

            actions, probs = self.select_actions()

            for j in range(Settings.num_actors):
                new_observation, reward, self.done[j], _ = self.envs[j].step(int(actions[j]))
                self.frames_seen += 1

                frames[j].append((self.observations[j], actions[j], probs[j], reward, self.done[j]))
                self.observations[j] = self.prepare_fn(new_observation)

                self.episode_rewards[j] += reward

        sampled_frames = []

        for j in range(Settings.num_actors):
            discounted_reward = 0.0
            for i in range(Settings.sample_frames-1, -1, -1):
                observation, action, prob, reward, done = frames[j][i]
                if done:
                    discounted_reward = 0.0
                discounted_reward = Settings.gamma * discounted_reward + reward
                sampled_frames.append((observation, action, prob, discounted_reward, done))

        return sampled_frames


class PPO(BasicAlgorithm):
    def __init__(self, env_fn, net, device="cpu", prepare_fn=lambda x: x, first_step=0):
        BasicAlgorithm.__init__(self)

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.prepare_fn = prepare_fn

        self.actors = Actors(env_fn, prepare_fn, self.device, self.net)

        self.optimizer = optim.Adam(
            self.net.parameters(), maximize=True, lr=Settings.lr, weight_decay=0.0)

        self.step = first_step

    def optimize(self, frames):
        observations = []
        rewards = []
        actions = []
        pi_old = []

        for observation, action, pi, reward, done in frames:
            observations.append(torch.Tensor(np.expand_dims(observation, 0)))
            actions.append(action)
            rewards.append(reward)
            pi_old.append(torch.Tensor(np.expand_dims(pi, 0)))

        observations = torch.cat(observations, axis=0).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        pi_old = torch.cat(pi_old, axis=0).to(self.device)

        N = observations.shape[0]

        self.optimizer.zero_grad()

        scores, V = self.net(observations)

        pi = torch.softmax(scores, axis=1)

        rate = pi[range(N), actions] / pi_old[range(N), actions]
        clipped_rate = torch.clip(
            rate, 1.0 - Settings.epsilon, 1.0 + Settings.epsilon)

        adv = rewards - V.detach()

        loss_clip = torch.mean(torch.min(rate * adv, clipped_rate * adv))

        log_pi = torch.log_softmax(scores, axis=1)
        loss_kl = torch.mean(F.kl_div(pi_old, log_pi, log_target=True))

        loss_value = torch.mean(torch.square(rewards - V))

        loss_entropy = -torch.mean(pi * log_pi)

        loss = loss_clip - \
            Settings.beta * loss_kl - \
            Settings.c_value * loss_value + \
            Settings.c_entropy * loss_entropy

        loss.backward()
        self.optimizer.step()

        self.writer.add_histogram("pi", pi.reshape(-1))
        self.writer.add_histogram("adv", adv.reshape(-1))

        self.writer.add_scalar("loss/clip", loss_clip, self.step)
        self.writer.add_scalar("loss/kl", loss_kl, self.step)
        self.writer.add_scalar("loss/value", loss_value, self.step)
        self.writer.add_scalar("loss/entropy", loss_entropy, self.step)
        self.writer.add_scalar("loss", loss, self.step)
        self.writer.add_scalar("rewards", self.last_episode_rewards, self.step)

        return self.last_episode_rewards, float(loss)

    def train(self, render=False):
        t1 = time.time()
        frames = self.actors.sample_frames(render)
        self.frames_seen = self.actors.frames_seen
        self.episodes_seen = self.actors.episodes_seen
        self.last_episode_rewards = np.mean(self.actors.last_episode_rewards)

        t2 = time.time()
        rewards, loss = self.optimize(frames)

        t3 = time.time()

        if self.step % 10 == 0:
            print("t_sample = %.3f, t_optimize = %.3f" % (t2 - t1, t3 - t2))

        self.step += 1

        return rewards, loss

    def write_video(self, episode=None, filename=None):
        #self.actors.write_video(episode, filename)
        pass


class TestPPO(unittest.TestCase):
    def setUp(self):
        self.save = (
            Settings.lr,
            Settings.gamma,
            Settings.sample_frames)

    def tearDown(self):
        (Settings.lr,
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

        def env(): return MarsRoverEnv()

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
            rewards, loss = trainer.train()
            if i % 10 == 9:
                print("mars rover", trainer.frames_seen,
                      trainer.episodes_seen, rewards, loss)

        assert rewards == 10.0, str(reward)

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
        Settings.c_value = 0.001

        def env(): return gym.make("CartPole-v1")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 64)
                self.linear2 = nn.Linear(64, 64)
                self.linear3 = nn.Linear(64, env().action_space.n)
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
            rewards, loss = trainer.train(
                render=magic)

            if i % 10 == 0 or magic:
                print("cart pole", trainer.frames_seen,
                      trainer.episodes_seen, rewards, loss)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(rewards, 400)


if __name__ == '__main__':
    unittest.main()
