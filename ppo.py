
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
    temp = 1.0
    gamma = 0.99

    sample_frames = 256
    num_actors = 8

    epsilon = 0.2
    beta = 0.0
    c_value = 0.1
    c_entropy = 0.01


class Actor(BasicActor):
    def __init__(self, env_fn, prepare_fn, device, net):
        BasicActor.__init__(self)

        self.env = env_fn()
        self.prepare_fn = prepare_fn
        self.device = device
        self.net = net

        self.done = True

        self.episodes_seen = 0
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

    def sample_frames(self, render=False):
        frames = []

        for i in range(Settings.sample_frames):
            if self.done:
                self.observation = self.prepare_fn(self.env.reset())
                self.done = False
                self.episodes_seen += 1

            action, prob = self.select_action()
            new_observation, reward, self.done, _ = self.env.step(int(action))

            frames.append((self.observation, action, prob, reward, self.done))
            self.observation = self.prepare_fn(new_observation)

            self.episode_rewards += reward
            if self.done:
                self.last_episode_rewards = self.episode_rewards
                self.episode_rewards = 0

            if render:
                self.render_frame()

        discounted_reward = 0.0
        for i in range(Settings.sample_frames-1, -1, -1):
            observation, action, prob, reward, done = frames[i]
            if done:
                discounted_reward = 0.0
            discounted_reward = Settings.gamma * discounted_reward + reward
            reward = discounted_reward
            frames[i] = (observation, action, prob, reward, done)

        return frames


class PPO(BasicAlgorithm):
    def __init__(self, env_fn, net, device="cpu", prepare_fn=lambda x: x, first_step=0):
        BasicAlgorithm.__init__(self)

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.prepare_fn = prepare_fn

        self.actors = [Actor(env_fn, prepare_fn, self.device, self.net)
                       for i in range(Settings.num_actors)]

        self.optimizer = optim.Adam(
            self.net.parameters(), maximize=True, lr=Settings.lr, weight_decay=0.0)

        self.frames_seen = 0
        self.step = first_step
        self.last_episode_rewards = 0

    def sample_frames(self, render):
        self.last_episode_rewards = 0
        self.episodes_seen = 0

        frames = []
        for i, actor in enumerate(self.actors):
            frames.extend(actor.sample_frames(render and i == 0))

            self.last_episode_rewards += actor.last_episode_rewards
            self.episodes_seen += actor.episodes_seen

        self.last_episode_rewards /= len(self.actors)
        self.frames_seen += len(frames)

        return frames

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

        pi = torch.softmax(scores/Settings.temp, axis=1)

        rate = pi[range(N), actions] / pi_old[range(N), actions]
        clipped_rate = torch.clip(
            rate, 1.0 - Settings.epsilon, 1.0 + Settings.epsilon)

        adv = rewards - V.detach()

        loss_clip = torch.mean(torch.min(rate * adv, clipped_rate * adv))

        log_pi = torch.log_softmax(scores/Settings.temp, axis=1)
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
        frames = self.sample_frames(render)

        rewards, loss = self.optimize(frames)
        self.step += 1

        return rewards, loss

    def write_video(self, episode=None, filename=None):
        self.actors[0].write_video(episode, filename)


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
