
import os
import random
import argparse
import unittest
import copy

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import skvideo.io

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="ALE/Breakout-v5")
parser.add_argument('--first_episode', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epsilon1', type=float, default=1.0)
parser.add_argument('--epsilon2', type=float, default=0.05)
parser.add_argument('--epsilon_decay', type=float, default=1000)
parser.add_argument('--steps_per_batch', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--replay_memory_size', type=int, default=200000)
parser.add_argument('--update_target_every_episodes', type=int, default=25)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = parser.parse_args("")


class AtariNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(3072, 512)
        self.linear2 = nn.Linear(512, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class AtariPre:
    def __init__(self):
        self.lookback = []

    def __call__(self, s):
        # Pong
        if False:
            s = s[32:196, :, :]

        # Breakout
        if True:
            s = s[26:, :, :]

        s = s.astype(np.float32)
        s = s.mean(axis=2)/255.0
        s = s.reshape((-1, 2, 160//2, 2)).max(axis=1).max(axis=2)
        s = np.expand_dims(s, 0)

        if len(self.lookback) == 0:
            self.lookback = [s, s, s, s]

        self.lookback.append(s)
        del self.lookback[0]

        return np.concatenate(self.lookback, axis=0).copy()


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
            net.parameters(), lr=args.lr, weight_decay=0.0)

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

        if len(self.replay_memory) >= args.replay_memory_size:
            self.replay_memory = self.replay_memory[1:]
        self.replay_memory.append(memory)
        self.observation = new_observation

        if self.frame % 1000 == 0:
            self.writer.add_image("step_observation", make_grid(torch.tensor(np.expand_dims(self.observation, 1))), self.episode)
            self.writer.add_histogram("step_observation_hist", self.observation.reshape(-1), self.episode)

        self.frame += 1

        return reward

    def optimize(self):
        batch = [random.choice(self.replay_memory)
                 for i in range(args.batch_size)]

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
                y.append(reward + args.gamma * best)

        y = torch.tensor(y, device=self.device)

        loss = torch.square(Qi[range(Qi.shape[0]), actions] - y)
        loss = torch.mean(loss)

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu())

    def train(self, render=False, steps_per_batch=args.steps_per_batch):
        self.observation = self.prepare(self.env.reset())
        self.done = False

        rewards = 0.0
        losses = []

        alpha = min(self.episode / args.epsilon_decay, 1.0)
        self.epsilon = args.epsilon1 + alpha * (args.epsilon2-args.epsilon1)

        if self.episode % args.update_target_every_episodes == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        while not self.done:
            for i in range(steps_per_batch):
                rewards += self.step()
                if self.done: break
                if render: self.render_frame()

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
            args.lr,
            args.gamma,
            args.update_target_every_episodes,
            args.epsilon1,
            args.epsilon2,
            args.epsilon_decay,
            args.stop_epsilon_interp_episodes)

        args.update_target_every_episodes = 100
        args.epsilon_decay = 100
        args.stop_epsilon_interp_episodes = 1000

    def tearDown(self):
        (args.lr,
         args.gamma,
         args.update_target_every_episodes,
         args.epsilon1,
         args.epsilon2,
         args.epsilon_decay,
        args.stop_epsilon_interp_episodes) = self.save

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
        args.lr = 0.005

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
        args.lr = 0.005

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
        args.lr = 0.005

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

        args.lr = 0.001

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


def main():
    env = gym.make(args.env, full_action_space=False)
    net = AtariNet(env)
    pre = AtariPre()
    dql = DQL(env, net, device="cuda", prepare=pre)

    if args.first_episode > 0:
        print("loading weights")
        net.load_state_dict(torch.load("episode_%06d" % args.first_episode))

    rewards = []

    for episode in range(args.first_episode, args.num_episodes):
        magic = (episode % 100 == 0)

        reward, loss = dql.train(render=magic)
        rewards.append(reward)
        if len(rewards) > 100:
            del rewards[0]

        print("episode %6d, loss = %3.6f, reward = %3.6f, mean_reward = %3.6f, epsilon = %.6f" %
              (episode, loss, reward, np.mean(rewards), dql.epsilon))

        if magic:
            dql.write_video(episode)
            torch.save(net.state_dict(), "episode_%06d" % episode)


if __name__ == '__main__':
    main()
