
import os
import gc
import random
import argparse
import unittest

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

import skvideo.io

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="ALE/Breakout-v5")
parser.add_argument('--temp', type=float, default=1)
parser.add_argument('--first_episode', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--steps_per_batch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--min_replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_size', type=int, default=100000)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = parser.parse_args("")


class AtariNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, 3)
        self.conv2 = nn.Conv2d(4, 8, 3, 2)
        self.conv3 = nn.Conv2d(8, 12, 3, 2)
        self.conv4 = nn.Conv2d(12, 14, 3, 2)
        self.conv5 = nn.Conv2d(14, 18, 3, 2)
        self.linear1 = nn.Linear(108, 50)
        self.linear2 = nn.Linear(50, 50)
        self.fc = nn.Linear(50, env.action_space.n)

    def forward(self, x):
        x = x.swapdims(1, 3).type(torch.FloatTensor).cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape((-1, 108))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.fc(x)
        return x


class DQL:
    def __init__(self, env, net, device="cpu"):
        self.env = env
        self.net = net
        self.device = torch.device(device)

        self.replay_memory = []
        self.frames = []
        self.done = False

        self.optimizer = optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=0.001)

    def select_action(self, epsilon=args.epsilon):
        if random.random() < args.epsilon:
            return int(random.randint(0, self.env.action_space.n-1))
        
        X = np.expand_dims(self.observation, 0)
        with torch.no_grad():
            Q = self.net(torch.tensor(X, device=self.device))
        return int(np.argmax(Q.cpu().numpy()[0]))

    def render_frame(self):
        image = self.env.render(mode="rgb_array")
        image = image.copy()
        image = np.expand_dims(image, axis=0)
        self.frames.append(image)

    def step(self):
        action = self.select_action()

        new_observation, reward, self.done, info = self.env.step(action)

        memory = (
            self.observation.copy(),
            int(action),
            float(reward),
            new_observation.copy(),
            self.done)

        if len(self.replay_memory) >= args.replay_memory_size:
            self.replay_memory = self.replay_memory[1:]
        self.replay_memory.append(memory)

        self.observation = new_observation

        return reward

    def optimize(self):
        batch = [random.choice(self.replay_memory)
                 for i in range(args.batch_size)]

        X = []
        for sample in batch:
            observation, action, reward, new_observation, done = sample
            X.append(np.expand_dims(new_observation, 0))
        X = np.concatenate(X, axis=0)
        with torch.no_grad():
            Qj = self.net(torch.tensor(X, device=self.device))

        X = []
        y = []
        actions = []
        for sample, qj in zip(batch, Qj):
            observation, action, reward, new_observation, done = sample

            best = np.max(qj.cpu().numpy())

            X.append(np.expand_dims(observation, 0))
            actions.append(action)

            if done:
                y.append(reward)
            else:
                y.append(reward + args.gamma * best)

        X = np.concatenate(X, axis=0)
        self.optimizer.zero_grad()
        Qi = self.net(torch.tensor(X, device=self.device))
        Qi = Qi[range(args.batch_size), torch.tensor(
            actions, device=self.device)]

        y = torch.tensor(y, device=self.device)
        loss = torch.square(Qi - y)

        loss = torch.mean(loss)

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu())

    def train(self, render=False, steps_per_batch=args.steps_per_batch):
        self.observation = self.env.reset()
        self.done = False

        rewards = 0.0
        loss = 0.0

        while not self.done:
            for i in range(steps_per_batch):
                rewards += self.step()
                if self.done:
                    break

                if render:
                    self.render_frame()

            loss = self.optimize()

        return rewards, loss

    def write_video(self, episode=None, filename=None):
        if episode is not None:
            filename = "episode_%06d.mp4" % episode

        images = np.concatenate(self.frames, axis=0)
        images = (images * 255).astype(np.uint8)

        skvideo.io.vwrite(filename, images)

        print("wrote %s from %s" % (filename, str(images.shape)))

        self.frames = []


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
            self.state -= 5

        if action == 1 and not self.done:
            if self.randomized:
                self.state += random.randint(1, 10)
            else:
                self.state += 10

        if action == 2:
            self.done = True

        if self.state > 21 or self.state < 0:
            self.done = True

        reward = 0
        if self.done and self.state <= 21:
            reward = self.state

        observation = np.array(
            [self.state+np.random.normal(), self.done]).astype(np.float32)

        return observation, reward, self.done, None


class TestDQL(unittest.TestCase):
    def setUp(self):
        self.saved_lr = args.lr
        self.saved_epsilon = args.epsilon

    def tearDown(self):
        args.lr = self.saved_lr
        args.epsilon = self.saved_epsilon

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
        self.assertEqual(trainer.select_action(epsilon=0), 1)

    def test_train_deterministic(self):
        args.lr = 0.005

        env = MockEnv(randomized=False)
        net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 3))

        trainer = DQL(env, net)

        for i in range(1000):
            rewards, loss = trainer.train()
            if i % 100 == 0:
                print("regular", rewards, loss)

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
        self.assertEqual(trainer.select_action(epsilon=0.0), 1)

    def test_train_randomized(self):
        args.lr = 0.01

        env = MockEnv(randomized=True)
        net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 3))

        trainer = DQL(env, net)

        for i in range(1000):
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
        self.assertEqual(trainer.select_action(epsilon=0.0), 1)

    def test_train_cartpole(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        args.lr = 0.0005

        env = gym.make("CartPole-v1")

        net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n))

        trainer = DQL(env, net)

        num_episodes = 2000
        for i in range(num_episodes):
            magic = (i > num_episodes - 5)

            if magic:
                args.epsilon = 0

            rewards, loss = trainer.train(render=magic)

            if i % 100 == 0 or magic:
                print("cartpole", i, rewards, loss)

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(rewards, 200)

    def test_train_lunar(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        args.lr = 0.0005

        env = gym.make("LunarLander-v2")

        net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n))

        trainer = DQL(env, net)

        num_episodes = 2000
        for i in range(num_episodes):
            magic = (i > num_episodes - 5)

            if magic:
                args.epsilon = 0

            rewards, loss = trainer.train(render=magic)

            if i % 100 == 0 or magic:
                print("lunar", i, rewards, loss)

        trainer.write_video(filename="test_lunar.mp4")

        self.assertGreater(rewards, 200)


def main():
    env = gym.make(args.env)
    net = AtariNet(env).cuda()
    dql = DQL(env, net, device="cuda")

    if args.first_episode > 0:
        net.load_state_dict(torch.load("episode_%06d" % args.first_episode))

    for episode in range(args.first_episode, args.num_episodes):
        magic = (episode % 100 == 0)

        rewards, loss = dql.train(render=magic)

        print("episode %6d, loss = %3.6f, rewards = %3.6f" %
              (episode, loss, rewards))

        if magic:
            dql.write_video(episode)
            torch.save(net.state_dict(), "episode_%06d" % episode)


if __name__ == '__main__':
    main()
