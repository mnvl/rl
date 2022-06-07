
import argparse
import gym

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dql

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="ALE/Breakout-v5")
parser.add_argument('--first_episode', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=10000)

args = parser.parse_args()


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


def main():
    env = gym.make(args.env, full_action_space=False)
    net = AtariNet(env)
    pre = AtariPre()

    if args.first_episode > 0:
        print("loading weights")
        net.load_state_dict(torch.load("episode_%06d" % args.first_episode))

    trainer = dql.DQL(env, net, device="cuda", episode=args.first_episode, prepare=pre)

    rewards = []

    for episode in range(args.first_episode, args.num_episodes):
        magic = (episode % 100 == 0)

        reward, loss = trainer.train(render=magic)
        rewards.append(reward)
        if len(rewards) > 100:
            del rewards[0]

        print("episode %6d, loss = %3.6f, reward = %3.6f, mean_reward = %3.6f, epsilon = %.6f" %
              (episode, loss, reward, np.mean(rewards), trainer.epsilon))

        if magic:
            trainer.write_video(episode)
            torch.save(net.state_dict(), "episode_%06d" % episode)


if __name__ == '__main__':
    main()
