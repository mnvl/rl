
import sys
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import dql
import ppo

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="ALE/Breakout-v5")
parser.add_argument('--algo', type=str, default="ppo")
parser.add_argument('--lr', type=float, default="0.001")
parser.add_argument('--load_step', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=10000)

args = parser.parse_args()


class AtariNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(3072, 512)
        self.fc1 = nn.Linear(512, env.action_space.n)

        if args.algo == "ppo":
            self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.linear1(x))

        if args.algo == "ppo":
            return self.fc1(x), self.fc2(x)

        return self.fc1(x)


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

    if args.load_step > 0:
        print("loading weights")
        net.load_state_dict(torch.load("step_%06d" % args.first_episode))

    if args.algo == "ppo":
        ppo.Settings.lr = args.lr
        trainer = ppo.PPO(env, net, device="cuda", prepare=pre)
    elif args.algo == "dql":
        dql.Settings.lr = args.lr
        trainer = dql.DQL(env, net, device="cuda", prepare=pre)
    else:
        print("unkonwn algorithm", args.algo)
        sys.exit(1)

    for i in range(args.num_steps):
        magic = (i % 100 == 0)

        reward, cpi_loss, kl_loss, v_loss, loss = trainer.train(render=magic)

        print(i, reward, cpi_loss, kl_loss, v_loss, loss)

        if magic:
            trainer.write_video(filename = "step_%06d.avi" % i)
            torch.save(net.state_dict(), "step_%06d" % i)


if __name__ == '__main__':
    main()
