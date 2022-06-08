
import argparse
import gfootball.env as football_env

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dql

dql.Settings.replay_memory_size = 1000000
dql.Settings.lr = 0.00001

parser = argparse.ArgumentParser()
parser.add_argument('--first_episode', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=10000)

args = parser.parse_args()


class FootballNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.linear1 = nn.Linear(460, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


def main():
    env = football_env.create_environment(
        env_name="11_vs_11_stochastic",
        stacked=True,
        representation = "simple115v2",
        write_video=True,
        dump_frequency=100,
        write_full_episode_dumps=True,
        render=False,
        logdir="logs")

    net = FootballNet(env)

    if args.first_episode > 0:
        print("loading weights")
        net.load_state_dict(torch.load("episode_%06d" % args.first_episode))

    trainer = dql.DQL(env, net, device="cuda",
                      episode=args.first_episode)

    rewards = []

    for episode in range(args.first_episode, args.num_episodes):
        magic = (episode % 100 == 0)

        reward, loss = trainer.train()
        rewards.append(reward)
        if len(rewards) > 100:
            del rewards[0]

        print("episode = %6d, frame = %6d, loss = %3.6f, reward = %3.6f, mean_reward = %3.6f, epsilon = %.6f" %
              (episode, trainer.frame, loss, reward, np.mean(rewards), trainer.epsilon))

        if magic:
            torch.save(net.state_dict(), "episode_%06d" % episode)


if __name__ == '__main__':
    main()
