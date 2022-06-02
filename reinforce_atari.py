
import random
import argparse

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
parser.add_argument('--traceback', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()

env = gym.make(args.env)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 3)
        self.conv2 = nn.Conv2d(8, 16, 5, 3)
        self.conv3 = nn.Conv2d(16, 32, 5, 3)
        self.conv4 = nn.Conv2d(32, 64, 5, 3)
        self.conv5 = nn.Conv2d(64, 128, 5, 3)
        self.conv6 = nn.Conv2d(128, 256, 5, 3)
        self.linear1 = nn.Linear(2560, 100)
        self.fc = nn.Linear(100, env.action_space.n)

    def forward(self, x):
        x = x.swapdims(1, 3).type(torch.FloatTensor).cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape((1, -1))
        x = F.relu(self.linear1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

net = Net()
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-7)

def train(episode):
    images = []
    history = []
    observation = env.reset()
    done = False

    sum_loss = 0.0

    render = (episode % 10 == 0)

    while not done or len(history) > 0:
        if not done:
            with torch.no_grad():
                log_probs = net(torch.tensor(np.expand_dims(observation, 0)))
            distr = D.Categorical(probs = torch.exp(log_probs[0] / args.temp))
            action = distr.sample()

            if render:
                images.append(np.expand_dims(env.render(mode="rgb_array"), axis=0))
            new_observation, reward, done, info = env.step(int(action))

            history.append((observation, action, reward))
            observation = new_observation


        if len(history)>args.traceback or done:
            optimizer.zero_grad()

            loss = 0.0
    
            G = 0.0
            for observation, action, reward in reversed(history):
                G = G * args.gamma + reward

            observation, action, reward = history[0]

            log_probs = net(torch.tensor(np.expand_dims(observation, 0)))
            loss = -G * log_probs[0, action]

            loss.backward()
            optimizer.step()

            sum_loss += float(loss.detach().cpu())

            history = history[1:]

    if render:
        filename = "episode_%06d.mp4" % episode
        images = np.concatenate(images, axis = 0)
        images = (images * 255).astype(np.uint8)

        skvideo.io.vwrite(filename, images)

        print("wrote %s from %s" % (filename, str(images.shape)))

    return sum_loss

###
if args.first_episode > 0:
    net.load_state_dict(torch.load("episode_%06d" % args.first_episode))

for episode in range(args.first_episode, args.num_episodes):
    loss = train(episode)
    print("episode %6d, loss = %3.6f" % (episode, loss))

    if episode % 10 == 0:
        torch.save(net.state_dict(), "episode_%06d" % episode)
