
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
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--replay_memory_size', type=int, default=100000)

args = parser.parse_args()

env = gym.make(args.env)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 3)
        self.conv2 = nn.Conv2d(8, 16, 3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, 2)
        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 100)
        self.fc = nn.Linear(100, env.action_space.n)

    def forward(self, x):
        x = x.swapdims(1, 3).type(torch.FloatTensor).cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape((-1, 768))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.fc(x)
        return x

net = Net()
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-7)

replay_memory = []

def train(episode):
    global replay_memory

    images = []

    render = (episode % 10 == 0)

    observation = env.reset()
    done = False

    sum_loss = 0.0

    while not done:
        if random.random() < args.epsilon:
            action = int(random.randint(0, env.action_space.n-1))
        else:
            with torch.no_grad():
                qi = net(torch.tensor(np.expand_dims(observation, 0)))
            action = int(np.argmax(qi.cpu().numpy()[0]))

        if render:
            images.append(np.expand_dims(env.render(mode="rgb_array"), axis=0))

        new_observation, reward, done, info = env.step(action)

        if len(replay_memory) > args.replay_memory_size:
            replay_memory = replay_memory[1:]

        replay_memory.append((observation, action, reward, new_observation))

        batch = [random.choice(replay_memory) for i in range(args.batch_size)]
        X = []
        for sample in batch:
            observation, action, reward, new_observation = sample
            X.append(np.expand_dims(new_observation, 0))
        X = np.concatenate(X, axis = 0)
        with torch.no_grad():
            Qj = net(torch.tensor(X))

        X = []
        y = []
        actions = []
        for sample, qj in zip(batch, Qj):
            observation, action, reward, new_observation = sample

            X.append(np.expand_dims(observation, 0))
            actions.append(action)
            y.append(reward + args.gamma * np.max(qj.cpu().numpy()))

        optimizer.zero_grad()
        X = np.concatenate(X, axis = 0)
        Qi = net(torch.tensor(X))
        Qi = Qi[range(args.batch_size), torch.tensor(actions)]

        y = torch.tensor(y).cuda()
        loss = (Qi - y)**2
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()

        sum_loss += float(loss.detach().cpu())

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
