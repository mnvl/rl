
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
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--min_replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_size', type=int, default=100000)

args = parser.parse_args()
env = gym.make(args.env)

class AtariNet(nn.Module):
    def __init__(self):
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
    def __init__(self, env, net):
        self.env = env
        self.net = net
        self.replay_memory = []
        self.frames = []
        self.done = False

    def select_action(self, epsilon=args.epsilon):
        if random.random() < args.epsilon:
            action = int(random.randint(0, env.action_space.n-1))
        else:
            with torch.no_grad():
                Q = self.net(torch.tensor(np.expand_dims(self.observation, 0)))
                action = int(np.argmax(Q.cpu().numpy()[0]))

        return action

    def render_frame(self):
        image = env.render(mode="rgb_array")
        image = image.copy()
        image = np.expand_dims(image, axis=0)
        self.frames.append(image)

    def step(self):
        action = self.select_action()

        new_observation, reward, self.done, info = env.step(action)
        self.replay_memory.append((observation.copy(), int(action), float(reward), new_observation.copy()))

        self.observation = new_observation


    def train(self):
        self.observation = env.reset()

        while not self.done:
            self.step()
        

class MockEnv:
    def __init__(self):
        self.state = 0
        self.action_space = { "n": 5 }

    def reset(self):
        self.state = 0
        return 0

    def step(self, action):
        self.state += random.randint(0, 10) * action
        return np.arrray([self.state]), self.state if self.state <= 21 else 0, self.state >= 19, None


class TestDQN(unittest.TestCase):
    def test_dqn_select_action(self):
        net = lambda x: torch.Tensor([[1, 2, 3, 5, 1]])
        trainer = DQL(MockEnv(), net)
        self.assertEqual(trainer.select_action(epsilon=0), 3)

        
        

def train(episode):
    global replay_memory

    images = []

    render = (episode % 100 == 0)

    observation = env.reset()
    done = False

    sum_rewards = 0.0
    sum_loss = 0.0

    while not done:
        for i in range(args.batch_size // 2):

            sum_rewards += reward

            if len(replay_memory) > args.replay_memory_size:
                replay_memory = replay_memory[1:]


            if done:
                break

        if len(replay_memory) < args.min_replay_memory_size:
            continue

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
        Qi = net(torch.tensor(X) / 256.0 - 0.5)
        Qi = Qi[range(args.batch_size), torch.tensor(actions)]

        y = torch.tensor(y).cuda()
        loss = torch.square(Qi - y)
        loss = torch.mean(loss)

        sum_loss += float(loss.detach().cpu())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)

        optimizer.step()

    if render:
        filename = "episode_%06d.mp4" % episode
        images = np.concatenate(images, axis = 0)
        images = (images * 255).astype(np.uint8)

        skvideo.io.vwrite(filename, images)

        print("wrote %s from %s" % (filename, str(images.shape)))

    return sum_loss, sum_rewards

###
'''
if args.first_episode > 0:
    net.load_state_dict(torch.load("episode_%06d" % args.first_episode))

for episode in range(args.first_episode, args.num_episodes):
    loss, reward = train(episode)

    print("episode %6d, loss = %3.6f, reward = %3.6f, rm = %d, gc = %s" % (episode, loss, reward, len(replay_memory), gc.collect()))

    if episode % 100 == 0:
        torch.save(net.state_dict(), "episode_%06d" % episode)
'''

if __name__ == '__main__':
    unittest.main()
