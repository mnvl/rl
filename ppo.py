
import copy
import time
import os
import unittest
import multiprocessing as mp

import gym
import numpy as np
import skvideo.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

from basic_algorithm import BasicActor, BasicAlgorithm, MarsRoverEnv


class Settings:
    lr = 0.001

    gamma = 0.99

    horizon = 256
    num_actors = 8

    epsilon = 0.2
    c_value = 1.0
    c_entropy = 0.01

    alpha_0 = 1.0
    alpha_1 = 0.0
    alpha_steps = 100000

    write_videos = True


class Worker:
    def __init__(self, index, env_fn, prepare_fn, device, net):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(target=self.run, args=(
            index, self.child_conn, env_fn, prepare_fn, device, net))
        self.process.start()

    def read_frames(self):
        frames = self.parent_conn.recv()
        return frames

    def send_state_dict(self, state_dict):
        self.parent_conn.send(state_dict)

    def stop(self):
        self.send_action(-1)
        self.process.join()

    def run(self, index, child_conn, env_fn, prepare_fn, device, net):
        env = env_fn()
        prepare = prepare_fn()
        net = copy.deepcopy(net).to(device)

        done = True

        while True:
            state_dict = child_conn.recv()

            if state_dict is None:
                return

            net.load_state_dict(state_dict)

            frames = []

            for i in range(Settings.horizon):
                if done:
                    observation = prepare(env.reset())

                s = np.expand_dims(observation, 0)
                with torch.no_grad():
                    scores, V = net(torch.tensor(s, device=device))
                probs = torch.softmax(scores, axis=1)
                distr = D.Categorical(probs=probs[0])
                action = distr.sample()

                new_observation, reward, done, _ = env.step(action)
                frames.append((observation, action.cpu(), np.array(probs[0].cpu()), float(V[0].cpu()), reward, done))

                observation = prepare(new_observation)

            child_conn.send(frames)


class Sampler:
    def __init__(self, env_fn, prepare_fn, device, net):
        self.workers = [Worker(j, env_fn, prepare_fn, device, net)
                        for j in range(Settings.num_actors)]

        self.net = net
        self.device = device

        self.frames_seen = 0
        self.episodes_seen = 0
        self.last_episode_rewards = [0.0 for j in range(Settings.num_actors)]
        self.episode_rewards = [0.0 for j in range(Settings.num_actors)]

    def sample_frames(self, render=False):
        frames = []

        state_dict = self.net.state_dict()

        for j in range(Settings.num_actors):
            self.workers[j].send_state_dict(state_dict)

        observations = []
        for j in range(Settings.num_actors):
            fr = self.workers[j].read_frames()

            frames.append(fr)
            observations.append(fr[-1][0])

            self.frames_seen += len(fr)

        s = [np.expand_dims(observation, 0)
             for observation in observations]
        s = np.concatenate(s, axis=0)
        with torch.no_grad():
            _, V = self.net(torch.tensor(s, device=self.device))
        V = V.cpu()

        updated_frames = []
        for j in range(Settings.num_actors):
            updated_value = V[j]

            for frame in reversed(frames[j]):
                observation, action, prob, value, reward, done = frame

                if done:
                    updated_value = reward
                else:
                    updated_value = updated_value * Settings.gamma + reward

                updated_frames.append((observation, action, prob,
                                       updated_value, done))

        return updated_frames

    def stop(self):
        for w in self.workers:
            w.stop()


class PPO(BasicAlgorithm):
    def __init__(self, env_fn, net, device="cpu", prepare_fn=lambda: lambda x: x, first_step=0):
        BasicAlgorithm.__init__(self)

        self.device = torch.device(device)
        self.net = net.to(self.device)

        self.sampler = Sampler(env_fn, prepare_fn, self.device, self.net)

        self.optimizer = optim.Adam(
            self.net.parameters(), maximize=True, lr=Settings.lr, weight_decay=0.0)

        self.step = first_step

    def optimize(self, frames):
        observations = []
        values = []
        actions = []
        pi_old = []

        for observation, action, pi, value, done in frames:
            observations.append(torch.Tensor(np.expand_dims(observation, 0)))
            actions.append(action)
            values.append(value)
            pi_old.append(torch.Tensor(np.expand_dims(pi, 0)))

        observations = torch.cat(observations, axis=0).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        values = torch.Tensor(values).to(self.device)
        pi_old = torch.cat(pi_old, axis=0).to(self.device)

        N = observations.shape[0]

        self.optimizer.zero_grad()

        scores, V = self.net(observations)

        pi = torch.softmax(scores, axis=1)

        rate = pi[range(N), actions] / pi_old[range(N), actions]
        clipped_rate = torch.clip(
            rate, 1.0 - Settings.epsilon * self.alpha, 1.0 + Settings.epsilon * self.alpha)

        adv = values - V.detach()

        loss_clip = torch.mean(torch.min(rate * adv, clipped_rate * adv))

        log_pi = torch.log_softmax(scores, axis=1)

        loss_value = torch.mean(torch.square(values - V))

        loss_entropy = -torch.mean(pi * log_pi)

        loss = loss_clip - \
            Settings.c_value * loss_value + \
            Settings.c_entropy * loss_entropy

        loss.backward()
        self.optimizer.step()

        self.writer.add_histogram("pi", pi.reshape(-1), self.step)
        self.writer.add_histogram(
            "value/values", values.reshape(-1), self.step)
        self.writer.add_histogram("value/V", V.reshape(-1), self.step)
        self.writer.add_histogram("value/adv", adv.reshape(-1), self.step)
        self.writer.add_histogram("pi/rate", rate.reshape(-1), self.step)

        self.writer.add_scalar("loss/clip", loss_clip, self.step)
        self.writer.add_scalar("loss/value", loss_value, self.step)
        self.writer.add_scalar("loss/entropy", loss_entropy, self.step)
        self.writer.add_scalar("loss", loss, self.step)
        self.writer.add_scalar("value/values_mean",
                               values.reshape(-1).mean(), self.step)
        self.writer.add_scalar(
            "value/values_std", values.reshape(-1).std(), self.step)
        self.writer.add_scalar("value/V_mean", V.reshape(-1).mean(), self.step)
        self.writer.add_scalar("value/V_std", V.reshape(-1).std(), self.step)
        self.writer.add_scalar(
            "value/adv_mean", adv.reshape(-1).mean(), self.step)
        self.writer.add_scalar(
            "value/adv_std", adv.reshape(-1).std(), self.step)

        return self.last_episode_rewards, float(loss)

    def train(self, render=False):
        self.alpha = Settings.alpha_0 + \
            (Settings.alpha_1 - Settings.alpha_0) * \
            self.step / Settings.alpha_steps
        for g in self.optimizer.param_groups:
            g['lr'] = Settings.lr * self.alpha

        t1 = time.time()
        frames = self.sampler.sample_frames(render)
        self.frames_seen = self.sampler.frames_seen
        self.episodes_seen = self.sampler.episodes_seen
        self.last_episode_rewards = np.mean(self.sampler.last_episode_rewards)

        t2 = time.time()
        rewards, loss = self.optimize(frames)

        t3 = time.time()

        t_sample = t2 - t1
        t_optimize = t3 - t2

        self.writer.add_scalar("time/sample", t_sample, self.step)
        self.writer.add_scalar("time/optimize", t_optimize, self.step)
        self.writer.add_scalar("rewards", self.last_episode_rewards, self.step)

        self.step += 1

        return rewards, loss

    def stop(self):
        self.sampler.stop()


class TestPPO(unittest.TestCase):
    def setUp(self):
        self.save = (
            Settings.lr,
            Settings.gamma,
            Settings.write_videos)

        Settings.write_videos = False

    def tearDown(self):
        (Settings.lr,
         Settings.gamma,
         Settings.write_videos) = self.save

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

        trainer.stop()

        self.assertEqual(rewards, 10.0)

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

        trainer.stop()

        self.assertGreater(rewards, 400)


if __name__ == '__main__':
    unittest.main()
