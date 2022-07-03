
import time
import os
import unittest
import multiprocessing as mp

import gym
import numpy as np
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

from basic_algorithm import BasicActor, BasicAlgorithm, MarsRoverEnv


class Settings:
    num_workers = mp.cpu_count()
    envs_per_worker = 1
    num_envs = num_workers * envs_per_worker

    lr = 0.001

    gamma = 0.99

    horizon = 32

    epsilon = 0.2
    c_value = 1.0
    c_entropy = 0.01

    alpha_0 = 1.0
    alpha_1 = 0.0
    alpha_steps = 100000

    write_videos = True


class EnvironmentWrapper:
    def __init__(self, index, env_fn, prepare_fn):
        self.index = index
        self.prepare_fn = prepare_fn

        self.env = env_fn()
        self.prepare = self.prepare_fn()

        self.observation = self.prepare(self.env.reset())
        self.reward = 0.0
        self.done = False

        self.num_episodes = 0
        self.num_frames = 0
        self.frames = []

    def get_observation(self):
        result = (self.observation, self.reward, self.done)
        return result

    def step(self, action):
        self.observation, self.reward, self.done, _ = self.env.step(action)
        self.observation = self.prepare(self.observation)

        self.num_frames += 1

        if self.index == 0 and Settings.write_videos and self.num_episodes % 25 == 0:
            image = env.render(mode="rgb_array")
            image = np.expand_dims(image, axis=0)
            self.frames.append(image)

            if self.done:
                frames = np.concatenate(frames, axis=0)
                frames = (frames * 255).astype(np.uint8)
                filename = "episode_%04d_%06d_step_%09d.mp4" % (
                    os.getpid(), num_episodes, num_frames // Settings.horizon)
                imageio.mimwrite(filename, frames, fps=60)

                frames = []
                last_save_time = time.time()

        if self.done:
            self.prepare = self.prepare_fn()
            self.observation = self.prepare(self.env.reset())
            self.num_episodes += 1


class Worker:
    def __init__(self, env_fn, prepare_fn):
        self.envs = [EnvironmentWrapper(i, env_fn, prepare_fn)
                     for i in range(Settings.envs_per_worker)]
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(target=self.run, args=(self.child_conn, ))
        self.process.start()

    def run(self, connection):
        while True:
            observations = [env.get_observation() for env in self.envs]
            connection.send(observations)

            actions = connection.recv()
            if type(actions) is int and actions == -1:
                return

            for action, env in zip(actions, self.envs):
                env.step(action)

    def receive_statuses(self):
        status = self.parent_conn.recv()
        return status

    def send_actions(self, actions):
        self.parent_conn.send(actions)

    def stop(self):
        self.send_actions(-1)
        self.process.join()


class Sampler:
    def __init__(self, env_fn, prepare_fn, device, net):
        self.workers = [Worker(env_fn, prepare_fn)
                        for j in range(Settings.num_workers)]
        self.device = device
        self.net = net

        self.observations = []
        for j in range(Settings.num_workers):
            statuses = self.workers[j].receive_statuses()
            self.observations.extend([observation for observation, reward, done in statuses])

        self.frames_seen = 0
        self.episodes_seen = 0
        self.last_episode_rewards = [0.0 for j in range(Settings.num_envs)]
        self.episode_rewards = [0.0 for j in range(Settings.num_envs)]

    def select_actions(self):
        s = [np.expand_dims(observation, 0)
             for observation in self.observations]
        s = np.concatenate(s, axis=0)
        with torch.no_grad():
            scores, V = self.net(torch.tensor(s, device=self.device))
        probs = torch.softmax(scores, axis=1)
        distr = D.Categorical(probs=probs)
        actions = distr.sample()
        return actions.cpu(), probs.cpu(), V.cpu()

    def sample_frames(self, render=False):
        frames = [[] for j in range(Settings.num_envs)]

        for i in range(Settings.horizon):
            actions, probs, values = self.select_actions()

            for j in range(Settings.num_workers):
                a = [int(action) for action in actions[j:j+Settings.envs_per_worker]]
                self.workers[j].send_actions(a)

            statuses = []
            for j in range(Settings.num_workers):
                statuses.extend(self.workers[j].receive_statuses())

            for j in range(Settings.num_envs):
                new_observation, reward, done = statuses[j]
                self.frames_seen += 1

                frames[j].append(
                    (self.observations[j], actions[j], probs[j], values[j], reward, done))
                self.observations[j] = new_observation

                self.episode_rewards[j] += reward

                if done:
                    self.last_episode_rewards[j] = self.episode_rewards[j]
                    self.episode_rewards[j] = 0
                    self.episodes_seen += 1

        s = [np.expand_dims(observation, 0)
             for observation in self.observations]
        s = np.concatenate(s, axis=0)
        with torch.no_grad():
            _, V = self.net(torch.tensor(s, device=self.device))
        V = V.cpu()

        updated_frames = []
        for j in range(Settings.num_envs):
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
