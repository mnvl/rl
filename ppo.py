
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
    def __init__(self, index, env_fn, prepare_fn):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(target=self.run, args=(
            index, self.child_conn, env_fn, prepare_fn))
        self.process.start()

    def read_status(self):
        status = self.parent_conn.recv()
        return status

    def send_action(self, action):
        self.parent_conn.send(int(action))

    def stop(self):
        self.send_action(-1)
        self.process.join()

    def run(self, index, child_conn, env_fn, prepare_fn):
        env = env_fn()
        prepare = prepare_fn()

        observation = prepare(env.reset())
        reward = 0.0
        done = False
        child_conn.send((observation, reward, done))

        episode = 0
        frames = []

        while True:
            action = child_conn.recv()
            if action == -1:
                return

            observation, reward, done, _ = env.step(action)
            observation = prepare(observation)
            child_conn.send((observation, reward, done))

            if index == 0 and episode % 100 == 0 and Settings.write_videos:
                image = env.render(mode="rgb_array")
                image = np.expand_dims(image, axis=0)
                frames.append(image)

                if done:
                    frames = np.concatenate(frames, axis=0)
                    frames = (frames * 255).astype(np.uint8)
                    skvideo.io.vwrite("episode_%06d.avi" % episode, frames)
                    frames = []

            if done:
                observation = prepare(env.reset())
                reward = 0.0
                done = False
                episode += 1


class Actors:
    def __init__(self, env_fn, prepare_fn, device, net):
        BasicActor.__init__(self)

        self.workers = [Worker(j, env_fn, prepare_fn)
                        for j in range(Settings.num_actors)]
        self.device = device
        self.net = net

        self.observations = []
        for j in range(Settings.num_actors):
            observation, reward, done = self.workers[j].read_status()
            self.observations.append(observation)
        self.frames = [[]for j in range(Settings.num_actors)]

        self.frames_seen = 0
        self.episodes_seen = 0
        self.last_episode_rewards = [0.0 for j in range(Settings.num_actors)]
        self.episode_rewards = [0.0 for j in range(Settings.num_actors)]

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
        while len(self.frames[0]) < Settings.horizon + 1:
            actions, probs, values = self.select_actions()

            for j in range(Settings.num_actors):
                self.workers[j].send_action(actions[j])

            for j in range(Settings.num_actors):
                new_observation, reward, done = self.workers[j].read_status()
                self.frames_seen += 1

                self.frames[j].append(
                    (self.observations[j], actions[j], probs[j], values[j], reward, done))
                self.observations[j] = new_observation

                self.episode_rewards[j] += reward

                if done:
                    self.last_episode_rewards[j] = self.episode_rewards[j]
                    self.episode_rewards[j] = 0
                    self.episodes_seen += 1

        updated_frames = []
        for j in range(Settings.num_actors):
            assert len(self.frames[j]) == Settings.horizon + 1
            _, _, _, value, reward, done = self.frames[j][Settings.horizon]
            updated_value = reward if done else value

            for i in range(Settings.horizon-1, -1, -1):
                observation, action, prob, value, reward, done = self.frames[j][i]

                if done:
                    updated_value = reward
                else:
                    updated_value = updated_value * Settings.gamma + reward

                updated_frames.append((observation, action, prob,
                                       updated_value, done))

            self.frames[j] = self.frames[j][Settings.horizon:]
            assert len(self.frames[j]) == 1

        return updated_frames

    def stop(self):
        for w in self.workers:
            w.stop()


class PPO(BasicAlgorithm):
    def __init__(self, env_fn, net, device="cpu", prepare_fn=lambda: lambda x: x, first_step=0):
        BasicAlgorithm.__init__(self)

        self.device = torch.device(device)
        self.net = net.to(self.device)

        self.actors = Actors(env_fn, prepare_fn, self.device, self.net)

        self.optimizer = optim.Adam(
            self.net.parameters(), maximize=True, lr=Settings.lr, weight_decay=0.0)

        self.step = first_step

    def optimize(self, frames):
        observations = []
        values = []
        actions = []
        pi_old = []

        for observation, action, pi, reward, done in frames:
            observations.append(torch.Tensor(np.expand_dims(observation, 0)))
            actions.append(action)
            values.append(reward)
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
        self.writer.add_histogram("value/values", values.reshape(-1), self.step)
        self.writer.add_histogram("value/V", V.reshape(-1), self.step)
        self.writer.add_histogram("value/adv", adv.reshape(-1), self.step)
        self.writer.add_histogram("pi/rate", rate.reshape(-1), self.step)

        self.writer.add_scalar("loss/clip", loss_clip, self.step)
        self.writer.add_scalar("loss/value", loss_value, self.step)
        self.writer.add_scalar("loss/entropy", loss_entropy, self.step)
        self.writer.add_scalar("loss", loss, self.step)
        self.writer.add_scalar("value/values_mean", values.reshape(-1).mean(), self.step)
        self.writer.add_scalar("value/values_std", values.reshape(-1).std(), self.step)
        self.writer.add_scalar("value/V_mean", V.reshape(-1).mean(), self.step)
        self.writer.add_scalar("value/V_std", V.reshape(-1).std(), self.step)
        self.writer.add_scalar("value/adv_mean", adv.reshape(-1).mean(), self.step)
        self.writer.add_scalar("value/adv_std", adv.reshape(-1).std(), self.step)

        return self.last_episode_rewards, float(loss)

    def train(self, render=False):
        self.alpha = Settings.alpha_0 + \
            (Settings.alpha_1 - Settings.alpha_0) * \
            self.step / Settings.alpha_steps
        for g in self.optimizer.param_groups:
            g['lr'] = Settings.lr * self.alpha

        t1 = time.time()
        frames = self.actors.sample_frames(render)
        self.frames_seen = self.actors.frames_seen
        self.episodes_seen = self.actors.episodes_seen
        self.last_episode_rewards = np.mean(self.actors.last_episode_rewards)

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
        self.actors.stop()

    def write_video(self, episode=None, filename=None):
        #self.actors.write_video(episode, filename)
        pass


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

        trainer.write_video(filename="test_cartpole.mp4")

        self.assertGreater(rewards, 400)


if __name__ == '__main__':
    unittest.main()
