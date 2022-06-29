
import sys
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import optuna

from tqdm import tqdm

import dql
import ppo

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="ALE/Breakout-v5")
parser.add_argument('--algo', type=str, default="ppo")
parser.add_argument('--lr', type=float, default="0.001")
parser.add_argument('--load_step', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=100000)

args = parser.parse_args()


class AtariNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(3072, 512)
        self.linear2 = nn.Linear(3072, 512)
        self.fc1 = nn.Linear(512, env.action_space.n)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear2(x))
        return self.fc1(x1), self.fc2(x2)


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


def ppo_objective(trial):
    ppo.Settings.lr = trial.suggest_float("lr", 1.0e-6, 1.0, log=True)
    ppo.Settings.lr_value = trial.suggest_float("lr_v", 1.0e-6, 1.0, log=True)
    ppo.Settings.c_entropy = trial.suggest_float(
        "c_entropy", 0.001, 1.0, log=True)
    ppo.Settings.c_value = trial.suggest_float(
        "c_value", 0.001, 1.0, log=True)
    ppo.Settings.split_pi_and_v_nets = trial.suggest_categorical("split_pi_and_vi_nets", [False, True])
    if ppo.Settings.split_pi_and_v_nets:
        ppo.Settings.c_value = 0.0
    ppo.Settings.write_videos = False
    ppo.Settings.horizon = 128
    ppo.Settings.sample_frames = 32
    ppo.Settings.num_actors = 100

    def env_fn(): return gym.make(args.env, full_action_space=False)
    net = AtariNet(env_fn())
    pre_fn = AtariPre
    trainer = ppo.PPO(env_fn, net, device="cuda",
                      prepare_fn=pre_fn, first_step=args.load_step)

    last_rewards = []

    for i in range(args.num_steps):
        rewards, loss = trainer.train()
        last_rewards.append(rewards)

        if i % 100 == 0:
            print(ppo.Settings.lr, ppo.Settings.c_value,
                  ppo.Settings.c_entropy, i, np.mean(last_rewards), loss)
            last_rewards.append(rewards)

            trial.report(float(np.mean(last_rewards)), i)

            if trial.should_prune():
                trainer.stop()
                raise optuna.TrialPruned()

            last_rewards = []

    trainer.stop()

    return float(np.mean(last_rewards))


def tune():
    study = optuna.create_study(
        study_name="atari",
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        storage="sqlite:///optuna.sqlite3",
        load_if_exists=True)
    study.optimize(ppo_objective, n_trials=100)
    print(study.best_trial)


def main():
    if False:
        tune()
        return

    def env_fn(): return gym.make(args.env, full_action_space=False)
    net = AtariNet(env_fn())
    pre_fn = AtariPre

    if args.load_step > 0:
        print("loading weights")
        net.load_state_dict(torch.load("step_%06d" % args.load_step))

    if args.algo == "ppo":
        ppo.Settings.lr = 0.001
        ppo.Settings.horizon = 128
        ppo.Settings.num_actors = 16
        ppo.Settings.c_value = 0.01
        ppo.Settings.c_entropy = 0.01
        ppo.Settings.split_pi_and_v_nets = True
        trainer = ppo.PPO(env_fn, net, device="cuda",
                          prepare_fn=pre_fn, first_step=args.load_step)
    elif args.algo == "dql":
        dql.Settings.lr = args.lr
        trainer = dql.DQL(env, net, device="cuda", prepare=pre)
    else:
        print("unkonwn algorithm", args.algo)
        sys.exit(1)

    pb = tqdm(range(args.load_step, args.num_steps))

    for i in pb:
        magic = (i % 1000 == 0)

        rewards, loss = trainer.train(render=magic)

        pb.set_description("%8d %6.6f %6.6f" % (i, rewards, loss))

        if magic:
            trainer.write_video(filename="step_%06d.avi" % i)
            torch.save(net.state_dict(), "step_%06d" % i)


if __name__ == '__main__':
    main()
