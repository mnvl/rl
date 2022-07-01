# set mpi_yield_when_idle = 1 in /etc/openmpi/openmpi-mca-params.conf
# check with ompi_info --param all all -l 9 | grep mpi_yield_when_idle
# (it should be set to true)


import os
import sys

import numpy as np
import gym
from mpi4py import MPI

import unittest


class Settings:
    environment_name = "CartPole-v1"
    environments_per_worker = 4


def is_root():
    return MPI.COMM_WORLD.Get_rank() == 0


def is_worker():
    return not is_root()


def num_workers():
    return MPI.COMM_WORLD.Get_size() - 1


def run_worker():
    if is_worker():
        Worker().run()
        return True
    return False


class Worker:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        os.environ["SDL_VIDEODRIVER"] = "dummy"

    def run(self):
        environments = [gym.make(Settings.environment_name)
                        for i in range(Settings.environments_per_worker)]
        observations = [env.reset() for env in environments]
        actions = np.zeros(
            shape=Settings.environments_per_worker, dtype=np.int32)

        num_frames = 0

        while True:
            self.comm.gather(observations, root=0)

            req = self.comm.Iscatter(None, actions, root=0)
            req.wait()

            if np.all(actions == -1):
                return

            for i in range(Settings.environments_per_worker):
                observations[i] = environments[i].step(actions[i])

            num_frames += 1


class ParallelSampler:
    def __init__(self):
        self.comm = MPI.COMM_WORLD

        self.observation_shape = gym.make(
            Settings.environment_name).reset().shape

        self.actions = np.zeros(
            shape=(num_workers()+1, Settings.environments_per_worker), dtype=np.int32)
        self.dummy = np.empty(Settings.environments_per_worker, dtype=np.int32)

    def receive_observations(self):
        return self.comm.gather(None, root=0)

    def send_actions(self, actions):
        self.actions[1:, :] = actions
        return self.comm.Iscatter(self.actions, self.dummy, root=0)


class TestParallelSampler(unittest.TestCase):
    def test_cartpole(self):
        if run_worker():
            return

        ps = ParallelSampler()

        actions = np.zeros(
            shape=(num_workers(), Settings.environments_per_worker), dtype=np.int32)

        for j in range(1000):
            observations = ps.receive_observations()

            req = ps.send_actions(actions)
            req.wait()

        actions[:] = -1
        req = ps.send_actions(actions)
        req.wait()
