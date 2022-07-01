# set mpi_yield_when_idle = 1 in /etc/openmpi/openmpi-mca-params.conf
# check with ompi_info --param all all -l 9 | grep mpi_yield_when_idle
# (it should be set to true)


import os

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


class Worker:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        os.environ["SDL_VIDEODRIVER"] = "dummy"

        print("sampler %d/%d" % (self.rank, self.size))

    def run(self):
        environments = [gym.make(Settings.environment_name)
                        for i in range(Settings.environments_per_worker)]
        observations = np.concatenate([np.expand_dims(env.reset(), 0) for env in environments])
        actions = np.zeros(shape=Settings.environments_per_worker)

        while True:
            print("send obs")
            observations = self.comm.Gather(observations, None, root=0)
            print("sent obs")

            print("read actions")
            actions = self.comm.scatter(actions, root=0)
            print("read actions", actions)

            for i in range(Settings.environments_per_worker):
                observations[i] = environments[i].step(actions[i])


class ParallelSampler:
    def __init__(self):
        self.comm = MPI.COMM_WORLD

    def receive_observations(self):
        return self.comm.gather(None, root=0)

    def send_actions(self, actions):
        self.comm.scatter(actions, root=0)


class TestParallelSampler(unittest.TestCase):
    def test_cartpole(self):
        if is_worker():
            return Worker().run()

        ps = ParallelSampler()

        for j in range(100):
            observations = ps.receive_observations()
            print(observations)

            actions = [[i % 2 for i in range(Settings.environments_per_worker)] for i in range(num_workers())]
            ps.send_actions(actions)
