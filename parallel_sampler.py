
import os
import sys
import pickle

import gym
from mpi4py import MPI
import imageio.v2 as iio

import unittest

from basic import MarsRoverEnv


def create_env(env_name):
    if env_name == "MarsRover":
        return MarsRoverEnv()

    return gym.make(env_name)


class Worker:
    def __init__(self):
        self.comm = MPI.Comm.Get_parent()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.env_name = sys.argv[1]
        self.env = create_env(self.env_name)
        self.clip = None

        os.environ["SDL_VIDEODRIVER"] = "dummy"

        print("worker %d/%d: %s" % (self.rank, self.size, self.env_name))

    def run(self):
        while True:
            command, args = self.comm.recv(source=0, tag=1)

            if command == "reset":
                observation = self.env.reset()
                self.comm.send((observation, 0.0, False), dest=0, tag=2)
                self.render()
            elif command == "step":
                action = args
                new_observation, reward, done, _ = self.env.step(action)
                self.comm.send((new_observation, reward, done), dest=0, tag=2)
                self.render()
            elif command == "quit":
                break
            elif command == "start_render":
                self.clip = iio.get_writer(uri=args, fps=60)
            elif command == "stop_render":
                self.clip.close()
                self.clip = None
            else:
                print("INVALID COMMAND", command)
                return

    def render(self):
        if self.clip:
            self.clip.append_data(self.env.render(mode="rgb_array"))


class Wrapper:
    def __init__(self, comm, index):
        self.comm = comm
        self.index = index

    def send(self, message):
        self.comm.send(message, dest=self.index, tag=1)

    def send_reset(self):
        self.send(("reset", None))

    def send_step(self, action):
        self.send(("step", action))

    def send_quit(self):
        self.send(("quit", None))

    def send_start_render(self, filename):
        self.send(("start_render", filename))

    def send_stop_render(self):
        self.send(("stop_render", None))

    def receive(self):
        new_observation, reward, done = self.comm.recv(
            source=self.index, tag=2)
        return new_observation, reward, done


class ParallelSampler:
    def __init__(self, env_name, num_workers):
        MPI.INFO_ENV.Set("hostfile", "hostfile")

        self.comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['./parallel_sampler.py', env_name],
            maxprocs=num_workers)

    def env(self, index):
        return Wrapper(self.comm, index)


class TestParallelSampler(unittest.TestCase):
    def test_cartpole(self):
        ps = ParallelSampler("CartPole-v1", 4)

        for i in range(4):
            ps.env(i).send_reset()

        for i in range(4):
            new_observation, _, _ = ps.env(i).receive()

        ps.env(0).send_start_render("test_parallel_sampler.mp4")

        for j in range(100):
            for i in range(4):
                ps.env(i).send_step(j % 2)

            for i in range(4):
                new_observation, reward, done = ps.env(i).receive()

                if done:
                    print("worker %d / step %d -- done" % (i, j))
                    ps.env(i).send_reset()
                    new_observation, _, _ = ps.env(i).receive()

        ps.env(0).send_stop_render()


def main():
    Worker().run()


if __name__ == '__main__':
    main()
