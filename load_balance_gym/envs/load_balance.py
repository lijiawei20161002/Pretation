from load_balance_gym.envs.param import config
from load_balance_gym.envs.job import Job
from load_balance_gym.envs.job_generator import generate_job
from load_balance_gym.envs.server import Server
from load_balance_gym.envs.timeline import Timeline
from load_balance_gym.envs.wall_time import WallTime

import gym
from gym import error, spaces, utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas
import numpy as np

class LoadBalanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def setup_space(self):
        self.obs_low = np.array([0] * (config.num_servers + 1))
        self.obs_high = np.array([config.load_balance_obs_high] * (config.num_servers +1))
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(config.num_servers)

    def np_random(self, seed=42):
        if not (isinstance(seed, int) and seed >= 0):
            raise ValueError('Seed must be a non-negative integer.')
        rng = np.random.RandomState()
        rng.seed(seed)
        return rng

    def seed(self, seed):
        self.np_random = self.np_random(seed)

    def contains(self, observation_space, x):
        return x.shape == observation_space.shape and (x>=observation_space.low).all() and (x<=observation_space.high).all()

    def __init__(self):
        self.setup_space()
        self.seed(config.seed)
        self.wall_time = WallTime()
        self.timeline = Timeline()
        self.num_stream_jobs = config.num_stream_jobs
        self.servers = self.initialize_servers(config.service_rates)
        self.incoming_job = None
        self.finished_jobs = []
        self.reset()

    def generate_job(self):
        if self.num_stream_jobs_left > 0:
            dt, size = generate_job(self.np_random)
            t = self.wall_time.curr_time
            self.timeline.push(t+dt, size)
            self.num_stream_jobs_left -= 1

    def initialize(self):
        assert self.wall_time.curr_time == 0
        self.generate_job()
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, int)
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_servers(self, service_rates):
        servers = []
        for server_id in range(config.num_servers):
            server = Server(server_id, service_rates[server_id], self.wall_time)
            servers.append(server)
        return servers

    def reset(self):
        for server in self.servers:
            server.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.num_stream_jobs_left = self.num_stream_jobs
        assert self.num_stream_jobs_left > 0
        self.incoming_job = None
        self.finished_jobs = []
        self.initialize()
        return self.observe()

    def observe(self):
        obs_arr = []
        for server in self.servers:
            load = sum(j.size for j in server.queue)
            if server.curr_job is not None:
                load += server.curr_job.finish_time - self.wall_time.curr_time
            if load > self.obs_high[server.server_id]:
                print('Server '+str(server.server_id)+' at time '+str(self.wall_time.curr_time)+' has load '+str(load)+' larger than obs_high '+str(self.obs_high[server.server_id]))
                load = self.obs_high[server.server_id]
            obs_arr.append(load)
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            if self.incoming_job.size > self.obs_high[-1]:
                print('Incoming job at time '+str(self.wall_time.curr_time)+' has size '+str(self.incoming_job.size)+' larger than obs_high '+str(self.obs_high[-1]))
                obs_arr.append(self.obs_high[-1])
            else:
                obs_arr.append(self.incoming_job.size)

        obs_arr = np.array(obs_arr)
        assert self.contains(self.observation_space, obs_arr)

        return obs_arr

    def step(self, action):
        assert self.action_space.contains(action)
        self.servers[action].schedule(self.incoming_job)
        running_job = self.servers[action].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)
        self.incoming_job = None
        self.generate_job()
        reward = 0
        while len(self.timeline) > 0:
            new_time, obj = self.timeline.pop()
            num_active_jobs = sum(len(w.queue) for w in self.servers)
            for server in self.servers:
                if server.curr_job is not None:
                    assert server.curr_job.finish_time >= \
                        self.wall_time.curr_time
                    num_active_jobs += 1
            reward -= (new_time - self.wall_time.curr_time) * num_active_jobs
            self.wall_time.update(new_time)

            if isinstance(obj, int):
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                break
            elif isinstance(obj, Job):
                job = obj
                if not np.isinf(self.num_stream_jobs_left):
                    self.finished_jobs.append(job)
                else:
                    if len(self.finished_jobs) > 0:
                        self.finished_jobs[-1] += 1
                    else:
                        self.finished_jobs = [1]
                if job.server.curr_job == job:
                    job.server.curr_job = None
                running_job = job.server.process()
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)
            else:
                print('illegal event type')
                exit(1)
        done = ((len(self.timeline) == 0) and \
            self.incoming_job is None)
        return self.observe(), reward, done, {'curr_time': self.wall_time.curr_time}