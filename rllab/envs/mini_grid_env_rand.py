import numpy as np
from collections import namedtuple
from gym.utils import seeding
from .base import Env
from rllab.spaces import Discrete
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from gym_minigrid.envs.mazeEnv import MazeEnv

Env_config = namedtuple('Env_config', [
    'name',
    'lava_prob',
    'obstacle_lvl',
    'box_to_ball_prob',
    'door_prob',
    'wall_prob',
])

DEFAULT_ENV = Env_config(
        name='default_env',
        lava_prob=[0., 0.],
        obstacle_lvl=[0., 1.],
        box_to_ball_prob=[0., 0.3],
        door_prob=[0., 0.3],
        wall_prob=[0., 0.])

class MiniGridEnvRand(Env, Serializable):
    
    def __init__(self,genomes=None):
        Serializable.quick_init(self, locals())
        self._maze = MazeEnv(size=2,limit=2)
        self._seed()
        if genomes is not None:
            self._genomes = genomes
            genome_num = np.random.randint(len(self._genomes),1)
            self.reset(genome_num)
        else:
            self._genomes = [DEFAULT_ENV]
            self.reset()

    def reset(self, reset_args=0):
        genome = self._genomes[reset_args]
        self._maze.lava_prob = self.np_random.uniform(*genome.lava_prob)
        self._maze.obstacle_level = self.np_random.uniform(*genome.obstacle_lvl)
        self._maze.box2ball = self.np_random.uniform(*genome.box_to_ball_prob)
        self._maze.door_prob = self.np_random.uniform(*genome.door_prob)
        self._maze.wall_prob = self.np_random.uniform(*genome.wall_prob)
        self._maze.reset()
        obs = self._maze.gen_obs()
        return np.append(obs['image'],obs['direction']).astype(np.int32)

    def _seed(self,seed=None):
        self.env_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_goals(self, num_goals):
        return np.random.randint(len(self._genomes), size=(num_goals,))

    def step(self, action):
        state, reward, done, _ = self._maze.step(action)
        state_flat_array = np.append(state['image'],state['direction']).astype(np.int32)
        return Step(observation=state_flat_array, reward=reward, done=done)

    @property
    def action_space(self):
        return Discrete(len(self._maze.actions)-1)

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=255,
            shape=(self._maze.agent_view_size*self._maze.agent_view_size*3+1,1)
        )

