import numpy as np
from .base import Env
from rllab.spaces import Discrete
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from gym_minigrid.envs.mazeEnv import MazeEnv

class MiniGridEnvRand(Env, Serializable):
    
    def __init__(self,genomes=None):
        Serializable.quick_init(self, locals())
        self._maze = MazeEnv()
        if genomes is not None:
            self._genomes = genomes
            genome_num = np.random.randint(len(self._genomes),1)
            self.reset(genome_num)
        else:
            self._genomes = [None]
            self.reset()


    def reset(self, reset_args=None):
        if reset_args is not None:
            genome = self._genomes[reset_args]
            #TODO: use the genome to set each gene to a value sampled from the provided ranges
            self._maze.lava_prob = 0
            self._maze.obstacle_level = 0
            self._maze.box2ball = 0
            self._maze.door_prob = 0
            self._maze.wall_prob = 1
            self._maze.seed = 0
            self._maze.reset()
        obs = self._maze.gen_obs()

        return np.append(obs['image'],obs['direction'])

    def sample_goals(self, num_goals):
        return np.random.randint(len(self._genomes), size=(num_goals,))

    def step(self, action):
        state, reward, done = self._maze.step(action)
        return Step(observation=state, reward=reward, done=done)

    @property
    def action_space(self):
        return Discrete(len(self._maze.actions)-1)

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=255,
            shape=((self._maze.agent_view_size, self._maze.agent_view_size, 3),1))

