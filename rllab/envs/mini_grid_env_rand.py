import numpy as np
import random
from gym.utils import seeding
from .base import Env
from rllab.spaces import Discrete
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from gym_minigrid.envs.mazeEnv import MazeEnv

DEFAULT_GENOME = {
    'name':'default',
    'lava_prob':[0., 0.],
    'obstacle_lvl':[0., 1.],
    'box_to_ball_prob':[0., 0.3],
    'door_prob':[0., 0.3],
    'wall_prob':[0., 0.]
}

class MiniGridEnvRand(Env, Serializable):
    
    def __init__(self,configs=None,num_mazes=20):
        Serializable.quick_init(self, locals())

        self._maze = MazeEnv(size=2,limit=2)
        self.num_unique_mazes = num_mazes
        self.genomes = [None]*self.num_unique_mazes
        self.seeds = [None]*self.num_unique_mazes

        if configs is not None:
            if len(configs) > self.num_unique_mazes:
                extra = len(configs) - self.num_unique_mazes
                print("WARNING: The number of given configurations (",len(configs),") is greater than the number of unique mazes to be generated (",self.num_unique_mazes,"). ",extra," configurations will not be used.")
            for i in range(self.num_unique_mazes):
                if i < len(configs):
                    self.genomes[i] = configs[i]['config']
                    self.seeds[i] = configs[i]['seed']
                else:
                    j = i%len(configs)
                    self.genomes[i] = configs[j]['config']
                    self.seeds[i] = i
        else:
            self.genomes = [DEFAULT_GENOME]*self.num_unique_mazes
            for i in range(self.num_unique_mazes):
                self.seeds[i] = i

        genome_num = np.random.randint(0,self.num_unique_mazes)
        self.reset(genome_num)

    def reset(self, reset_args=None):
        if reset_args is None:
            reset_args = 0	
        genome = self.genomes[reset_args]
        self._maze.seed(self.seeds[reset_args])
        self._maze.lava_prob = self._maze.np_random.uniform(*genome['lava_prob'])
        self._maze.obstacle_level = self._maze.np_random.uniform(*genome['obstacle_lvl'])
        self._maze.box2ball = self._maze.np_random.uniform(*genome['box_to_ball_prob'])
        self._maze.door_prob = self._maze.np_random.uniform(*genome['door_prob'])
        self._maze.wall_prob = self._maze.np_random.uniform(*genome['wall_prob'])
        self._maze.reset()
        obs = self._maze.gen_obs()
        return np.append(obs['image'],obs['direction']).astype(np.int32)

    def sample_goals(self, num_goals):
        return np.random.randint(self.num_unique_mazes, size=(num_goals,))

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

