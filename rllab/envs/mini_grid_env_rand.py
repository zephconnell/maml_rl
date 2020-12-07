import numpy as np
from .base import Env
from rllab.spaces import Discrete
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from gym_minigrid.gym_minigrid.envs.mazeEnv import MazeEnv

class MiniGridEnvRand(Env, Serializable):
    
    def __init__(self,
                 lava_prob=0, # float(0-0.2); chance to add lava float(0-0.2)
                 obstacle_level=0, # float(0-5); increase obstacles
                 box2ball=0, # flaot(0-1) chance to convert box to ball
                 door_prob=0, # float(0-0.5); chance to add doors
                 wall_prob=1, # float(0-1); cahnce to keep wall
                 seed=0
    ):
        Serializable.quick_init(self, locals())
        #TODO: get list of genomes and sample from uniform distribution of 1 to get gene values
        self._maze = MazeEnv(lava_prob=lava_prob,obstacle_level=obstacle_level,box2ball=box2ball,door_prob=door_prob,wall_prob=wall_prob,seed=seed)


    def reset(self):
        self._maze.lava_prob = 0 #TODO
        self._maze.obstacle_level = 0 #TODO
        self._maze.box2ball = 0 #TODO
        self._maze.door_prob = 0 #TODO
        self._maze.wall_prob = 1 #TODO
        self._maze.seed = 0 #TODO
        self._maze.reset()

        return self._maze.gen_obs()

    def step(self, action):
        state, reward, done = self._maze.step(action)
        return Step(observation=state, reward=reward, done=done)

    @property
    def action_space(self):
        return self._maze.action_space

    @property
    def observation_space(self):
        return self._maze.observation_space

