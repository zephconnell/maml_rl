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
            self.reset(self.sample_goal())
        else:
            self._genomes = None
            self.reset()


    def reset(self, reset_args=None):
        if reset_args is not None:
            self._maze.lava_prob = reset_args[0]
            self._maze.obstacle_level = reset_args[1]
            self._maze.box2ball = reset_args[2]
            self._maze.door_prob = reset_args[3]
            self._maze.wall_prob = reset_args[4]
            self._maze.seed = reset_args[5]
        self._maze.reset()

        return self._maze.gen_obs()

    def sample_goals(self, num_goals):
        goals = []
        for i in range(num_goals):
            goals.append(self.sample_goal())
        return goals

    def sample_goal(self):
        if self._genomes is not None:
            #TODO: modify the goal to be a sample from list of genomes
            goal = [0,0,0,0,1,0]
        else:
            goal = None
        return goal

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
            shape=(self._maze.agent_view_size, self._maze.agent_view_size, 3))

