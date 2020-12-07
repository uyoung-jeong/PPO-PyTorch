import torch
"""
import torch.nn as nn
from torch.distributions import MultivariateNormal
"""
import numpy as np

# actions: [joint index to occlude. roitation of occluding object]
class DummyEnv():
    def __init__(self):
        self.action_dim = 17+1
        self.action_space = torch.tensor(np.array([[-1 for _ in range(17)] + [-0.1],
                                                   [1 for _ in range(17)] + [0.1]]))

        self.observation_space = torch.zeros(256)

        print(f"Using {type(self).__name__}")

    def reset(self):
        return self.observation_space

    def seed(self, seed):
        return

    def render(self):
        return

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action)

        # rescale action
        action = action * self.action_space[1,:]

        # make dummy state
        state = self.observation_space

        # make dummy reward
        dummy_gt = torch.tensor(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.05]))
        reward = torch.sum(torch.abs(dummy_gt - action)) # maximize MAE

        done = True

        return state, reward, done, None
