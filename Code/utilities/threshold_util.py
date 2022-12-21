import numpy as np
from enum import Enum


class Distribution(Enum):
  """
  Enumerate class for the different distributions
  that can be used to sample the agent thresholds.
  """
  NORMAL = 0
  UNIFORM = 1
  UNIFORM_MODIFIED = 2


def createThresholds(n, mu, sigma):
  """
  Sample agent thresholds from a normal distribution.

  :param n: Number of agents in the model.
  :param mu: Mean value of the normal distribution
  :param sigma: Standard deviation of the normal distribution
  :return: Numpy array of agent thresholds.
  """
  thresholds = np.random.normal(mu, sigma, n)

  thresholds[thresholds > 1.0] = 1.0
  thresholds[thresholds < 0.0] = 0.0

  return thresholds
