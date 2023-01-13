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


def createThresholds(distributionType, n, mu, sigma):
  """
  Sample agent thresholds from a normal distribution.

  :param distributionType: The type of distribution used to create/sample the thresholds.
  :param n: Number of agents in the model.
  :param mu: Mean value of the normal distribution
  :param sigma: Standard deviation of the normal distribution
  :return: Numpy array of agent thresholds.
  """

  if distributionType == Distribution.NORMAL.value:  # We use a normal distribution.
    thresholds = np.random.normal(mu, sigma, n)

    thresholds[thresholds > 1.0] = 1.0
    thresholds[thresholds < 0.0] = 0.0

  else:  # We use a (modified) uniform distribution.
    thresholds = np.arange(0.0, 1.0, (1.0 / n))

    if distributionType == Distribution.UNIFORM_MODIFIED.value:
      thresholds[thresholds == 0.01] = 0.02

  return thresholds
