import numpy as np
import matplotlib.pyplot as plt

from enum import Enum


class Distribution:
  def __init__(self, dType, n):
    self.dType = dType
    self.n = n


class UniformDistribution(Distribution):
  def __init__(self, dType, n, modification=None):
    super().__init__(dType, n)
    self.modification = modification


class NormalDistribution(Distribution):
  def __init__(self, dType, n, mu, sigma):
    super().__init__(dType, n)
    self.mu = mu
    self.sigma = sigma


class BetaDistribution(Distribution):
  def __init__(self, dType, n, a, b):
    super().__init__(dType, n)
    self.a = a
    self.b = b


class DistributionType(Enum):
  """
  Enumerate class for the different distributions
  that can be used to sample the agent thresholds.
  """
  NORMAL = 0
  UNIFORM = 1
  UNIFORM_MODIFIED = 2
  BETA = 3
  NOTHINGSPECIFIC = 4


def createThresholds(distributionType, n, mu, sigma, a, b):
  """
  Sample agent thresholds from a normal distribution.

  :param distributionType: The type of distribution used to create/sample the thresholds.
  :param n: Number of agents in the model.
  :param mu: Mean value of the normal distribution
  :param sigma: Standard deviation of the normal distribution
  :param a:
  :param b:
  :return: Numpy array of agent thresholds.
  """

  if distributionType == DistributionType.NORMAL.value:  # We use a normal distribution.
    thresholds = np.random.normal(mu, sigma, n)

    thresholds[thresholds > 1.0] = 1.0
    thresholds[thresholds < 0.0] = 0.0
    # print("                   A normal distribution was used")

  elif distributionType == DistributionType.BETA.value:  # We use a beta distribution.
    thresholds = np.random.beta(a, b, n)

    plt.hist(thresholds, bins="auto", density=True)
    # # plt.plot(bins, scipy.special.comb(N, bins) * (P ** bins) * ((1 - P) ** (N - bins)), linewidth=2, color='r')
    plt.show()
    print("yes")

    thresholds[thresholds > 1.0] = 1.0
    thresholds[thresholds < 0.0] = 0.0

    # print("                   A beta distribution was used:", min(thresholds))

  else:  # We use a (modified) uniform distribution.
    thresholds = np.arange(0.0, 1.0, (1.0 / n))
    # print("A uniform distribution was used", end='')

    if distributionType == DistributionType.UNIFORM_MODIFIED.value:
      thresholds[thresholds == 0.01] = 0.02
      # print(" which was manipulated", end='')

    # print(thresholds)

  return thresholds


def createThresholds2(distribution, n):
  """
  Sample agent thresholds from a normal distribution.

  :param distribution: A distribution class containing information about the preferred distribution used to create/sample the thresholds.
  :param n: Number of agents in the model.
  :return: Numpy array of agent thresholds.
  """

  if type(distribution) == NormalDistribution:  # We use a normal distribution.
    thresholds = np.random.normal(distribution.mu, distribution.sigma, n)

    thresholds[thresholds > 1.0] = 1.0
    thresholds[thresholds < 0.0] = 0.0

  elif type(distribution) == UniformDistribution:  # We use a (modified) uniform distribution.
    thresholds = np.arange(0.0, 1.0, (1.0 / n))

    if distribution.modification is not None:
      thresholds[thresholds == distribution.modification[0]] = distribution.modification[1]
  else:
    thresholds = []

  return thresholds
