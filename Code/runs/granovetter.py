import numpy as np

from utilities.threshold_util import Distribution
from utilities.network_util import NetworkType, NetworkData
from runs.run_types import *
from results.plot_graphs import sigmaPlot


def runGranovetterModel(n, i, mu, in_degree):
  """
  Run function for running a simple Granovetter model and replicating the Granovetter's results.
  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param in_degree: The in-degree of each node in the network.
  """

  # Uniform distribution
  singleRun(RunType.Granovetter, n, NetworkType.DIRECTED, False, Distribution.UNIFORM, 0.0, 0.0, in_degree, NetworkData(), ' using a uniform distribution')

  # Manipulated uniform distribution
  singleRun(RunType.Granovetter, n, NetworkType.DIRECTED, False, Distribution.UNIFORM_MODIFIED, 0.0, 0.0, in_degree, NetworkData(), ' using a manipulated uniform distribution')

  # Varying sigmas test
  sigmas = np.linspace(0.0, 1.0, 101).round(decimals=2)
  results = batchRunGranovetter(n, i, NetworkType.DIRECTED, Distribution.NORMAL, mu, sigmas, in_degree)

  sigmaPlot(results)
