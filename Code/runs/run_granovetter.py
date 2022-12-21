import numpy as np

from utilities.threshold_util import Distribution
from utilities.network_util import NetworkType
from runs.run_types import singleRun, batchRun
from results.plot_graphs import SigmaPlot


def runGranovetterModel(n, i, mu, in_degree):
  """
  Run function for running a simple Granovetter model and replicating the Granovetter's results.
  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param in_degree: The in-degree of each node in the network.
  """

  # Uniform distribution
  singleRun(n, NetworkType.DIRECTED, False, Distribution.UNIFORM, 0.0, 0.0, in_degree)

  # Manipulated uniform distribution
  singleRun(n, NetworkType.DIRECTED, False, Distribution.UNIFORM_MODIFIED, 0.0, 0.0, in_degree)

  # Varying sigmas test
  sigmas = np.linspace(0.0, 1.0, 101).round(decimals=2)
  results = batchRun(n, i, NetworkType.DIRECTED, False, Distribution.NORMAL, mu, sigmas, in_degree)

  SigmaPlot(results)
