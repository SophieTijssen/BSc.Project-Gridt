from runs.run_types import batchRun
from utilities.threshold_util import Distribution
from utilities.network_util import NetworkType


def runNetworkComparison(n, i, mu, sigma, in_degree):
  """
  Run function for comparing a neighbourhood model on a directed network and an undirected network.
  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param sigma: The standard deviation of the threshold distribution.
  :param in_degree: The in-degree of each node in the network.
  """
  results_undirected = batchRun(n, i, NetworkType.UNDIRECTED, True, Distribution.NORMAL, mu, sigma, in_degree)
  results_undirected.to_csv('results/results_undirected.csv')

  results_directed = batchRun(n, i, NetworkType.DIRECTED, True, Distribution.NORMAL, mu, sigma, in_degree)
  results_directed.to_csv('results/results_directed.csv')

  # TODO: Plot results
