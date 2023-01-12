from results.plot_graphs import multipleRunPlot
from runs.run_types import *
from utilities.threshold_util import Distribution
from utilities.network_util import NetworkType, NetworkData


def runNetworkComparison(n, i, mu, sigma, in_degree):
  """
  Run function for comparing a neighbourhood model on a directed network and an undirected network.

  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param sigma: The standard deviation of the threshold distribution.
  :param in_degree: The in-degree of each node in the network.
  """

  batch = True

  # TODO: use the same network for directed as for undirected to get an accurate conversion
  if batch:
    results = batchRunNeighbourhood(RunType.NetworkComparison, n, i, [NetworkType.DIRECTED, NetworkType.UNDIRECTED], True, False, Distribution.NORMAL, mu, sigma, in_degree)
    results.to_csv('results/raw_data/network_comparison.csv')

    results_undirected = results[results['networkType'] == NetworkType.UNDIRECTED.value]
    results_undirected.to_csv('results/raw_data/results_undirected.csv')

    results_directed = results[results['networkType'] == NetworkType.DIRECTED.value]
    results_directed.to_csv('results/raw_data/results_directed.csv')

    maxSteps = max(max(results_undirected['Step']), max(results_directed['Step'])) + 1
    print(maxSteps)

    multipleRunPlot(results_undirected, maxSteps, ' in an undirected network')
    multipleRunPlot(results_directed, maxSteps, ' in a directed network')

  else:
    networkData = NetworkData()
    networkData.createNewNetwork(NetworkType.DIRECTED, n, in_degree, Distribution.NORMAL, mu, sigma)

    singleRun(RunType.NetworkComparison, n, NetworkType.DIRECTED, True, False, Distribution.NORMAL, mu, sigma, in_degree, networkData, ' in a directed network')

    networkData.convertNetwork()

    singleRun(RunType.NetworkComparison, n, NetworkType.UNDIRECTED, True, False, Distribution.NORMAL, mu, sigma, in_degree, networkData, ' in an undirected network')
