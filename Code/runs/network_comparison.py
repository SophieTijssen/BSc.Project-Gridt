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
    # Run simulations
    results = batchRunNeighbourhood(RunType.NetworkComparison, n, i, [NetworkType.DIRECTED, NetworkType.UNDIRECTED], True, False, Distribution.NORMAL, mu, sigma, in_degree)
    results.to_csv('results/raw_data/network_comparison.csv')

    # Plot multiple runs in one graph
    results_undirected = results[results['networkType'] == NetworkType.UNDIRECTED.value]
    results_undirected.to_csv('results/raw_data/results_undirected.csv')

    results_directed = results[results['networkType'] == NetworkType.DIRECTED.value]
    results_directed.to_csv('results/raw_data/results_directed.csv')

    maxSteps = max(max(results_undirected['Step']), max(results_directed['Step'])) + 1
    multipleRunPlot(results_undirected, maxSteps, 'in an undirected network', 'undirected_network')
    multipleRunPlot(results_directed, maxSteps, 'in a directed network', 'directed_network')

    # Plot single run in one graph
    comparisonPlot(results, 'network_comparison', 'networkType')

    # Alternate sigmas
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)
    results = batchRunNeighbourhood(RunType.NetworkComparison, n, i, [NetworkType.DIRECTED, NetworkType.UNDIRECTED],
                                    KnowledgeType.Neighbourhood.value, False, Distribution.NORMAL, mu, sigmas,
                                    in_degree, collectionPeriod=-1)

    results.to_csv('results/raw_data/network_sigma_comparison.csv')

    sigma_undirected = results[results['networkType'] == NetworkType.UNDIRECTED.value]
    sigma_directed = results[results['networkType'] == NetworkType.DIRECTED.value]

    multipleVariablesPlot(sigma_undirected, sigma_directed, 'sigma', ['Undirected', 'Directed'], 'sigma')
    # TODO: Plot effect on diffusion rate

    # Alternating number of nodes
    nums = range(80, 140, 5)
    results = batchRunNeighbourhood(RunType.NetworkComparison, nums, i, [NetworkType.DIRECTED, NetworkType.UNDIRECTED],
                                    KnowledgeType.Neighbourhood.value, False, Distribution.NORMAL, mu, sigma, in_degree,
                                    collectionPeriod=-1)

    results.to_csv('results/raw_data/network_n_comparison.csv')

    n_undirected = results[results['networkType'] == NetworkType.UNDIRECTED.value]
    n_directed = results[results['networkType'] == NetworkType.DIRECTED.value]

    multipleVariablesPlot(n_undirected, n_directed, 'num_of_nodes', ['Undirected', 'Directed'], 'n')
    # TODO: Plot effect on diffusion rate

  else:
    networkData = NetworkData()
    networkData.createNewNetwork(NetworkType.DIRECTED, n, in_degree, Distribution.NORMAL, mu, sigma)

    singleRun(RunType.NetworkComparison, n, NetworkType.DIRECTED, True, False, Distribution.NORMAL, mu, sigma, in_degree, networkData, 'in a directed network')

    networkData.convertNetwork()

    singleRun(RunType.NetworkComparison, n, NetworkType.UNDIRECTED, True, False, Distribution.NORMAL, mu, sigma, in_degree, networkData, 'in an undirected network')
