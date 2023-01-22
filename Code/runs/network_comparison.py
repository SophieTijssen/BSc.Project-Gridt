from runs.run_types import *
from utilities.model_util import RunType
from utilities.threshold_util import Distribution
from utilities.network_util import NetworkType, NetworkData


path_figure = 'results/figures/network_comparison/'
path_data = 'results/raw_data/network_comparison/'


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

  if batch:
    # Run standard comparison simulations
    results = batchRunNeighbourhood(RunType.NetworkComparison.value, n, i,
                                    [NetworkType.Directed.value, NetworkType.Undirected.value],
                                    KnowledgeType.Neighbourhood.value, Distribution.NORMAL.value, mu, sigma, in_degree)
    results.to_csv(path_data + 'network_comparison.csv')

    # Box plots
    boxplotComparison(path_figure, results, 'networkType', 'engagement_ratio')  # , 'Equilibrium number of agents')
    boxplotComparison(path_figure, results, 'networkType', 'Step')  # , 'Diffusion of behaviour')

    # Plot single run in one graph
    comparisonPlot(path_figure, results, 'network_comparison', 'networkType')

    # Alternate sigmas
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)
    results_sigma = batchRunNeighbourhood(RunType.NetworkComparison.value, n, i,
                                          [NetworkType.Directed.value, NetworkType.Undirected.value],
                                          KnowledgeType.Neighbourhood.value, Distribution.NORMAL.value,
                                          mu, sigmas, in_degree, collectionPeriod=-1)

    results_sigma.to_csv(path_data + 'sigma_comparison.csv')

    sigma_undirected = results_sigma[results_sigma['networkType'] == NetworkType.Undirected.value]
    sigma_directed = results_sigma[results_sigma['networkType'] == NetworkType.Directed.value]

    multipleVariablesPlot(path_figure, (NetworkType.Undirected.value, sigma_undirected),
                          (NetworkType.Directed.value, sigma_directed),
                          'sigma', 'networkType')

    # Alternating number of nodes
    nums = range(80, 140, 5)
    results_n = batchRunNeighbourhood(RunType.NetworkComparison.value, nums, i,
                                      [NetworkType.Directed.value, NetworkType.Undirected.value],
                                      KnowledgeType.Neighbourhood.value, Distribution.NORMAL.value,
                                      mu, sigma, in_degree, collectionPeriod=-1)

    results_n.to_csv(path_data + 'n_comparison.csv')

    n_undirected = results_n[results_n['networkType'] == NetworkType.Undirected.value]
    n_directed = results_n[results_n['networkType'] == NetworkType.Directed.value]

    multipleVariablesPlot(path_figure, (NetworkType.Undirected.value, n_undirected),
                          (NetworkType.Directed.value, n_directed),
                          'num_of_nodes', 'networkType')

    # Alternating in-degrees
    in_degrees = range(1, 11, 1)
    results_degree = batchRunNeighbourhood(RunType.NetworkComparison.value, n, i,
                                           [NetworkType.Directed.value, NetworkType.Undirected.value],
                                           KnowledgeType.Neighbourhood.value, Distribution.NORMAL.value,
                                           mu, sigma, in_degrees, collectionPeriod=-1)

    results_degree.to_csv(path_data + 'in-degree_comparison.csv')

    degree_undirected = results_degree[results_degree['networkType'] == NetworkType.Undirected.value]
    degree_directed = results_degree[results_degree['networkType'] == NetworkType.Directed.value]

    multipleVariablesPlot(path_figure, (NetworkType.Undirected.value, degree_undirected),
                          (NetworkType.Directed.value, degree_directed),
                          'in_degree', 'networkType')

  else:
    networkData = NetworkData()
    networkData.createNewNetwork(NetworkType.Directed, n, in_degree, Distribution.NORMAL, mu, sigma)

    singleRun(RunType.NetworkComparison.value, n, NetworkType.Directed.value, KnowledgeType.Neighbourhood.value,
              Distribution.NORMAL.value, mu, sigma, in_degree, networkData, path_figure, 'directed_network')

    networkData.convertNetwork()

    singleRun(RunType.NetworkComparison.value, n, NetworkType.Undirected.value, KnowledgeType.Neighbourhood.value,
              Distribution.NORMAL.value, mu, sigma, in_degree, networkData, path_figure, 'undirected_network')
