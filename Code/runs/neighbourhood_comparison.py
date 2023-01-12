from results.plot_graphs import multipleRunPlot
from runs.run_types import *
from utilities.model_util import RunType
from utilities.threshold_util import Distribution
from utilities.network_util import NetworkType, NetworkData


def runNeighbourhoodComparison(n, i, mu, sigma, in_degree):
  """
  Run function for comparing a granovetter model and neighbourhood model on a directed network.

  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param sigma: The standard deviation of the threshold distribution.
  :param in_degree: The in-degree of each node in the network.
  """

  batch = True

  if batch:
    # results_general_gran = batchRunGranovetter(n, i, NetworkType.DIRECTED, Distribution.NORMAL,
    #                                            mu, sigma, in_degree)
    # results_general_gran.to_csv('results/raw_data/results_general_gran.csv')
    # multipleRunPlot(results_general_gran, max(results_general_gran['Step']) + 1, ' general granovetter')
    #
    # results_general = batchRunNeighbourhood(RunType.Neighbourhood, n, i, NetworkType.DIRECTED, False,
    #                                         Distribution.NORMAL, mu, sigma,
    #                                         in_degree)
    # results_general.to_csv('results/raw_data/results_general.csv')
    # multipleRunPlot(results_general, max(results_general['Step']) + 1, ' general')

    results = batchRunNeighbourhood(RunType.Neighbourhood, n, i, NetworkType.DIRECTED, [False, True], False,
                                    Distribution.NORMAL, mu, sigma,
                                    in_degree)
    results.to_csv('results/raw_data/neighbourhood_comparison.csv')

    results_whole_network = results[results['neighbourhood'] == False]
    results_whole_network.to_csv('results/raw_data/results_whole_network.csv')

    results_neighbourhood = results[results['neighbourhood']]
    results_neighbourhood.to_csv('results/raw_data/results_neighbourhood.csv')

    maxSteps = max(max(results_whole_network['Step']), max(results_neighbourhood['Step'])) + 1
    print(maxSteps)

    multipleRunPlot(results_whole_network, maxSteps, ' with network information available', 'whole_network')
    multipleRunPlot(results_neighbourhood, maxSteps, ' with neighbourhood information available', 'neighbourhood')

  else:
    networkData = NetworkData()
    networkData.createNewNetwork(NetworkType.DIRECTED, n, in_degree, Distribution.NORMAL, mu, sigma)

    singleRun(RunType.Neighbourhood, n, NetworkType.DIRECTED, False, False, Distribution.NORMAL, mu, sigma, in_degree,
              networkData, ' with network information available', 'whole_network')

    singleRun(RunType.Neighbourhood, n, NetworkType.DIRECTED, True, False, Distribution.NORMAL, mu, sigma, in_degree,
              networkData, ' with neighbourhood information available', 'neighbourhood')
