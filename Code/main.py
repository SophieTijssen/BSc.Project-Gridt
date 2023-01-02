from runs.neighbourhood_comparison import runNeighbourhoodComparison
from utilities.model_util import RunType
from utilities.network_util import NetworkType
from utilities.threshold_util import Distribution
from runs.granovetter import runGranovetterModel
from runs.network_comparison import runNetworkComparison
from runs.run_types import *

# Hyperparameters
n = 100
iterations = 10
mu = 0.25
sigma = 0.2
in_degree = 3

# Run parameters
run_type = RunType.Neighbourhood
network_type = NetworkType.DIRECTED
distribution = Distribution.NORMAL


def main():
  """
  The main function used to run the project.
  """

  if run_type == RunType.Granovetter:
    # Replicate results found by Granovetter TODO: include citation.
    runGranovetterModel(n, iterations, mu, in_degree)

  elif run_type == RunType.Neighbourhood:
    runNeighbourhoodComparison(n, iterations, mu, sigma, in_degree)

  elif run_type == RunType.NetworkComparison:
    # Compare a directed network to an undirected network.
    runNetworkComparison(n, iterations, mu, sigma, in_degree)

    # sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)
    #
    # # 3
    # results_undirected = batchRun(n, iterations, NetworkType.UNDIRECTED, True, Distribution.NORMAL, mu, sigmas, in_degree)
    # sigmaBoxPlot(results_undirected)
    #
    # # 4
    # results_directed = batchRun(n, iterations, NetworkType.DIRECTED, True, Distribution.NORMAL, mu, sigmas, in_degree)
    # sigmaBoxPlot(results_directed)

  elif run_type == RunType.Single:
    # Single Run
    singleRun(n, network_type, run_type, distribution, mu, sigma, in_degree)

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)
    # results = batchRun(n, iterations, network_type, run_type, distribution, mu, sigmas, in_degree)
    #
    # results.to_csv('results/results.csv')
    #
    # sigmaBoxPlot(results)

    # 1
    results_whole = batchRunNeighbourhood(n, iterations, [network_type], False, distribution, mu, sigmas, in_degree)
    sigmaBoxPlot(results_whole)

    # 2
    results_neighbours = batchRunNeighbourhood(n, iterations, [network_type], True, distribution, mu, sigmas, in_degree)
    sigmaBoxPlot(results_neighbours)


if __name__ == '__main__':
  main()
