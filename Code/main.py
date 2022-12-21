from utilities.network_util import NetworkType
from utilities.threshold_util import Distribution
from runs.run_granovetter import runGranovetterModel
from runs.network_comparison import runNetworkComparison
from runs.run_types import *

n = 100
iterations = 100
mu = 0.25
sigma = 0.2
in_degree = 3
run_type = RunType.NetworkComparison
network_type = NetworkType.DIRECTED
distribution = Distribution.NORMAL


def main():
  """
  The main function used to runs the project.
  """

  if run_type == RunType.Granovetter:
    # Replicate results found by Granovetter TODO: include citation.
    runGranovetterModel(n, iterations, mu, in_degree)

  elif run_type == RunType.NetworkComparison:
    # Compare a directed network to an undirected network.
    runNetworkComparison(n, iterations, mu, sigma, in_degree)

  elif run_type == RunType.Single:
    # Single Run
    singleRun(n, network_type, run_type, distribution, mu, sigma, in_degree)

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)
    results = batchRun(n, iterations, network_type, run_type, distribution, mu, sigmas, in_degree)

    results.to_csv('results/results.csv')

    sigmaBoxPlot(results)


if __name__ == '__main__':
  main()
