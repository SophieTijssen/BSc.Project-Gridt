import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

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
# run_types = [RunType.Granovetter, RunType.Neighbourhood, RunType.NetworkComparison]
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
    # results.to_csv('results/raw_data/results.csv')
    #
    # sigmaBoxPlot(results)

    # 1
    results_whole = batchRunNeighbourhood(n, iterations, [network_type], False, distribution, mu, sigmas, in_degree)
    sigmaBoxPlot(results_whole)

    # 2
    results_neighbours = batchRunNeighbourhood(n, iterations, [network_type], True, distribution, mu, sigmas, in_degree)
    sigmaBoxPlot(results_neighbours)


if __name__ == '__main__':
  # for run in run_types:
  #   main(run)

  main()

  # threshold = 0.7
  # probability = 0.5
  #
  # # TODO: Determine the amount of quantiles
  # x = np.linspace(0, 1, 11)
  # print(x)
  # # x = np.linspace(0, 1, 100)
  #
  # y1 = beta.pdf(x, 1+1, 3+1)
  # y2 = beta.pdf(x, 2+1, 2+1)
  # y3 = beta.pdf(x, 3+1, 1+1)
  # plt.plot(x, y1, "-", label='0.25%')
  # plt.plot(x, y2, "r--", label='0.5%')
  # plt.plot(x, y3, "g:", label='0.75%')
  # plt.axvline(x=threshold, linestyle='-', color='lightgray')
  # plt.legend()
  # plt.show()
  #
  # y4 = beta.cdf(x, 1 + 1, 3 + 1)
  # y5 = beta.cdf(x, 2 + 1, 2 + 1)
  # y6 = beta.cdf(x, 3 + 1, 1 + 1)
  # plt.plot(x, y4, "-", label='0.25%')
  # plt.plot(x, y5, "r--", label='0.5%')
  # plt.plot(x, y6, "g:", label='0.75%')
  # plt.axvline(x=threshold, linestyle='-', color='lightgray')
  # plt.legend()
  # plt.show()
  #
  # lower = min(np.where(x >= threshold)[0])
  # print(lower)
  # upper = np.where(x == 1.0)[0][0]
  #
  # for y in [y4, y5, y6]:
  #   p = y[upper] - y[lower]
  #   print(p)
  #
  #   if p >= probability:
  #     print('Yes')
  #   else:
  #     print('No')
