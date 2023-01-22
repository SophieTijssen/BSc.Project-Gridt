from runs.knowledge_comparison import runKnowledgeComparison
from utilities.threshold_util import Distribution
from runs.granovetter import runGranovetterModel
from runs.network_comparison import runNetworkComparison
from runs.run_types import *

# Hyperparameters
n = 100
iterations = 1
mu = 0.25
sigma = 0.2
in_degree = 3

path_figure = 'results/figures/'

# Run parameters
# run_types = [RunType.Granovetter]
# run_types = [RunType.KnowledgeComparison]
# run_types = [RunType.NetworkComparison]
run_types = [RunType.Granovetter, RunType.KnowledgeComparison, RunType.NetworkComparison]
network_type = NetworkType.Directed.value
distribution = Distribution.NORMAL.value


def main(run_type):
  """
  The main function used to run the project.
  """

  if run_type == RunType.Granovetter:
    # Replicate results found by Granovetter TODO: include citation.
    runGranovetterModel(n, iterations, mu, in_degree)

  elif run_type == RunType.KnowledgeComparison:
    runKnowledgeComparison(n, iterations, mu, sigma, in_degree)

  elif run_type == RunType.NetworkComparison:
    # Compare a directed network to an undirected network.
    runNetworkComparison(n, iterations, mu, sigma, in_degree)

  elif run_type == RunType.Single:
    # Single Run
    knowledge = KnowledgeType.Network.value

    singleRun(run_type, n, network_type, knowledge, distribution, mu, sigma, in_degree,
              NetworkData(), path=path_figure, filename='single_run')

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)

    # 1
    results_whole = batchRunNeighbourhood(run_type, n, iterations, network_type, False, False, distribution,
                                          mu, sigmas, in_degree)
    sigmaBoxPlot(path_figure, results_whole)

    # 2
    results_neighbours = batchRunNeighbourhood(run_type, n, iterations, network_type, True, False, distribution,
                                               mu, sigmas, in_degree)
    sigmaBoxPlot(path_figure, results_neighbours)


if __name__ == '__main__':
  # main()

  for run in run_types:
    main(run)
