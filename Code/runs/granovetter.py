from utilities.threshold_util import Distribution
from runs.run_types import *


path_figure = 'results/figures/granovetter/'
path_data = 'results/raw_data/granovetter/'


def runGranovetterModel(n, i, mu, in_degree):
  """
  Run function for running a simple Granovetter model and replicating the Granovetter's results.
  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param in_degree: The in-degree of each node in the network.
  """

  # Uniform distribution
  singleRun(RunType.Granovetter.value, n, NetworkType.Directed.value, KnowledgeType.Network.value,
            Distribution.UNIFORM.value, None, None, in_degree, NetworkData(), path_figure, 'granovetter_uniform')

  # Manipulated uniform distribution
  singleRun(RunType.Granovetter.value, n, NetworkType.Directed.value, KnowledgeType.Network.value,
            Distribution.UNIFORM_MODIFIED.value, None, None, in_degree, NetworkData(), path_figure,
            'granovetter_manipulated_uniform')

  # Varying sigmas test
  sigmas = np.linspace(0.0, 2.0, 201).round(decimals=2)
  results = batchRunGranovetter(n, i, NetworkType.Directed.value, Distribution.NORMAL.value, mu, sigmas, in_degree)
  results.to_csv(path_data + 'sigma.csv')

  # print(results.groupby(by=['sigma'])[['engagement_ratio']].mean())
  # print(results.groupby(by=['sigma'])[['engagement_ratio']].std())

  sigmaPlot(path_figure, results)
