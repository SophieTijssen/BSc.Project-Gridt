import numpy as np

from results.plot_graphs import createGranovetterBoxplots
from runs.run_types import singleRun, batchRunGranovetter
from utilities.model_util import RunType
from utilities.network_util import NetworkType, KnowledgeType, NetworkData
from utilities.threshold_util import DistributionType


path_figure = 'results/figures/granovetter/'
path_data = 'results/raw_data/granovetter/'


def runUniformExperiments(n, out_degree):
  # Uniform distribution
  singleRun(RunType.Granovetter.value, n, NetworkType.Directed.value, KnowledgeType.Network.value,
            DistributionType.UNIFORM.value, None, None, None, None, out_degree, NetworkData(), path_figure,
            'granovetter_uniform')

  # Manipulated uniform distribution
  singleRun(RunType.Granovetter.value, n, NetworkType.Directed.value, KnowledgeType.Network.value,
            DistributionType.UNIFORM_MODIFIED.value, None, None, None, None, out_degree, NetworkData(), path_figure,
            'granovetter_manipulated_uniform')


def runGranovetterModel(n, i, mu, out_degree):
  """
  Run function for running a simple Granovetter model and replicating the Granovetter's results.
  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param out_degree: The out-degree of each node in the network.
  """
  # Uniform distribution experiments
  runUniformExperiments(n, out_degree)

  # # Normal distribution with varying sigmas experiment
  # sigmas = np.linspace(0.0, 2.0, 201).round(decimals=2)
  # results = batchRunGranovetter(n, i, NetworkType.Directed.value, DistributionType.NORMAL.value, mu, sigmas, out_degree)
  # results.to_csv(path_data + 'sigma.csv')
  #
  # # sigmaPlot(path_figure, results)
  # createGranovetterBoxplots(path_figure, results)
