import os
import numpy as np

from results.plot_graphs import sigmaBoxPlot, generateFigures
from results.statistical_tests import statistical_tests
from runs.expansion_comparison import runExpansionComparison
from runs.knowledge_comparison import runKnowledgeComparison
from runs.granovetter import runGranovetterModel, runUniformExperiments
from runs.network_comparison import runNetworkComparison
from runs.run_types import singleRun, batchRunNeighbourhood
from utilities.model_util import getRuntypeName, RunType
from utilities.network_util import NetworkType, KnowledgeType, NetworkData
from utilities.threshold_util import DistributionType

import pandas as pd
from scipy.stats import norm, skewnorm
from scipy.stats import shapiro, wilcoxon
import pingouin as pt

# Hyperparameters
n = 100
iterations = 10
mu = 0.25
sigma = 0.2
a = 0.5
b = 0.5
out_degree = 3

path_figure = 'results/figures/'

# Run parameters
# run_types = [RunType.Granovetter]
# run_types = [RunType.KnowledgeComparison]
# run_types = [RunType.NetworkComparison]
# run_types = [RunType.KnowledgeComparison, RunType.NetworkComparison]
run_types = [RunType.Granovetter, RunType.KnowledgeComparison, RunType.NetworkComparison]
network_type = NetworkType.Directed.value
distribution = DistributionType.NORMAL.value

comparison_variable = ['knowledge', 'networkType']
independent_variable = ['out_degree', 'num_of_nodes', 'sigma']
dependent_variable = ['engagement_ratio', 'diffusion_rate']

paths_knowledge = ['results/figures/knowledge_comparison/', 'results/raw_data/knowledge_comparison/']
paths_network = ['results/figures/network_comparison/', 'results/raw_data/network_comparison/']


def main(run_type):
  """
  The main function used to run the project.
  """
  print("starting ", run_type, "...")

  file_path_normal = "results/statistical_analysis/" + getRuntypeName(run_type=run_type,
                                                                      return_type='folder') + "/statistical_analysis.txt"
  file_path_latex = "results/statistical_analysis/" + getRuntypeName(run_type=run_type,
                                                                     return_type='folder') + "/statistical_analysis_latex.txt"

  if run_type == RunType.Granovetter:
    # Replicate results found by Granovetter TODO: include citation.
    runGranovetterModel(n, iterations, mu, out_degree)

  elif run_type == RunType.KnowledgeComparison:
    # Compare network knowledge to neighbourhood knowledge.

    for file_path in [file_path_normal, file_path_latex]:
      if os.path.exists(file_path):
        os.remove(file_path)

      file = open(file_path, 'x')
      file.write("Statistical analysis for " + getRuntypeName(run_type=run_type, return_type='name'))
      file.write("\n----------------------------------------------------\n")
      file.write("----------------------------------------------------")
      file.close()

    runKnowledgeComparison((file_path_normal, file_path_latex), n, iterations, mu, sigma, a, b, out_degree)
    # runExpansionComparison(results_paths=paths_knowledge, stats_paths=(file_path_normal, file_path_latex),
    #                        run_type=run_type, n=n, i=iterations,  comparison_variable='knowledge',
    #                        network_type=NetworkType.Directed.value, knowledge_type=[KnowledgeType.Network.value,
    #                                                                                 KnowledgeType.Neighbourhood.value],
    #                        mu=mu, sigma=sigma, a=a, b=b, out_degree=out_degree)

  elif run_type == RunType.NetworkComparison:
    # Compare a directed network to an undirected network.
    for file_path in [file_path_normal, file_path_latex]:
      if os.path.exists(file_path):
        os.remove(file_path)

      file = open(file_path, 'x')
      file.write("Statistical analysis for " + getRuntypeName(run_type=run_type, return_type='name'))
      file.write("\n----------------------------------------------------\n")
      file.write("----------------------------------------------------")
      file.close()

    runNetworkComparison((file_path_normal, file_path_latex), n, iterations, mu, sigma, a, b, out_degree)
    # runExpansionComparison(results_paths=paths_network, stats_paths=(file_path_normal, file_path_latex),
    #                        run_type=run_type, n=n, i=iterations, comparison_variable='networkType',
    #                        network_type=[NetworkType.Directed.value, NetworkType.Undirected.value],
    #                        knowledge_type=KnowledgeType.Neighbourhood.value, mu=mu, sigma=sigma, a=a, b=b,
    #                        out_degree=out_degree)

  elif run_type == RunType.Single:
    # Single Run
    knowledge = KnowledgeType.Network.value

    singleRun(run_type, n, network_type, knowledge, distribution, mu, sigma, a, b, out_degree,
              NetworkData(), path=path_figure, filename='single_run')

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)

    # 1
    results_whole = batchRunNeighbourhood(run_type, n, iterations, network_type, False, False, distribution,
                                          mu, sigmas, (a, b), out_degree)
    sigmaBoxPlot(path_figure, results_whole)

    # 2
    results_neighbours = batchRunNeighbourhood(run_type, n, iterations, network_type, True, False, distribution,
                                               mu, sigmas, (a, b), out_degree)
    sigmaBoxPlot(path_figure, results_neighbours)

  print("finished ", run_type)


# def testPrintFunction(dist):
#   print(type(dist))
#   print(dist.dType)
#   print(dist.n)
#   if type(dist) == UniformDistribution:
#     print(dist.modification)
#     if dist.modification is not None:
#       print(dist.modification[0])
#       print(dist.modification[1])
#   elif type(dist) == NormalDistribution:
#     print(dist.mu)
#     print(dist.sigma)
#   elif type(dist) == BetaDistribution:
#     print(dist.a)
#     print(dist.b)
#   else:
#     print("no specific distribution")


# def test():
#   # Set the random seed for reproducibility
#   np.random.seed(123)
#
#   # Generate normally distributed data (dependent variable 1)
#   n1_pre = norm.rvs(loc=20, scale=5, size=50)
#   n1_post = norm.rvs(loc=25, scale=6, size=50)
#
#   # Generate skewed data (dependent variable 2)
#   n2_pre = skewnorm.rvs(a=-5, loc=20, scale=5, size=50)
#   n2_post = skewnorm.rvs(a=-5, loc=25, scale=6, size=50)
#
#   # Create a dictionary to store the data
#   data = {'N1_pre': n1_pre, 'N1_post': n1_post, 'N2_pre': n2_pre, 'N2_post': n2_post}
#
#   # Create a Pandas DataFrame from the dictionary
#   df = pd.DataFrame(data)
#
#   # Print the first few rows of the DataFrame
#   print(df.head())
#
#   # Check normality of N1 (pre-test)
#   stat, p = shapiro(df['N1_pre'])
#   print('N1 pre-test:', 'Statistics=%.3f, p=%.3f' % (stat, p))
#   if p > 0.05:
#     print('N1 pre-test data is normally distributed')
#   else:
#     print('N1 pre-test data is not normally distributed')
#
#   # Check normality of N1 (post-test)
#   stat, p = shapiro(df['N1_post'])
#   print('N1 post-test:', 'Statistics=%.3f, p=%.3f' % (stat, p))
#   if p > 0.05:
#     print('N1 post-test data is normally distributed')
#   else:
#     print('N1 post-test data is not normally distributed')
#
#   # Check normality of N2 (pre-test)
#   stat, p = shapiro(df['N2_pre'])
#   print('N2 pre-test:', 'Statistics=%.3f, p=%.3f' % (stat, p))
#   if p > 0.05:
#     print('N2 pre-test data is normally distributed')
#   else:
#     print('N2 pre-test data is not normally distributed')
#
#   # Check normality of N2 (post-test)
#   stat, p = shapiro(df['N2_post'])
#   print('N2 post-test:', 'Statistics=%.3f, p=%.3f' % (stat, p))
#   if p > 0.05:
#     print('N2 post-test data is normally distributed')
#   else:
#     print('N2 post-test data is not normally distributed')
#
#   # Subset the dataframe to include only the n2 variable and pre/post-test measures
#   n2_data = df[['N2_pre', 'N2_post']]
#
#   # Carry out the Wilcoxon signed-rank test on the n2 variable
#   stat, p = wilcoxon(n2_data['N2_pre'], n2_data['N2_post'])
#
#   # Print the test statistic and p-value
#   print("Wilcoxon signed-rank test for n2:")
#   print(f"Statistic: {stat}")
#   print(f"p-value: {p}")
#
#   pg_wilcoxon_test = pt.wilcoxon(n2_data['N2_pre'], n2_data['N2_post'])
#   print(pg_wilcoxon_test.to_string())


if __name__ == '__main__':
  for run in run_types:
    main(run)
  # # run, n, network, knowledge, distribution, mu, sigma, a, b, out_degree, networkData, path, filename):
  # singleRun(run=RunType.KnowledgeComparison, n=20, network=NetworkType.Directed.value,
  #           knowledge=KnowledgeType.Neighbourhood.value, distribution=DistributionType.NORMAL.value, mu=mu, sigma=sigma,
  #           a=None, b=None, out_degree=out_degree, networkData=NetworkData(), path=path_figure, filename='test')

# TODO: Change the plots in the [run].py files to the plots that are used in the generateFigures() method

# for run in run_types:
#   if run == RunType.Granovetter:
#     # Uniform distribution experiments
#     runUniformExperiments(n, out_degree)
#
#   generateFigures(run)

# for run in run_types:
#   statistical_tests(run)

# calculateBetaDistribution()
# calculateParetoDistribution()
# test()
