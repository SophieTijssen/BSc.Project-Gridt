from results.plot_graphs import createMultipleBoxplots, boxplotComparison
from runs.run_types import *
from utilities.model_util import RunType, sigmas1, sigmas2, nums, out_degrees, beta_parameters
from utilities.threshold_util import DistributionType
from utilities.network_util import NetworkType, NetworkData, KnowledgeType
from results.statistical_tests import performStatisticalAnalysis, t_tests, single_paired_t_test

path_figure = 'results/figures/knowledge_comparison/'
path_data = 'results/raw_data/knowledge_comparison/'


def runKnowledgeComparison(file_paths, n, i, mu, sigma, a, b, out_degree):
  """
  Run function for comparing a granovetter model and neighbourhood model on a directed network.

  :param file_paths:
  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param sigma: The standard deviation of the threshold distribution.
  :param a:
  :param b:
  :param out_degree: The out-degree of each node in the network.
  """

  # Run standard comparison simulations on normal distribution
  print("Starting 'standard simulation on normal distribution'...")
  results_normal = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=n, i=i,
                                         network=NetworkType.Directed.value,
                                         knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                         distribution=DistributionType.NORMAL.value, mu=mu, sigma=sigma,
                                         beta_parameters=[(None, None)], out_degree=out_degree, collectionPeriod=-1)
  results_normal.to_csv(path_data + 'knowledge_comparison_normal.csv')

  boxplotComparison(path_figure, results_normal, 'knowledge', 'engagement_ratio')  # , 'Equilibrium number of agents')
  boxplotComparison(path_figure, results_normal, 'knowledge', 'diffusion_rate')  # , 'Diffusion of behaviour')

  single_paired_t_test(file_paths=file_paths, results=results_normal, comparison_variable='knowledge')

  # # Run standard comparison simulations on Beta distribution
  # print("Starting 'standard simulation on Beta distribution'...")
  # results_beta = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=n, i=i,
  #                                      network=NetworkType.Directed.value,
  #                                      knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
  #                                      distribution=DistributionType.BETA.value, mu=None, sigma=None,
  #                                      beta_parameters=[(a, b)], out_degree=out_degree, collectionPeriod=-1)
  # results_beta.to_csv(path_data + 'knowledge_comparison_beta.csv')
  #
  # boxplotComparison(path_figure, results_beta, 'knowledge', 'engagement_ratio')  # , 'Equilibrium number of agents')
  # boxplotComparison(path_figure, results_beta, 'knowledge', 'Step')  # , 'Diffusion of behaviour')

  # # Alternating beta distribution parameters
  # print("Starting 'varying b'...")
  # results_b = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=n, i=i, network=NetworkType.Directed.value,
  #                                   knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
  #                                   distribution=DistributionType.BETA.value, mu=None, sigma=None,
  #                                   beta_parameters=beta_parameters, out_degree=out_degree, collectionPeriod=-1)
  # results_b.to_csv(path_data + 'b_comparison.csv')
  #
  # for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
  #   createMultipleBoxplots(results_b, 'knowledge', 'beta_parameters', dependent_variable,
  #                          filename='knowledge' + '_' + 'beta_parameters' + '_' + dependent_variable)
  #
  #   print('finished', 'knowledge', 'b', dependent_variable)

  # Alternate sigmas 10
  print("Starting 'varying sigma' 10...")
  results_sigma = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=n, i=i, network=NetworkType.Directed.value,
                                        knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                        distribution=DistributionType.NORMAL.value, mu=mu, sigma=sigmas1,
                                        beta_parameters=[(None, None)], out_degree=out_degree, collectionPeriod=-1)
  results_sigma.to_csv(path_data + 'sigma10_comparison.csv')

  for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
    createMultipleBoxplots(results=results_sigma, comparison_variable='knowledge', independent_variable='sigma',
                           dependent_variable=dependent_variable, path=path_figure,
                           filename='knowledge' + '_' + 'sigma' + '_' + dependent_variable, sigma_identifier='10')

    print('finished', 'knowledge', 'sigma', dependent_variable)

  performStatisticalAnalysis(file_paths=file_paths, data=results_sigma, comparison_variable='knowledge',
                             independent_variable='sigma', independent_values=sigmas1, file_identifier='10')

  # Alternate sigmas 100
  print("Starting 'varying sigma' 100...")
  results_sigma = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=n, i=i, network=NetworkType.Directed.value,
                                        knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                        distribution=DistributionType.NORMAL.value, mu=mu, sigma=sigmas2,
                                        beta_parameters=[(None, None)], out_degree=out_degree, collectionPeriod=-1)
  results_sigma.to_csv(path_data + 'sigma100_comparison.csv')

  for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
    createMultipleBoxplots(results=results_sigma, comparison_variable='knowledge', independent_variable='sigma',
                           dependent_variable=dependent_variable, path=path_figure,
                           filename='knowledge' + '_' + 'sigma' + '_' + dependent_variable, sigma_identifier='100')

    print('finished', 'knowledge', 'sigma', dependent_variable)

  performStatisticalAnalysis(file_paths=file_paths, data=results_sigma, comparison_variable='knowledge',
                             independent_variable='sigma', independent_values=sigmas2, file_identifier='100')

  # Alternating number of nodes
  print("Starting 'varying population size'...")
  results_n = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=nums, i=i, network=NetworkType.Directed.value,
                                    knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                    distribution=DistributionType.NORMAL.value, mu=mu, sigma=sigma,
                                    beta_parameters=[(None, None)], out_degree=out_degree, collectionPeriod=-1)

  results_n.to_csv(path_data + 'n_comparison.csv')

  for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
    createMultipleBoxplots(results=results_n, comparison_variable='knowledge', independent_variable='num_of_nodes',
                           dependent_variable=dependent_variable, path=path_figure,
                           filename='knowledge' + '_' + 'num_of_nodes' + '_' + dependent_variable)

    print('finished', 'knowledge', 'num_of_nodes', dependent_variable)

  performStatisticalAnalysis(file_paths=file_paths, data=results_n, comparison_variable='knowledge', independent_variable='num_of_nodes', independent_values=nums)

  # Alternating out-degrees
  print("Starting 'varying out-degree'...")
  results_degree = batchRunNeighbourhood(run=RunType.KnowledgeComparison.value, n=n, i=i, network=NetworkType.Directed.value,
                                         knowledge=[KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                         distribution=DistributionType.NORMAL.value, mu=mu, sigma=sigma,
                                         beta_parameters=[(None, None)], out_degree=out_degrees, collectionPeriod=-1)

  results_degree.to_csv(path_data + 'out-degree_comparison.csv')

  for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
    createMultipleBoxplots(results=results_degree, comparison_variable='knowledge', independent_variable='out_degree',
                           dependent_variable=dependent_variable, path=path_figure,
                           filename='knowledge' + '_' + 'out_degree' + '_' + dependent_variable)

    print('finished', 'knowledge', 'out_degree', dependent_variable)

  performStatisticalAnalysis(file_paths=file_paths, data=results_degree, comparison_variable='knowledge', independent_variable='out_degree', independent_values=out_degrees)
