from runs.run_types import *
from utilities.model_util import RunType, sigmas, nums, out_degrees
from utilities.threshold_util import DistributionType
from utilities.network_util import NetworkType, NetworkData


path_figure = 'results/figures/knowledge_comparison/'
path_data = 'results/raw_data/knowledge_comparison/'


def runKnowledgeComparison(n, i, mu, sigma, out_degree):
  """
  Run function for comparing a granovetter model and neighbourhood model on a directed network.

  :param n: The number of agents.
  :param i: The number of iterations.
  :param mu: The mean of the threshold distribution.
  :param sigma: The standard deviation of the threshold distribution.
  :param out_degree: The out-degree of each node in the network.
  """

  batch = True

  if batch:
    # Run standard comparison simulations
    results = batchRunNeighbourhood(RunType.KnowledgeComparison.value, n, i, NetworkType.Directed.value,
                                    [KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                    DistributionType.NORMAL.value, mu, sigma, out_degree, collectionPeriod=-1)
    results.to_csv(path_data + 'knowledge_comparison.csv')

    # Box plots
    boxplotComparison(path_figure, results, 'knowledge', 'engagement_ratio')  # , 'Equilibrium number of agents')
    boxplotComparison(path_figure, results, 'knowledge', 'Step')  # , 'Diffusion of behaviour')

    # Plot single run in one graph
    comparisonPlot(path_figure, results, 'knowledge_comparison', 'knowledge')

    # Alternate sigmas
    results_sigma = batchRunNeighbourhood(RunType.KnowledgeComparison.value, n, i, NetworkType.Directed.value,
                                          [KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                          DistributionType.NORMAL.value, mu, sigmas, out_degree, collectionPeriod=-1)

    results_sigma.to_csv(path_data + 'sigma_comparison.csv')

    sigma_network = results_sigma[results_sigma['knowledge'] == KnowledgeType.Network.value]
    sigma_neighbourhood = results_sigma[results_sigma['knowledge'] == KnowledgeType.Neighbourhood.value]

    multipleVariablesPlot(path_figure, (KnowledgeType.Network.value, sigma_network),
                          (KnowledgeType.Neighbourhood.value, sigma_neighbourhood),
                          'sigma', 'knowledge')

    # Alternating number of nodes
    results_n = batchRunNeighbourhood(RunType.KnowledgeComparison.value, nums, i, NetworkType.Directed.value,
                                      [KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                      DistributionType.NORMAL.value, mu, sigma, out_degree, collectionPeriod=-1)

    results_n.to_csv(path_data + 'n_comparison.csv')

    n_network = results_n[results_n['knowledge'] == KnowledgeType.Network.value]
    n_neighbourhood = results_n[results_n['knowledge'] == KnowledgeType.Neighbourhood.value]

    multipleVariablesPlot(path_figure, (KnowledgeType.Network.value, n_network),
                          (KnowledgeType.Neighbourhood.value, n_neighbourhood),
                          'num_of_nodes', 'knowledge')

      print('finished', 'knowledge', 'num_of_nodes', dependent_variable)

    # Alternating out-degrees
    results_degree = batchRunNeighbourhood(RunType.KnowledgeComparison.value, n, i, NetworkType.Directed.value,
                                           [KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value],
                                           DistributionType.NORMAL.value, mu, sigma, out_degrees, collectionPeriod=-1)

    results_degree.to_csv(path_data + 'out-degree_comparison.csv')

    degree_network = results_degree[results_degree['knowledge'] == KnowledgeType.Network.value]
    degree_neighbourhood = results_degree[results_degree['knowledge'] == KnowledgeType.Neighbourhood.value]

    multipleVariablesPlot(path_figure, (KnowledgeType.Network.value, degree_network),
                          (KnowledgeType.Neighbourhood.value, degree_neighbourhood),
                          'in_degree', 'knowledge')

  else:
    networkData = NetworkData()
    networkData.createNewNetwork(NetworkType.Directed, n, out_degree, DistributionType.NORMAL, mu, sigma)

    singleRun(RunType.KnowledgeComparison.value, n, NetworkType.Directed.value, KnowledgeType.Network.value,
              DistributionType.NORMAL.value, mu, sigma, out_degree, networkData, path_figure, 'whole_network')

    singleRun(RunType.KnowledgeComparison.value, n, NetworkType.Directed.value, KnowledgeType.Neighbourhood.value,
              DistributionType.NORMAL.value, mu, sigma, out_degree, networkData, path_figure, 'neighbourhood')
