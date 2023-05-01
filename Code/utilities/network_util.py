import networkx as nx
from enum import Enum
from random import sample

from utilities.threshold_util import createThresholds, DistributionType


class NetworkType(Enum):
  """
  Enumerate class for the different network types.
  """
  Undirected = 0
  Directed = 1


class KnowledgeType(Enum):
  """
  Enumerate class for the different network types.
  """
  Network = False
  Neighbourhood = True


def getComparisonValue(comparison_variable):
  if comparison_variable == 'knowledge':
    return [KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value]
  elif comparison_variable == 'networkType':
    return [NetworkType.Undirected.value, NetworkType.Directed.value]


class NetworkData:
  def __init__(self):
    """

    """
    self.n = 0
    self.mu = None
    self.sigma = None
    self.network = None
    self.thresholds = []

  def createNewNetwork(self, networkType, n, out_degree, distributionType, mu, sigma):
    """

    :param networkType:
    :param n:
    :param out_degree:
    :param distributionType:
    :param mu:
    :param sigma:
    """
    self.n = n

    self.network = createDirectedNetwork(n, out_degree)

    if networkType == NetworkType.Undirected:
      self.convertNetwork()

    self.generateNewThresholds(distributionType, mu, sigma)

    # print("directed: ", self.network)

  def convertNetwork(self):
    """

    """
    # print("Convert network is called")
    self.network = convertToUndirectedNetwork(self.network)

  def generateNewThresholds(self, distributionType, mu, sigma):
    """

    :param distributionType:
    :param mu:
    :param sigma:
    """
    self.thresholds = createThresholds(distributionType, self.n, mu, sigma)

    if distributionType == DistributionType.NORMAL.value:
      self.mu = mu
      self.sigma = sigma


def constrained_sum_sample_pos(n, total):
  """
  Return a randomly chosen list of n positive integers summing to a total.
  Each such list is equally likely to occur.
  source: https://stackoverflow.com/a/3590105

  :param n: Number of nodes in the network.
  :param total: The sum of in-degrees of all nodes.
  :return: A list containing the ou-degrees for the nodes.
  """

  dividers = sorted(sample(range(1, total), n - 1))
  return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def createDirectedNetwork(n, out_degree):
  """
  Function for creating a directed network.

  :param n: Number of nodes in the network.
  :param out_degree: The number of neighbours of a node in the network.
  :return: A networkx DiGraph.
  """
  # in_degree_list = [in_degree] * n
  # out_degree_list = constrained_sum_sample_pos(n, sum(in_degree_list))

  out_degree_list = [out_degree] * n
  in_degree_list = constrained_sum_sample_pos(n, sum(out_degree_list))

  # TODO: Remember that parallel edges are not possible in the network (only one edge between two nodes) -> can there be an edge from a -> b if there is an edge from b -> a?
  G = nx.directed_configuration_model(
    in_degree_sequence=in_degree_list,
    out_degree_sequence=out_degree_list,
    create_using=nx.DiGraph,
    # seed=self.seed
  )
  G.remove_edges_from(nx.selfloop_edges(G))

  return G


def convertToUndirectedNetwork(G):
  """
  Convert a directed network to an undirected network.

  :param G: An undirected network.
  :return: An undirected networkx Graph.
  """

  H = G.to_undirected()
  # print("undirected: ", H)

  return H
