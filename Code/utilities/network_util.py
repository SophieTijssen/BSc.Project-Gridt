import networkx as nx
from enum import Enum
from random import sample

from utilities.threshold_util import createThresholds


class NetworkType(Enum):
  """
  Enumerate class for the different network types.
  """
  UNDIRECTED = 0
  DIRECTED = 1


class NetworkData:
  def __init__(self):
    """

    """
    self.n = 0
    self.network = None
    self.thresholds = []

  def createNewNetwork(self, networkType, n, in_degree, distributionType, mu, sigma):
    """

    :param networkType:
    :param n:
    :param in_degree:
    :param distributionType:
    :param mu:
    :param sigma:
    """
    self.n = n

    self.network = createDirectedNetwork(n, in_degree)

    if networkType == NetworkType.UNDIRECTED:
      self.convertNetwork()

    self.generateNewThresholds(distributionType, mu, sigma)

  def convertNetwork(self):
    """

    """
    self.network = convertToUndirectedNetwork(self.network)

  def generateNewThresholds(self, distributionType, mu, sigma):
    """

    :param distributionType:
    :param mu:
    :param sigma:
    """
    self.thresholds = createThresholds(distributionType, self.n, mu, sigma)


def constrained_sum_sample_pos(n, total):
  """
  Return a randomly chosen list of n positive integers summing to total.
  Each such list is equally likely to occur.
  source: https://stackoverflow.com/a/3590105

  :param n: Number of nodes in the network.
  :param total: The sum of in-degrees of all nodes.
  :return: A list containing the ou-degrees for the nodes.
  """

  dividers = sorted(sample(range(1, total), n - 1))
  return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def createDirectedNetwork(n, in_degree):
  """
  Function for creating a directed network.

  :param n: Number of nodes in the network.
  :param in_degree: The in-degree of a node in the network.
  :return: A networkx DiGraph.
  """
  in_degree_list = [in_degree] * n
  out_degree_list = constrained_sum_sample_pos(n, sum(in_degree_list))

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
  G.to_undirected()

  return G
