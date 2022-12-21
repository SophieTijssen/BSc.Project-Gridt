import networkx as nx
from enum import Enum
from random import sample


class NetworkType(Enum):
  """
  Enumerate class for the different network types.
  """
  UNDIRECTED = 0
  DIRECTED = 1


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
