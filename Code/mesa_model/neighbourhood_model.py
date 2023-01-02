from mesa import DataCollector
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
# import numpy as np

from mesa_model.granovetter_model import GranovetterModel
from mesa_model.neighbourhood_agent import NeighbourhoodAgent
from utilities.model_util import *
from utilities.network_util import *


class NeighbourhoodModel(GranovetterModel):

  def __init__(self, run, num_of_nodes, networkType, neighbourhood, distributionType, mu, sigma, in_degree, networkData):
    """
    Initialisation of the model.

    :param run:
    :param num_of_nodes: Number of agents (or nodes) in the network.
    :param networkType: The type of network used for the model (directed/undirected).
    :param neighbourhood: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
    :param distributionType: The type of distribution used to sample the agent thresholds.
    :param mu: The mean of the threshold distribution.
    :param sigma: The standard deviation of the threshold distribution.
    :param in_degree: The in-degree of each node in the network.
    :param networkData: A class containing data about the network and agent thresholds.
    """
    super().__init__(num_of_nodes, networkType, distributionType, mu, sigma, in_degree)

    # Initialization
    self.seed = 13648  # TODO: Use random seed?? Or use same seed for every iteration?
    self.schedule = SimultaneousActivation(self)
    self.cooperating = 0.0
    self.networkData = networkData
    self.running = True

    # Create network (and grid) with set in-degree and random out-degree.
    if self.networkData.network is None:
      # We do not have a previously used network
      self.networkData.createNewNetwork(networkType, self.num_of_nodes, in_degree, distributionType, mu, sigma)
      print("First network created")

    else:

      if run == RunType.Neighbourhood.value:
        # We want to create a new network for each iteration (when neighbourhood is False)
        if not neighbourhood:
          self.networkData.createNewNetwork(networkType, self.num_of_nodes, in_degree, distributionType, mu, sigma)
          print(neighbourhood, ": New network created where whole network is visible to agents")
        else:
          print(neighbourhood, ": Using the existing network but only the neighbourhood is visible to agents")

      elif run == RunType.NetworkComparison.value:
        if networkType == NetworkType.DIRECTED.value:
          # We want to create a new directed network
          self.networkData.createNewNetwork(networkType, self.num_of_nodes, in_degree, distributionType, mu, sigma)
          print(networkType, ": Directed network created")

        else:
          # Convert previously used directed network to an undirected network
          self.networkData.convertNetwork()
          print(networkType, ": Network converted to undirected network")

    self.G = networkData.network
    self.grid = NetworkGrid(self.G)

    # Create agent thresholds.
    self.thresholds = networkData.thresholds

    # Create agents
    for node in list(self.G.nodes()):
      agent = NeighbourhoodAgent(node, self, neighbourhood, State.DEFECT, self.thresholds[node])
      self.schedule.add(agent)
      self.grid.place_agent(agent, node)

    self.datacollector = DataCollector(
      model_reporters={"engagement_ratio": get_engagement_ratio},
      agent_reporters={"state": "state.value"}
    )

  def step(self):
    """
    A single step of the model.
    The step function of all the agents is activated in the order
    specified by the scheduler and data is collected by the DataCollector.
    """
    self.datacollector.collect(self)
    self.schedule.step()

    # Stop the model if all agents are cooperating
    if number_cooperating(self) == self.schedule.get_agent_count():
      self.running = False
    elif number_cooperating(self) == self.cooperating:
      self.running = False

    self.cooperating = number_cooperating(self)
