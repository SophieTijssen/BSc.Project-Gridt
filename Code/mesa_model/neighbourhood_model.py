from mesa import DataCollector

from mesa_model.granovetter_model import GranovetterModel
from mesa_model.neighbourhood_agent import NeighbourhoodAgent
from utilities.model_util import *
from utilities.network_util import *


class NeighbourhoodModel(GranovetterModel):

  def __init__(self, run, num_of_nodes, mu, sigma, out_degree, networkType, knowledge, distributionType, networkData):
    """
    Initialisation of the model.

    :param run:
    :param num_of_nodes: Number of agents (or nodes) in the network.
    :param networkType: The type of network used for the model (directed/undirected).
    :param knowledge: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
    :param distributionType: The type of distribution used to sample the agent thresholds.
    :param mu: The mean of the threshold distribution.
    :param sigma: The standard deviation of the threshold distribution.
    :param out_degree: The out-degree of each node in the network.
    :param networkData: A class containing data about the network and agent thresholds.
    """

    # Initialization
    self.run = run
    self.neighbourhood = knowledge
    self.networkData = networkData

    super().__init__(num_of_nodes, networkType, distributionType, mu, sigma, out_degree)

  def createDataCollector(self):
    datacollector = DataCollector(
      model_reporters={"engagement_ratio": calculate_engagement_ratio,
                       "diffusion_rate": calculate_diffusion_rate},
      # agent_reporters={"state": "state.value"}
    )
    return datacollector

  def generateNetwork(self):
    """
    Create network (and grid) with set in-degree and random out-degree.
    :return:
    """

    # We do not have a previously used network
    if self.networkData.network is None:
      # print("First network created")
      self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.out_degree, self.distributionType, self.mu, self.sigma)

    elif self.run == RunType.KnowledgeComparison.value:
      # We want to create a new network for each iteration (when neighbourhood is False)
      if not self.neighbourhood:
        # print(self.neighbourhood, ": New network created where whole network is visible to agents")
        self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.out_degree, self.distributionType, self.mu, self.sigma)
      # else:
        # print(self.neighbourhood, ": Using the existing network but only the neighbourhood is visible to agents")

    elif self.run == RunType.NetworkComparison.value:
      # We want to create a new directed network
      if self.networkType == NetworkType.Directed.value:
        # print(self.networkType, ": Directed network created")
        self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.out_degree, self.distributionType, self.mu, self.sigma)

      # Convert previously used directed network to an undirected network
      else:
        # print(self.networkType, ": Network converted to undirected network")
        self.networkData.convertNetwork()

    elif self.run == RunType.SigmaComparison.value:
      self.networkData.generateNewThresholds(self.distributionType, self.mu, self.sigma)

    else:
      self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.out_degree, self.distributionType, self.mu, self.sigma)

    G = self.networkData.network

    # Create agent thresholds.
    thresholds = self.networkData.thresholds
    return G, thresholds

  def getNetworkType(self):
    """

    :return:
    """
    return self.networkType

  def generateAgents(self):
    """

    """
    # Create agents
    for node in list(self.G.nodes()):
      agent = NeighbourhoodAgent(node, self, self.neighbourhood, State.DEFECT, self.thresholds[node])
      self.schedule.add(agent)
      self.grid.place_agent(agent, node)

  def step(self):
    """
    A single step of the model.
    The step function of all the agents is activated in the order
    specified by the scheduler and data is collected by the DataCollector.
    """
    self.datacollector.collect(self)
    self.schedule.step()

    # Stop the model if all agents are cooperating
    if number_cooperating(self) == self.cooperating:
      self.datacollector.collect(self)
      self.running = False

    self.cooperating = number_cooperating(self)
