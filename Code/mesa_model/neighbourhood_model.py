from mesa import DataCollector
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid

from mesa_model.granovetter_model import GranovetterModel
from mesa_model.neighbourhood_agent import NeighbourhoodAgent
from mesa_model.utility_agent import UtilityAgent
from utilities.model_util import *
from utilities.network_util import *


class NeighbourhoodModel(GranovetterModel):

  def __init__(self, run, num_of_nodes, networkType, neighbourhood, utility, distributionType, mu, sigma, in_degree, networkData):
    """
    Initialisation of the model.

    :param run:
    :param num_of_nodes: Number of agents (or nodes) in the network.
    :param networkType: The type of network used for the model (directed/undirected).
    :param neighbourhood: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
    :param utility:
    :param distributionType: The type of distribution used to sample the agent thresholds.
    :param mu: The mean of the threshold distribution.
    :param sigma: The standard deviation of the threshold distribution.
    :param in_degree: The in-degree of each node in the network.
    :param networkData: A class containing data about the network and agent thresholds.
    """

    # Initialization
    # self.seed = 13648  # TODO: Use random seed?? Or use same seed for every iteration?
    # self.schedule = SimultaneousActivation(self)
    # self.cooperating = 0.0
    # self.running = True

    self.run = run
    self.neighbourhood = neighbourhood
    self.utility = utility
    self.networkData = networkData

    super().__init__(num_of_nodes, networkType, distributionType, mu, sigma, in_degree)

    # self.G, self.thresholds = self.generateNetwork()
    # self.grid = NetworkGrid(self.G)

    check = []
    for agent in self.schedule.agents:
      neighbourhood = self.grid.get_neighbors(agent.unique_id, include_center=False)
      print(self.networkType, ":", "agent", agent.unique_id, "=", neighbourhood)

      # if self.networkType == NetworkType.UNDIRECTED.value:
      #   for neighbour in neighbourhood:
      #     if agent.unique_id not in self.grid.get_neighbors(neighbour, include_center=False):
      #       print('wrong conversion. agent:', agent.unique_id, ', neighbour:', neighbour)
      #       check.append(False)
      #     else:
      #       check.append(True)

    # if self.networkType == NetworkType.UNDIRECTED.value:
    #   print("correct conversion:", all(check))

    #
    # self.datacollector = DataCollector(
    #   model_reporters={"engagement_ratio": get_engagement_ratio},
    #   agent_reporters={"state": "state.value"}
    # )

  def generateNetwork(self):
    # Create network (and grid) with set in-degree and random out-degree.
    # print("weird hierarchy stuff works")
    if self.networkData.network is None:
      # We do not have a previously used network
      print("First network created")
      self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.in_degree, self.distributionType, self.mu, self.sigma)

    elif self.run == RunType.Neighbourhood.value:
      # We want to create a new network for each iteration (when neighbourhood is False)
      if not self.neighbourhood:
        print(self.neighbourhood, ": New network created where whole network is visible to agents")
        self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.in_degree, self.distributionType, self.mu, self.sigma)
      else:
        print(self.neighbourhood, ": Using the existing network but only the neighbourhood is visible to agents")

    elif self.run == RunType.Utility.value:
      if not self.utility:
        print(self.neighbourhood, ": New network created where we want to use utility")
        self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.in_degree, self.distributionType, self.mu, self.sigma)
      else:
        print(self.neighbourhood, ": Using the existing network but only the neighbourhood is visible to agents")

    elif self.run == RunType.NetworkComparison.value:
      if self.networkType == NetworkType.DIRECTED.value:
        # We want to create a new directed network
        print(self.networkType, ": Directed network created")
        self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.in_degree, self.distributionType, self.mu, self.sigma)
      else:
        # Convert previously used directed network to an undirected network
        print(self.networkType, ": Network converted to undirected network")
        self.networkData.convertNetwork()

    else:
      self.networkData.createNewNetwork(self.networkType, self.num_of_nodes, self.in_degree, self.distributionType, self.mu, self.sigma)

    G = self.networkData.network

    # Create agent thresholds.
    thresholds = self.networkData.thresholds
    return G, thresholds

  def getNetworkType(self):
    return self.networkType

  def generateAgents(self):
    # Create agents
    for node in list(self.G.nodes()):
      # if utility:
      #   agent = UtilityAgent(node, self, neighbourhood, State.DEFECT, self.thresholds[node])
      # else:
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
    # TODO:
    # if number_cooperating(self) == self.schedule.get_agent_count():
    #   self.datacollector.collect(self)
    #   self.running = False
    # print("", number_cooperating(self))
    if number_cooperating(self) == self.cooperating:
      # if number_cooperating(self) != 100:
        # print(self.schedule.agents[0].getState())
        # agents = [agent for agent in self.schedule.agents if agent.getState() is not State.COOPERATE]
        # print([agent.getThreshold() for agent in agents])
      self.datacollector.collect(self)
      self.running = False

    self.cooperating = number_cooperating(self)
