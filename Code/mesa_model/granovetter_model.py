from mesa import Model, DataCollector
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid

from mesa_model.granovetter_agent import GranovetterAgent
from utilities.model_util import calculate_engagement_ratio, State, number_cooperating
from utilities.network_util import createDirectedNetwork, convertToUndirectedNetwork
from utilities.threshold_util import createThresholds


class GranovetterModel(Model):

  def __init__(self, num_of_nodes, networkType, distributionType, mu, sigma, out_degree):
    """
    Initialisation of the model.

    :param num_of_nodes: Number of agents (or nodes) in the network.
    :param networkType: The type of network used for the model (directed/undirected).
    :param distributionType: The type of distribution used to sample the agent thresholds.
    :param mu: The mean of the threshold distribution.
    :param sigma: The standard deviation of the threshold distribution.
    :param out_degree: The out-degree of each node in the network.
    """
    super().__init__()

    # Initialization
    self.num_of_nodes = num_of_nodes
    self.seed = 13648  # TODO: Use random seed?? Or use same seed for every iteration?
    self.schedule = SimultaneousActivation(self)
    self.cooperating = 0.0
    self.running = True

    self.distributionType = distributionType
    self.mu = mu
    self.sigma = sigma
    self.out_degree = out_degree
    self.networkType = networkType

    self.G, self.thresholds = self.generateNetwork()
    self.grid = NetworkGrid(self.G)

    self.generateAgents()

    self.datacollector = self.createDataCollector()

  def createDataCollector(self):
    datacollector = DataCollector(
      model_reporters={"engagement_ratio": calculate_engagement_ratio},
      agent_reporters={"state": "state.value"}
    )
    return datacollector

  def generateNetwork(self):
    # Create network (and grid) with set in-degree and random out-degree.
    G = createDirectedNetwork(self.num_of_nodes, self.out_degree)

    if self.networkType == 0:  # Use an undirected network
      G = convertToUndirectedNetwork(G)

    # Create agent thresholds.
    thresholds = createThresholds(self.distributionType, self.num_of_nodes, self.mu, self.sigma, None, None)

    return G, thresholds

  def generateAgents(self):
    # Create agents
    for node in list(self.G.nodes()):
      agent = GranovetterAgent(node, self, State.DEFECT, self.thresholds[node])
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
    # print(number_cooperating(self))
    # print(self.schedule.get_agent_count())
    if number_cooperating(self) == self.schedule.get_agent_count():
      self.running = False
    elif number_cooperating(self) == self.cooperating:
      self.running = False

    self.cooperating = number_cooperating(self)
