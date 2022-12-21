from mesa import Model, DataCollector
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
import numpy as np

from mesa_model.agent import *
from utilities.model_util import *
from utilities.network_util import *
from utilities.threshold_util import createThresholds


class GranovetterModel(Model):

  def __init__(self, num_of_nodes, network, neighbourhood, distribution, mu, sigma, in_degree):
    """
    Initialisation of the model.

    :param num_of_nodes: Number of agents (or nodes) in the network.
    :param network: The type of network used for the model (directed/undirected).
    :param neighbourhood: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
    :param distribution: The type of distribution used to sample the agent thresholds.
    :param mu: The mean of the threshold distribution
    :param sigma: The standard deviation of the threshold distribution.
    :param in_degree: The in-degree of each node in the network.
    """
    super().__init__()

    # Initialization
    self.num_of_nodes = num_of_nodes
    self.seed = 13648  # TODO: Use random seed?? Or use same seed for every iteration?
    self.schedule = SimultaneousActivation(self)
    self.cooperating = 0.0
    self.running = True

    self.datacollector = DataCollector(
      model_reporters={"engagement_ratio": get_engagement_ratio},
      agent_reporters={"state": "state.value"}
    )

    # Create agent thresholds.
    if distribution == 0:  # We use a normal distribution.
      self.thresholds = createThresholds(num_of_nodes, mu, sigma)

    else:  # We use a (modified) uniform distribution.
      self.thresholds = np.arange(0.0, 1.0, (1.0 / self.num_of_nodes))

      if distribution == 2:
        self.thresholds[self.thresholds == 0.01] = 0.02

    # Create network (and grid) with set in-degree and random out-degree.
    G = createDirectedNetwork(self.num_of_nodes, in_degree)

    if network == 1:  # Use an undirected network
      self.G = convertToUndirectedNetwork(G)
    else:  # Use a directed network
      self.G = G

    self.grid = NetworkGrid(self.G)

    # Create agents
    for node in list(self.G.nodes()):
      agent = GranovetterAgent(node, self, neighbourhood, State.DEFECT, self.thresholds[node])
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
    if number_cooperating(self) == self.schedule.get_agent_count():
      self.running = False
    elif number_cooperating(self) == self.cooperating:
      self.running = False

    self.cooperating = number_cooperating(self)
