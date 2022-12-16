from mesa import Model, DataCollector
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
import networkx as nx

from agent import *
from util import *


class GranovetterModel(Model):

  def __init__(self, num_of_nodes, distribution, mu, sigma, in_degree):
    """
    Initialisation of the model
    """
    super().__init__()

    # Initialization
    self.num_of_nodes = num_of_nodes
    self.seed = 13648
    self.schedule = SimultaneousActivation(self)
    self.cooperating = 0.0
    self.running = True

    self.datacollector = DataCollector(
      model_reporters={"engagement_ratio": get_engagement_ratio},
      agent_reporters={"state": "state.value"}
    )

    # Create agent thresholds.
    # if distribution == Distribution.NORMAL:
    if distribution == 0:  # We use a normal distribution.
      self.thresholds = self.createThresholds(mu, sigma)

    else:  # We use a (modified) uniform distribution.
      self.thresholds = np.arange(0.0, 1.0, (1.0 / self.num_of_nodes))

      if distribution == 2:
        self.thresholds[self.thresholds == 0.01] = 0.02

    # Create network (and grid) with set in-degree and random out-degree.
    in_degree_list = [in_degree] * num_of_nodes
    out_degree_list = constrained_sum_sample_pos(num_of_nodes, sum(in_degree_list))

    self.G = nx.directed_configuration_model(
      in_degree_sequence=in_degree_list,
      out_degree_sequence=out_degree_list,
      create_using=nx.DiGraph,
      # seed=self.seed
    )
    self.G.remove_edges_from(nx.selfloop_edges(self.G))

    self.grid = NetworkGrid(self.G)

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
    if number_cooperating(self) == self.schedule.get_agent_count():
      self.running = False
    elif number_cooperating(self) == self.cooperating:
      self.running = False

    self.cooperating = number_cooperating(self)

  def createThresholds(self, mu, sigma):
    """
    Sample agent thresholds from a normal distribution.

    :param mu: Mean value of the normal distribution
    :param sigma: Standard deviation of the normal distribution
    :return: Numpy array of thresholds
    """
    thresholds = np.random.normal(mu, sigma, self.num_of_nodes)

    thresholds[thresholds > 1.0] = 1.0
    thresholds[thresholds < 0.0] = 0.0

    return thresholds
