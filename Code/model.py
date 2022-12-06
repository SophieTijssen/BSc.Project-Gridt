from mesa import Model, DataCollector
# TODO: Will we use RandomActivation or SimultaneousActivation
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid

import networkx as nx

from agent import *
from util import *


class GranovetterModel(Model):

  # def __init__(self, num_of_nodes=10, mu=0.25, sigma=0.1, in_degree=3):
  def __init__(self, num_of_nodes, distribution, mu, sigma, in_degree):
    super().__init__()

    # Initialization
    self.num_of_nodes = num_of_nodes
    self.schedule = SimultaneousActivation(self)
    self.seed = 13648

    if distribution == Distribution.NORMAL:
      self.thresholds = self.createThresholds(mu, sigma)

    else:  # distribution == Distribution.UNIFORM or distribution == Distribution.UNIFORM2
      self.thresholds = np.arange(0.0, 1.0, (1.0 / self.num_of_nodes))

      if distribution == Distribution.UNIFORM2:
        self.thresholds[self.thresholds == 0.01] = 0.02

    self.cooperating = 0.0

    # Create Network
    in_degree_list = [in_degree] * num_of_nodes
    # Random out-degree for each node (but sum of out-degrees = sum of in-degrees)
    out_degree_list = constrained_sum_sample_pos(num_of_nodes, sum(in_degree_list))

    self.G = nx.directed_configuration_model(
      in_degree_sequence=in_degree_list,
      out_degree_sequence=out_degree_list,
      create_using=nx.DiGraph,
      seed=self.seed
    )
    self.G.remove_edges_from(nx.selfloop_edges(self.G))

    self.grid = NetworkGrid(self.G)

    # Create agents
    for node in list(self.G.nodes()):
      agent = GranovetterAgent(node, self, State.DEFECT, self.thresholds[node])
      self.schedule.add(agent)
      self.grid.place_agent(agent, node)

    # infected_nodes = self.random.sample(list(self.G), self.initial_following)
    # for a in self.grid.get_cell_list_contents(infected_nodes):
    #   a.state = State.COOPERATE

    self.datacollector = DataCollector(
      model_reporters={"engagement_ratio": get_engagement_ratio},
      agent_reporters={"state": "state.value"}
    )

    self.running = True

  def step(self):
    self.datacollector.collect(self)
    self.schedule.step()

    # Stop the model if all agents are cooperating
    if number_cooperating(self) == self.schedule.get_agent_count():
      self.running = False
    elif number_cooperating(self) == self.cooperating:
      self.running = False

    self.cooperating = number_cooperating(self)

  def createThresholds(self, mu, sigma):
    thresholds = np.random.normal(mu, sigma, self.num_of_nodes)

    thresholds[thresholds > 1.0] = 1.0
    thresholds[thresholds < 0.0] = 0.0

    return thresholds
