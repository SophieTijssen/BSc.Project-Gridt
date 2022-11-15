from mesa import Model, DataCollector
from mesa.time import RandomActivation
from mesa.space import NetworkGrid

import networkx as nx

from agent import *
from util import *


class NetworkModel(Model):

  def __init__(self, num_of_steps=10, num_of_nodes=10, in_degree=3, out_degree=3, alpha=0.5, seed=13648, initial_following=1):

    super().__init__()

    self.num_of_steps = num_of_steps
    self.num_of_nodes = num_of_nodes
    self.in_degree = in_degree
    self.out_degree = out_degree
    self.alpha = alpha
    self.seed = seed  # Seed random number generators for reproducibility

    # self.G = nx.random_k_out_graph(n=num_of_nodes, k=out_degree, alpha=alpha, self_loops=False, seed=seed)

    in_degree_list = [in_degree for i in range(num_of_nodes)]
    # variabele out degree (som moet wel koppen)
    out_degree_list = [out_degree for i in range(num_of_nodes)]
    # in_degree_list = [in_degree] * num_of_nodes
    # out_degree_list = [out_degree] * num_of_nodes
    self.G = nx.directed_configuration_model(
      in_degree_sequence=in_degree_list,
      out_degree_sequence=out_degree_list,
      create_using=nx.DiGraph,
      seed=seed
    )

    self.G.remove_edges_from(nx.selfloop_edges(self.G))

    self.grid = NetworkGrid(self.G)
    self.schedule = RandomActivation(self)
    self.initial_following = (
      initial_following if initial_following <= num_of_nodes else num_of_nodes
    )

    self.datacollector = DataCollector(
      {
        "Cooperate": number_cooperating,
        "Defect": number_defecting,
      }
    )

    # Create agents
    for i, node in enumerate(self.G.nodes()):
      a = NetworkAgent(
        i,
        self,
        State.DEFECT,
        self.out_degree,
      )
      self.schedule.add(a)
      # Add the agent to the node
      self.grid.place_agent(a, node)

    infected_nodes = self.random.sample(list(self.G), self.initial_following)
    for a in self.grid.get_cell_list_contents(infected_nodes):
      a.state = State.COOPERATE

  def step(self):
    self.schedule.step()
    # collect data
    self.datacollector.collect(self)

  def run_model(self):
    for i in range(self.num_of_steps):
      self.step()
