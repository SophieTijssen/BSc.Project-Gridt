from random import random
from mesa import Agent

from util import *


class NetworkAgent(Agent):
  def __init__(self, unique_id, model, initial_state, out_degree):

    super().__init__(unique_id, model)

    self.state = initial_state
    self.out_degree = out_degree
    self.cooperate_threshold = random()

  def step(self):
    neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
    cooperating_neighbors = [
      agent
      for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
      if agent.state is State.COOPERATE
    ]
    if len(cooperating_neighbors)/len(neighbors_nodes) + random() > self.cooperate_threshold:
      self.state = State.COOPERATE
    else:
      self.state = State.DEFECT
