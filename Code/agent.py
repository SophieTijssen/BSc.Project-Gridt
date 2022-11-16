from mesa import Agent

# from model import NetworkModel
from util import *


class NetworkAgent(Agent):
  def __init__(self, unique_id, model, initial_state, cooperate_threshold):
    super().__init__(unique_id, model)

    self.cooperate_threshold = cooperate_threshold
    self.state = initial_state
    self.will_cooperate = False

  def step(self):
    # TODO: If we want to let the agents stop deciding once they have cooperated, include this statement
    # if self.state == State.COOPERATE:
    #   return

    neighbourhood = self.model.grid.get_neighbors(self.pos, include_center=False)

    if len(neighbourhood) > 0:
      cooperating_neighbours = len([neighbour for neighbour in self.model.grid.get_cell_list_contents(neighbourhood) if neighbour.state == State.COOPERATE])
      proportion_cooperating_neighbours = cooperating_neighbours/len(neighbourhood)
    else:
      proportion_cooperating_neighbours = 0

    if proportion_cooperating_neighbours > self.cooperate_threshold:
      self.will_cooperate = True
    # TODO: Only include the following statements if we want the agents to be able to change their decision to cooperate
    else:
      self.will_cooperate = False

  def advance(self):
    if self.will_cooperate and self.state == State.DEFECT:
      self.state = State.COOPERATE
    # TODO: Only include the following statements if we want the agents to be able to change their decision to cooperate
    elif not self.will_cooperate and self.state == State.COOPERATE:
      self.state = State.DEFECT
