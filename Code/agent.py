from mesa import Agent

from util import *


class GranovetterAgent(Agent):
  def __init__(self, unique_id, model, initial_state, cooperate_threshold):
    super().__init__(unique_id, model)

    self.cooperate_threshold = cooperate_threshold
    self.state = initial_state
    self.cooperating = False

  def step(self):
    # Only include this statement if we want to let the agents stop deciding once they have cooperated
    if self.state == State.COOPERATE:
      return

    # We let every agent be aware of the whole network
    total_agents = self.model.schedule.agents
    cooperating_rate = number_cooperating(self.model) / (len(total_agents) - 1)

    # We let an agent only be aware of the agents they are linked with
    # neighbourhood = self.model.grid.get_neighbors(self.pos, include_center=False)
    #
    # if len(neighbourhood) > 0:
    #   cooperating_neighbours = len([neighbour for neighbour in self.model.grid.get_cell_list_contents(neighbourhood) if neighbour.state == State.COOPERATE])
    #   cooperating_rate = cooperating_neighbours/len(neighbourhood)
    # else:
    #   cooperating_rate = 0.0

    if cooperating_rate >= self.cooperate_threshold:
      self.cooperating = True
    # Only include the following statements if we want the agents to be able to change their decision to cooperate
    # else:
    #   self.cooperating = False

  def advance(self):
    if self.cooperating and self.state == State.DEFECT:
      self.state = State.COOPERATE
    # Only include the following statements if we want the agents to be able to change their decision to cooperate
    # elif not self.cooperating and self.state == State.COOPERATE:
    #   self.state = State.DEFECT
