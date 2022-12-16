from mesa import Agent

from util import *


class GranovetterAgent(Agent):

  def __init__(self, unique_id, model, initial_state, cooperate_threshold):
    """
    Initialisation of the agent.
    """
    super().__init__(unique_id, model)

    self.cooperate_threshold = cooperate_threshold
    self.state = initial_state
    self.cooperating = False

  def step(self):
    """
    Step function of the agent that allows the agent to decide whether to cooperate or not.
    """
    # Let the agent stop deciding if they have already made the decision to cooperate.
    if self.state == State.COOPERATE:
      return

    """ Include for Granovetter Model """
    total_agents = self.model.schedule.agents
    cooperating_rate = number_cooperating(self.model) / (len(total_agents) - 1)

    """ Include for Neighbourhood Model """
    # neighbourhood = self.model.grid.get_neighbors(self.pos, include_center=False)
    #
    # if len(neighbourhood) > 0:
    #   cooperating_neighbours = len([neighbour
    #                                 for neighbour in self.model.grid.get_cell_list_contents(neighbourhood)
    #                                 if neighbour.state == State.COOPERATE
    #                                 ])
    #   cooperating_rate = cooperating_neighbours / len(neighbourhood)
    # else:
    #   cooperating_rate = 0.0

    if cooperating_rate >= self.cooperate_threshold:
      self.cooperating = True

  def advance(self):
    """
    Publish the actions of all agents simultaneously.
    """
    if self.cooperating and self.state == State.DEFECT:
      self.state = State.COOPERATE
