from mesa import Agent

from utilities.model_util import *


class GranovetterAgent(Agent):

  def __init__(self, unique_id, model, neighbourhood, initial_state, cooperate_threshold):
    """
    Initialisation of the agent.

    :param unique_id: Unique integer of the agent.
    :param model: The mesa model in which the agent occurs.
    :param neighbourhood: Parameter that shows whether the agent can see the whole network or only its neighbourhood.
    :param initial_state: The initial state of the agent
    :param cooperate_threshold: The cooperating threshold of the agent.
    """
    super().__init__(unique_id, model)

    self.neighbourhood = neighbourhood
    self.state = initial_state
    self.cooperate_threshold = cooperate_threshold
    self.cooperating = False

  def step(self):
    """
    Step function of the agent that allows the agent to decide whether to cooperate or not.
    """
    # Let the agent stop deciding if they have already made the decision to cooperate.
    if self.state == State.COOPERATE:
      return

    if not self.neighbourhood:  # Let agents see whole network for Granovetter runs
      total_agents = self.model.schedule.agents
      cooperating_rate = number_cooperating(self.model) / (len(total_agents) - 1)

    else:  # Let agents only see their neighbourhood
      neighbourhood = self.model.grid.get_neighbors(self.pos, include_center=False)

      if len(neighbourhood) > 0:
        cooperating_neighbours = len([neighbour for neighbour in self.model.grid.get_cell_list_contents(neighbourhood)
                                      if neighbour.state == State.COOPERATE
                                      ])
        cooperating_rate = cooperating_neighbours / len(neighbourhood)
      else:
        cooperating_rate = 0.0

    if cooperating_rate >= self.cooperate_threshold:
      self.cooperating = True

  def advance(self):
    """
    Publish the actions of all agents simultaneously.
    """
    if self.cooperating and self.state == State.DEFECT:
      self.state = State.COOPERATE
