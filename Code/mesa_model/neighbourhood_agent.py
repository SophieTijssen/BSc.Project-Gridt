from mesa_model.granovetter_agent import GranovetterAgent
from utilities.model_util import State, number_cooperating


class NeighbourhoodAgent(GranovetterAgent):

  def __init__(self, unique_id, model, neighbourhood, initial_state, cooperate_threshold):
    """
    Initialisation of the agent.

    :param unique_id: Unique integer of the agent.
    :param model: The mesa model in which the agent occurs.
    :param neighbourhood: Parameter that shows whether the agent can see the whole network or only its neighbourhood.
    :param initial_state: The initial state of the agent
    :param cooperate_threshold: The cooperating threshold of the agent.
    """

    self.neighbourhood = neighbourhood
    self.cooperating = False

    super().__init__(unique_id, model, initial_state, cooperate_threshold)

  # def getState(self):
  #   """
  #
  #   :return:
  #   """
  #   return self.state
  #
  # def getThreshold(self):
  #   """
  #
  #   :return:
  #   """
  #   return self.cooperate_threshold

  def step(self):
    """
    Step function of the agent that allows the agent to decide whether to cooperate or not.
    """

    # Let the agent stop deciding if they have already made the decision to cooperate.
    if self.state == State.COOPERATE:
      return

    if not self.neighbourhood:  # Let agents see whole network for Granovetter runs
      total_agents = len(self.model.schedule.agents)
      cooperating_rate = number_cooperating(self.model) / (total_agents - 1)

    else:  # Let agents only see their neighbourhood
      neighbourhood = self.model.grid.get_neighbors(self.pos, include_center=False)
      # print(len(neighbourhood))

      if len(neighbourhood) > 0:
        cooperating_neighbours = len([neighbour for neighbour in self.model.grid.get_cell_list_contents(neighbourhood)
                                      if neighbour.state == State.COOPERATE
                                      ])
        cooperating_rate = cooperating_neighbours / len(neighbourhood)
      else:
        cooperating_rate = 0.0

    if cooperating_rate >= self.cooperate_threshold:
      self.cooperating = True
