from enum import Enum
import numpy as np


sigmas = np.delete(np.linspace(0.0, 1.0, 11).round(decimals=2), 0)
nums = range(10, 205, 5)
out_degrees = range(1, 11, 1)


class RunType(Enum):
  """
  Enumerate class for the possible runs types.
  """
  Granovetter = 0
  KnowledgeComparison = 1
  NetworkComparison = 2
  SigmaComparison = 3
  Single = 4
  Batch = 5


class State(Enum):
  """
  Enumerate class for the different states agents can have.
  """
  DEFECT = 0
  COOPERATE = 1


def number_state(model, state):
  """
  Count the total number of agents in a specific state in the network.
  :param model: The mesa model.
  :param state: The specific state.
  :return The total number of agents in the specific state.
  """

  return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_cooperating(model):
  """
  Count the number of cooperating agents using the number_state() function.
  :param model: The mesa model.
  :return The total number of cooperating agents.
  """

  return number_state(model, State.COOPERATE)


def number_defecting(model):
  """
  Count the number of defecting agents using the number_state() function.
  :param model: The mesa model.
  :return The total number of defecting agents.
  """

  return number_state(model, State.DEFECT)


def get_engagement_ratio(model):
  """
  Calculate the engagement ratio (the ratio of cooperating agents over all agents).
  :param model: The mesa model.
  :return The engagement ratio.
  """

  cooperating = number_cooperating(model)
  total_agents = len(model.grid.get_all_cell_contents())
  cooperating_rate = cooperating/total_agents
  return cooperating_rate
