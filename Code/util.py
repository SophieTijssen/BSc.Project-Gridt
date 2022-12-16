from enum import Enum
from random import sample
import numpy as np


class RunType(Enum):
  """
  Enumerate class for the possible run types.
  """
  Granovetter = 0
  Single = 1
  Batch = 2


class State(Enum):
  """
  Enumerate class for the different states agents can have.
  """
  DEFECT = 0
  COOPERATE = 1


class Distribution(Enum):
  """
  Enumerate class for the different distributions
  that can be used to sample the agent thresholds.
  """
  NORMAL = 0
  UNIFORM = 1
  UNIFORM_MODIFIED = 2


# class Thresholds:
#   def __init__(self, n, mu, sigmas):
#     self.thresholds = {}
#     self.create_thresholds(n, mu, sigmas)
#
#   def create_thresholds(self, n, mu, sigmas):
#     for sigma in sigmas:
#       threshold = np.random.normal(mu, sigma, n)
#
#       threshold[threshold > 1.0] = 1.0
#       threshold[threshold < 0.0] = 0.0
#
#       self.thresholds[sigma] = threshold
#
#   def getThresholds(self):
#     return self.thresholds
#
#   def setThresholds(self, sigma, thresholds):
#     self.thresholds = {sigma: thresholds}


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


def constrained_sum_sample_pos(n, total):
  """
  Return a randomly chosen list of n positive integers summing to total.
  Each such list is equally likely to occur.

  source: https://stackoverflow.com/a/3590105
  """

  dividers = sorted(sample(range(1, total), n - 1))
  return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
