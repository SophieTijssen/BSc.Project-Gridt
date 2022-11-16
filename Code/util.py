from enum import Enum
from random import sample


class State(Enum):
  DEFECT = 0
  COOPERATE = 1


def number_state(model, state):
  return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_cooperating(model):
  return number_state(model, State.COOPERATE)


def number_defecting(model):
  return number_state(model, State.DEFECT)


def get_cooperating_ratio(model):
  cooperating = number_cooperating(model)
  total_agents = len(model.grid.get_all_cell_contents())
  cooperating_rate = cooperating/total_agents
  return cooperating_rate


# https://stackoverflow.com/a/3590105
def constrained_sum_sample_pos(n, total):
  """Return a randomly chosen list of n positive integers summing to total.
  Each such list is equally likely to occur."""

  dividers = sorted(sample(range(1, total), n - 1))
  return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
